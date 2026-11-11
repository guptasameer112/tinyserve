"""E2: instrumented, fixed-work direct engine versus cached HF generation."""
import math
import time

import torch
from transformers import GenerationConfig

from tinyserve.engine.block_manager import _positive_integer
from tinyserve.engine.engine import InferenceEngine
from tinyserve.engine.memory import plan_kv_memory
from tinyserve.generation import _validate_generation_request
from tinyserve.roofline import summarize


def metrics_from_trace(token_times_s, total_seconds):
    """Times are relative to one shared start; every row is one request."""
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("total_seconds must be finite and positive")
    if not token_times_s or not token_times_s[0]:
        raise ValueError("each trace needs at least one request and one output token")
    count = len(token_times_s[0])
    requests = []
    for row in token_times_s:
        if (len(row) != count or any(not math.isfinite(t) or t <= 0 for t in row)
                or any(right < left for left, right in zip(row, row[1:]))
                or row[-1] > total_seconds):
            raise ValueError("token times must be finite, positive, ordered and within total_seconds")
        ready_ms = [value * 1e3 for value in row]
        intervals = [(right - left) * 1e3 for left, right in zip(row, row[1:])]
        requests.append({
            "token_ready_ms": ready_ms,
            "ttft_ms": ready_ms[0],
            "latency_ms": ready_ms[-1],
            "inter_token_ms": intervals,
        })
    outputs = len(requests) * count
    return {
        "total_ms": total_seconds * 1e3,
        "output_tokens": outputs,
        "output_tokens_per_s": outputs / total_seconds,
        "requests": requests,
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _start(device):
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    return time.perf_counter(), baseline


def _finish(device, start, baseline, times, tokens):
    _sync(device)
    seconds = time.perf_counter() - start
    record = metrics_from_trace(times, seconds)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    record.update({
        "generated_token_ids": tokens,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": peak - baseline if baseline is not None else None,
    })
    return record


def _measure_tinyserve(model, input_ids, new_tokens, block_size, max_batch_tokens):
    batch, prompt = input_ids.shape
    blocks = batch * ((prompt + new_tokens - 1 + block_size - 1) // block_size)
    unit = plan_kv_memory(model, 10**9, block_size).bytes_per_block
    engine = InferenceEngine(model, blocks * unit, block_size, max_batch_tokens)
    times, tokens = [[] for _ in range(batch)], [[] for _ in range(batch)]
    try:
        start, baseline = _start(input_ids.device)
        for owner, row in enumerate(input_ids):
            engine.add_request(owner, row, max_new_tokens=new_tokens)
        while engine.has_pending_requests:
            updates = engine.step()
            _sync(input_ids.device)
            ready = time.perf_counter() - start
            for output in updates:
                owner = output.request_id
                if len(output.generated_tokens) != len(tokens[owner]) + 1:
                    raise RuntimeError("engine must publish exactly one new token per update")
                times[owner].append(ready)
                tokens[owner] = list(output.generated_tokens)
        record = _finish(input_ids.device, start, baseline, times, tokens)
        if any(len(row) != new_tokens for row in tokens):
            raise RuntimeError("engine did not produce the fixed output budget")
        if (engine.cache.num_free_blocks != blocks or engine.cache.block_tables
                or engine.cache.lengths):
            raise RuntimeError("engine benchmark leaked KV blocks")
        preemptions = sum(state.preemptions for state in engine.scheduler.requests.values())
        if preemptions:
            raise RuntimeError("no-pressure benchmark unexpectedly preempted a request")
        record.update({
            "kv_pool_bytes": engine.cache.allocated_bytes,
            "num_blocks": blocks,
            "final_free_blocks": engine.cache.num_free_blocks,
            "preemptions": preemptions,
        })
        return record
    finally:
        engine.close()


class _HFTrace:
    """HF calls put once for the prompt, then once per generated batch column."""
    def __init__(self, input_ids):
        self.device = input_ids.device
        self.prompt_shape = tuple(input_ids.shape)
        self.prompt_seen = False
        self.ended = False
        self.start = None
        self.times = [[] for _ in input_ids]
        self.tokens = [[] for _ in input_ids]

    def put(self, value):
        if self.ended or self.start is None:
            raise RuntimeError("HF streamer called outside an active generation")
        if not self.prompt_seen:
            if tuple(value.shape) != self.prompt_shape:
                raise RuntimeError("HF streamer must first receive the complete prompt batch")
            self.prompt_seen = True
            return
        if tuple(value.shape) != (len(self.times),):
            raise RuntimeError("HF streamer must emit one token per request")
        _sync(self.device)
        ready = time.perf_counter() - self.start
        for owner, token in enumerate(value.tolist()):
            self.times[owner].append(ready)
            self.tokens[owner].append(token)

    def end(self):
        self.ended = True


def _measure_hf(model, input_ids, new_tokens):
    trace = _HFTrace(input_ids)
    config = GenerationConfig(
        max_new_tokens=new_tokens, do_sample=False, use_cache=True,
        cache_implementation="dynamic", eos_token_id=None,
        pad_token_id=model.config.pad_token_id if model.config.pad_token_id is not None else 0,
    )
    trace.start, baseline = _start(input_ids.device)
    output = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        generation_config=config, use_model_defaults=False, eos_token_id=None,
        streamer=trace,
    )
    record = _finish(input_ids.device, trace.start, baseline, trace.times, trace.tokens)
    batch, prompt = input_ids.shape
    if (not trace.ended or tuple(output.shape) != (batch, prompt + new_tokens)
            or any(len(row) != new_tokens for row in trace.tokens)
            or not torch.equal(output[:, :prompt], input_ids)
            or output[:, prompt:].tolist() != trace.tokens):
        raise RuntimeError("HF output and streamer must agree on the fixed output budget")
    return record


def _summaries(records):
    result = {}
    for mode in ("tinyserve", "hf"):
        rows = [pair[mode] for pair in records]
        requests = [request for row in rows for request in row["requests"]]
        intervals = [value for request in requests for value in request["inter_token_ms"]]
        result[mode] = {
            "total_ms": summarize([row["total_ms"] for row in rows]),
            "output_tokens_per_s": summarize([row["output_tokens_per_s"] for row in rows]),
            "ttft_ms": summarize([request["ttft_ms"] for request in requests]),
            "request_latency_ms": summarize([request["latency_ms"] for request in requests]),
            "inter_token_ms": summarize(intervals) if intervals else None,
        }
        for field in ("baseline_allocated_bytes", "peak_allocated_bytes", "incremental_peak_allocated_bytes"):
            values = [row[field] for row in rows]
            result[mode][field] = summarize(values) if values[0] is not None else None
    return result


@torch.no_grad()
def run_case(model, reference, input_ids, new_tokens, trials=3, block_size=16, max_batch_tokens=128):
    """Simultaneous equal-length requests; no HTTP, padding, EOS or pool pressure.

    Includes admission/generate overhead and per-token synchronization. Engine
    pool setup is outside timing; HF DynamicCache allocation is inside generate.
    Never share these models with concurrent work while this benchmark runs.
    """
    new_tokens = _positive_integer("new_tokens", new_tokens)
    trials = _positive_integer("trials", trials)
    block_size = _positive_integer("block_size", block_size)
    max_batch_tokens = _positive_integer("max_batch_tokens", max_batch_tokens)
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or input_ids.shape[0] == 0 or input_ids.shape[1] == 0
            or input_ids.dtype != torch.int64):
        raise ValueError("input_ids must be a nonempty int64 (batch, prompt) tensor")
    weight, ref_weight = (current.get_input_embeddings().weight for current in (model, reference))
    if weight.device != ref_weight.device or weight.dtype != ref_weight.dtype:
        raise ValueError("both models must have the same device and dtype")
    if weight.shape != ref_weight.shape:
        raise ValueError("model embedding shapes must agree")
    for current in (model, reference):
        for row in input_ids:
            _validate_generation_request(current, row, new_tokens, 0.0, 0, 1.0, None)
    eos = model.config.eos_token_id
    records = []
    try:
        model.config.eos_token_id = None
        for trial in range(-1, trials):
            order = ("tinyserve", "hf") if trial % 2 == 0 or trial == -1 else ("hf", "tinyserve")
            pair = {"trial": trial, "order": list(order)}
            for mode in order:
                pair[mode] = (
                    _measure_tinyserve(model, input_ids, new_tokens, block_size, max_batch_tokens)
                    if mode == "tinyserve" else _measure_hf(reference, input_ids, new_tokens)
                )
            if trial >= 0:
                pair["token_ids_equal_by_request"] = [
                    ours == theirs for ours, theirs in zip(
                        pair["tinyserve"]["generated_token_ids"], pair["hf"]["generated_token_ids"],
                    )
                ]
                pair["hf_over_tinyserve_total_time"] = pair["hf"]["total_ms"] / pair["tinyserve"]["total_ms"]
                records.append(pair)
    finally:
        model.config.eos_token_id = eos
    return {
        "batch": input_ids.shape[0], "prompt_tokens": input_ids.shape[1],
        "prompt_token_ids": input_ids.tolist(), "new_tokens": new_tokens, "trials": trials,
        "dtype": str(weight.dtype), "device": str(input_ids.device),
        "hf_attention_implementation": getattr(reference.config, "_attn_implementation", None),
        "hf_cache": "dynamic", "tinyserve_attention": "PyTorch paged gather",
        "block_size": block_size, "max_batch_tokens": max_batch_tokens,
        "eos_disabled_for_benchmark": True, "original_eos_token_id": eos,
        "warmup_runs_per_mode": 1,
        "timing": "synchronized perf_counter; common start before admission/generate; token IDs ready; no HTTP or text decoding",
        "pool_setup_excluded": True, "hf_dynamic_cache_setup_included": True,
        "summaries": _summaries(records),
        "hf_over_tinyserve_total_time": summarize([pair["hf_over_tinyserve_total_time"] for pair in records]),
        "token_ids_equal_in_all_trials": all(all(pair["token_ids_equal_by_request"]) for pair in records),
        "trial_records": records,
    }
