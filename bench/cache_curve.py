"""C4: fixed-length wall-clock comparison of our cached/no-cache generator."""
import math
import time
from numbers import Integral

import torch

from tinyserve.generation import generate
from tinyserve.roofline import summarize
from tinyserve.weights import kv_cache_bytes_per_token


def _timed_generate(model, input_ids, new_tokens, use_cache):
    device = input_ids.device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device) if cuda else None
    start = time.perf_counter()
    output = generate(model, input_ids, max_new_tokens=new_tokens, use_cache=use_cache)
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("generation timing must be finite and positive")
    if output.shape != (input_ids.numel() + new_tokens,):
        raise RuntimeError("fixed-length benchmark did not produce the requested token count")
    peak_bytes = torch.cuda.max_memory_allocated(device) if cuda else None
    return {
        "total_ms": seconds * 1e3,
        "new_tokens": new_tokens,
        "end_to_end_tokens_per_s": new_tokens / seconds,
        "generated_token_ids": output[input_ids.numel():].tolist(),
        "peak_allocated_bytes": peak_bytes,
        "incremental_peak_allocated_bytes": peak_bytes - baseline_bytes if cuda else None,
    }


@torch.no_grad()
def run_case(model, input_ids, new_tokens, trials=3):
    """One prompt/budget pair, using the public generator and a fresh cache per call.

    EOS stopping is temporarily disabled to keep work equal, then restored.
    One full-length warmup runs per mode. Trial order alternates to reduce
    drift bias. Output token equality is observed, not required in FP16.
    Do not share this benchmark model with concurrent requests.
    """
    for name, value in (("new_tokens", new_tokens), ("trials", trials)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    # Reuse C2's complete prompt/eval/EOS validation outside the measured region.
    generate(model, input_ids, max_new_tokens=0)
    if input_ids.numel() + new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus new_tokens exceeds the model context limit")
    weight = model.get_input_embeddings().weight
    eos = model.config.eos_token_id
    records = []
    try:
        model.config.eos_token_id = None
        for use_cache in (False, True):
            generate(model, input_ids, max_new_tokens=new_tokens, use_cache=use_cache)
        for trial in range(trials):
            order = (False, True) if trial % 2 == 0 else (True, False)
            pair = {"trial": trial, "order": ["cached" if mode else "no_cache" for mode in order]}
            for use_cache in order:
                mode = "cached" if use_cache else "no_cache"
                pair[mode] = _timed_generate(model, input_ids, new_tokens, use_cache)
            pair["token_ids_equal"] = (
                pair["no_cache"]["generated_token_ids"] == pair["cached"]["generated_token_ids"]
            )
            pair["speedup"] = pair["no_cache"]["total_ms"] / pair["cached"]["total_ms"]
            records.append(pair)
    finally:
        model.config.eos_token_id = eos

    modes = {}
    for mode in ("no_cache", "cached"):
        rows = [pair[mode] for pair in records]
        modes[mode] = {
            "total_ms": summarize([row["total_ms"] for row in rows]),
            "end_to_end_tokens_per_s": summarize([row["end_to_end_tokens_per_s"] for row in rows]),
            "peak_allocated_bytes": (
                summarize([row["peak_allocated_bytes"] for row in rows])
                if input_ids.device.type == "cuda" else None
            ),
            "incremental_peak_allocated_bytes": (
                summarize([row["incremental_peak_allocated_bytes"] for row in rows])
                if input_ids.device.type == "cuda" else None
            ),
        }
    return {
        "prompt_tokens": input_ids.numel(), "prompt_token_ids": input_ids.tolist(),
        "new_tokens": int(new_tokens), "trials": int(trials),
        "dtype": str(weight.dtype), "device": str(input_ids.device),
        "eos_disabled_for_benchmark": True, "original_eos_token_id": eos,
        "warmup_runs_per_mode": 1,
        "timing": "perf_counter around public generate, synchronized before/after on the input CUDA device",
        "cache_reserved_bytes": (
            kv_cache_bytes_per_token(model.config, weight.element_size()) * (input_ids.numel() + new_tokens)
        ),
        "no_cache": modes["no_cache"], "cached": modes["cached"],
        "speedup": summarize([pair["speedup"] for pair in records]),
        "token_ids_equal_in_all_trials": all(pair["token_ids_equal"] for pair in records),
        "trial_records": records,
    }
