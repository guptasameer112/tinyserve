"""Lesson A3: baseline speed of Hugging Face `generate` (greedy, fixed lengths).

Definitions (they stay the same for our own engine in Part E):
- TTFT: time to first token = prefill of the prompt + producing one token.
- step latency: time per decode step = (time for N tokens - TTFT) / (N - 1). With a batch,
  every sequence gets one token per step, so this is also the inter-token latency.
- decode tokens/s: batch / step latency (all sequences together).
Timings use perf_counter around generate() plus a device sync, so they include Python overhead,
which is what a user actually experiences.
"""
import time

import torch

from tinyserve.roofline import summarize

FILLER_TEXT = " ".join(["The quick brown fox jumps over the lazy dog."] * 400)


def make_prompt_ids(tokenizer, length, batch, device):
    """(batch, length) ids, all rows identical. Exact length, no padding, so no attention-mask effects."""
    ids = tokenizer(FILLER_TEXT, return_tensors="pt").input_ids[0]
    if len(ids) < length:
        raise ValueError("filler text has only %d tokens, need %d" % (len(ids), length))
    return ids[:length][None, :].repeat(batch, 1).to(device)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def timed_generate(model, input_ids, new_tokens, pad_token_id, use_cache=True):
    """Exactly `new_tokens` greedy tokens per row (min_new_tokens stops early EOS). Returns (seconds, ids)."""
    device = input_ids.device
    _sync(device)
    start = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        use_cache=use_cache,
    )
    _sync(device)
    return time.perf_counter() - start, out


def measure_ttft_ms(model, tokenizer, batch, prompt_len, trials=5):
    """TTFT in ms per trial: prefill plus one generated token."""
    device = next(model.parameters()).device
    ids = make_prompt_ids(tokenizer, prompt_len, batch, device)
    timed_generate(model, ids, 1, tokenizer.eos_token_id)  # warmup
    return [timed_generate(model, ids, 1, tokenizer.eos_token_id)[0] * 1e3 for _ in range(trials)]


def run_case(model, tokenizer, batch, prompt_len, new_tokens, trials=5, use_cache=True):
    """One benchmark point. Each trial pairs a 1-token run (TTFT) with an N-token run."""
    if new_tokens < 2:
        raise ValueError("new_tokens must be >= 2 to separate prefill from decode")
    device = next(model.parameters()).device
    pad = tokenizer.eos_token_id
    ids = make_prompt_ids(tokenizer, prompt_len, batch, device)
    timed_generate(model, ids, min(8, new_tokens), pad, use_cache)  # warmup (kernels, allocator)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ttft_ms, step_ms, total_s = [], [], []
    for _ in range(trials):
        first_s, _ = timed_generate(model, ids, 1, pad, use_cache)
        full_s, out = timed_generate(model, ids, new_tokens, pad, use_cache)
        if tuple(out.shape) != (batch, prompt_len + new_tokens):
            raise RuntimeError("unexpected output shape %s" % (tuple(out.shape),))
        ttft_ms.append(first_s * 1e3)
        step_ms.append((full_s - first_s) / (new_tokens - 1) * 1e3)
        total_s.append(full_s)

    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "use_cache": use_cache,
        "ttft_ms": summarize(ttft_ms),
        "step_ms": summarize(step_ms),
        "decode_tokens_per_s": summarize([batch / (s * 1e-3) for s in step_ms]),
        "end_to_end_tokens_per_s": summarize([batch * new_tokens / s for s in total_s]),
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None,
    }
