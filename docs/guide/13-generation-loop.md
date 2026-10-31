# 13. The generation loop (C2)

Code: [generation.py](../../tinyserve/generation.py).
Tests: [test_c2_generation.py](../../tests/test_c2_generation.py).
Notebook: [C2_generation.ipynb](../../notebooks/C2_generation.ipynb).

## Grow the context

```
ids = copy(prompt)
repeat up to max_new_tokens:
    positions = 0..len(ids)-1
    logits = model(ids, positions, AttentionMetadata(positions))
    token = sample_token(logits[-1], sampling settings)
    append token to ids
    if token is EOS: stop
```

The model processes the growing prompt on every step. This is deliberately
no-cache: C3 will remove recomputation of earlier tokens.

If the prompt is "The capital of France is", the loop might append "Paris",
then ".", then EOS. Every choice uses a new model forward with the updated
context, not repeated sampling of one unchanged logits row.

## Output and stopping contract

`generate` returns a 1-D tensor containing **prompt plus generated tokens**,
including a generated EOS. It preserves input dtype/device and does not
modify the prompt. Slice `output[prompt.numel():]` for the continuation only.

- `max_new_tokens` counts new tokens, not prompt tokens.
- Zero returns an independent prompt copy without running the model.
- EOS comes from `model.config.eos_token_id`: integer, list/tuple or None.
- A token already in the prompt is not a stopping event. This matters for
  chat templates with EOS delimiters between prior conversation turns.
- Prompt length plus the entire requested budget must fit the context
  limit. Oversized requests raise **before forward**, even if generation
  might have stopped early. This is the user's chosen strict boundary rule.
- Eval mode is required. The loop disables gradients.

The sampler defaults to greedy. Temperature, top-k, top-p and a device-matched
generator enable reproducible sampling, with C1's conventions. The shared
option validator also runs for zero-token requests.

For CUDA sampling, use `torch.Generator(device=input_ids.device)` to select
the prompt's exact GPU. An unindexed `"cuda"` generator is also accepted when
the current CUDA device matches the prompt; a different GPU is rejected.

## Raw text versus a chat template

Raw text asks the model to continue a document. A chat template formats
system/user/assistant turns as expected during chat training. Use
`tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True)`
to include the correct special tokens once.

With `return_tensors="pt", return_dict=True`, select `["input_ids"][0]`
for the single prompt tensor. Indexing the container directly with `[0]`
can return a tokenizer `Encoding`, which has no tensor `.to()` method.

A chat template often improves answer style and EOS behavior, but it does
not guarantee EOS. The new-token budget is always a separate stopping bound.
`skip_special_tokens=True` hides EOS when decoding; the returned IDs still
contain it.

## Verification

Scripted test models expose each complete prefix, position tensor and chosen
token. Tests cover immediate EOS, multiple EOS IDs, prompt EOS, exact budgets,
context errors, immutability, seeded sampler agreement and invalid inputs.
Small real HF models check token-exact greedy `generate(use_cache=False)`
agreement, with both budget stopping and forced first-token EOS.

The notebook runs B8/C1/C2 tests, compares real TinyLlama greedy generation
against HF in FP32, then shows a seeded sampled answer using our model.
No equality is demanded for stochastic trajectories across differing
implementations or devices. Sampler probabilities already have C1 reference
checks.

Two FP32 model copies need about 8.8 GB in weight memory. Use a fresh Colab
T4 runtime, not several old model kernels. Sampling demos free the HF copy
after the greedy comparisons. There are no cache or speed claims yet.

Results, token IDs, stopping observations, test output and environment info
are saved in `results/C2_generation.json`. Runtime validation awaits Colab.

## Interview questions

1. **Why use only the last logits row?** It predicts the token after the
   entire current prefix; earlier rows predict tokens at earlier positions.
2. **EOS versus max_new_tokens?** EOS is a model-selected end marker; the
   token budget is an independent hard bound.
3. **Why is this loop wasteful?** It recomputes every earlier token on every
   step. Causality lets C3 reuse their keys and values instead.
