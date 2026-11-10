# 20. Engine loop and memory budgeting (D5)

Code: [engine.py](../../tinyserve/engine/engine.py),
[memory.py](../../tinyserve/engine/memory.py),
[scheduler.py](../../tinyserve/engine/scheduler.py).
Tests: [test_d5_engine.py](../../tests/test_d5_engine.py).
Notebook: [D5_engine.ipynb](../../notebooks/D5_engine.ipynb).

## From a scheduling decision to generated text

D4 makes a plan and reserves blocks. Its caller still has to supply token
IDs, execute the model and acknowledge success. D5 supplies that caller:
the **inference engine**. A request contains a prompt, sampling options,
token history and a finite output budget.

The engine owns one evaluated model, one managed paged KV pool and one
FCFS scheduler. It uses the same flattened model interface:

```text
input_ids: (N,)
positions: (N,)
sequence_ids: (N,)
logits: (N, vocabulary_size)
```

For each plan, it concatenates the scheduled slices from each request.
Positions start at that request's resident cache length. Sequence IDs
identify ownership; they are not token IDs or positions. The existing
paged attention backend prevents attention between unrelated requests.

The operation order is:

```text
schedule and reserve
    -> pack token IDs, positions and owners
    -> model forward and whole-batch KV commit
    -> route final logits rows and stage sampled tokens
    -> scheduler.complete_step()
    -> publish histories and release early-EOS requests
```

There is no model rewrite, HF forward inside our engine, server or
background worker. The C2/C3 public `generate()` paths remain supported.
Admission validation is shared with them.

## Processed tokens are not sampled outputs

Let the prompt length be `P` and the requested output length be `G > 0`.
The engine schedules `P + G - 1` model rows before any recomputation.
The final output has `P + G` token IDs.

For `P=3`, `G=3` and a large enough batch budget:

| Forward | Input token positions | Sampled output | Resident entries needed |
|---|---|---|---:|
| Prefill | 0, 1, 2 | First new token, position 3 | 3 |
| Decode | 3 | Second new token, position 4 | 4 |
| Decode | 4 | Third new token, position 5 | 5 |

The final sampled token is returned without another forward. It does not
need a KV entry. Therefore D4 receives `decode_steps=G-1`, not `G`.
The engine deliberately preserves C2's stricter admission rule:
`P + G <= max_position_embeddings`, even though the last output is not
processed. Oversized requests raise instead of silently shortening output.

Chunked prefill only samples when the chunk reaches the end of the prompt.
Recomputation never samples. Sampling a partial prompt or a rebuilt prefix
would produce extra outputs and consume extra random draws.

For `G=0`, the engine returns an independent prompt snapshot immediately.
It creates no scheduler request and reserves no blocks for that request.
Engine construction still allocates its shared pool.

## Continuous arrivals and cancellation

The minimal API is:

```python
from tinyserve.engine.engine import InferenceEngine

engine = InferenceEngine(model, kv_budget_bytes=8 * 1024**2,
                         block_size=16, max_batch_tokens=32)
engine.add_request(0, prompt_ids, max_new_tokens=8)
updates = engine.step()
engine.add_request(1, another_prompt_ids, max_new_tokens=4)
results = engine.run()
```

The byte budget is KV-only, not a total process-memory limit. The example
is not a universal safe configuration; a request must fit in the chosen
pool by itself.

`step()` returns immutable, cumulative `RequestOutput` snapshots for requests
that sampled during that step. A partial prompt or replay step can return
an empty tuple while work remains. Use `has_pending_requests`, not whether
the latest update tuple is empty, to decide when work is finished.

Each output contains the request ID, full token history, generated tokens
and `finish_reason`: `None`, `"length"`, `"eos"` or `"cancelled"`.
`get_output()` returns a snapshot at any time. `run()` drains finite queued
work and returns all retained results in arrival order. Use `step()` when
new requests must join between forwards.

EOS is included in the output. Prompt EOS does not stop a new turn.
An early EOS cancels the remaining scheduled count after the successful
batch is acknowledged, returning its blocks. EOS on the final scheduled
forward reports `"eos"` while D4 already regards the work as finished.

Cancellation accepts waiting, running or preempted requests between
completed steps. It retains tokens already emitted and releases resident
blocks. Terminal IDs cannot be reused, including zero-output requests.
IDs must fit the int64 sequence-ID representation.

The model must stay in eval mode on its original device and dtype.
The engine and scheduler exclusively own their cache. Do not reserve,
reset, release or commit its requests externally.

## Preemption keeps history, not random decisions

Use two requests with `P=2`, `G=4`, a three-block pool, two slots per
block and a three-row batch budget. Each needs five processed positions,
so either fits alone. Their combined block demand eventually exceeds
the pool.

The scheduler releases the newer request's K/V. Its history still contains
the prompt and the output already selected from that prompt. Later it
replays the two processed prompt tokens. The engine emits nothing during
replay. Its next decode processes the previously sampled output token,
then selects the next output.

As in D4's worked timeline, ten distinct model positions become twelve
executed rows because two rows are replayed. The engine still returns four
outputs per request, not five. Per-request `torch.Generator` instances
keep seeded sampling independent of interleaving. A shared generator or
the default global RNG intentionally depends on scheduling order.
Reproducibility is limited to the same device and software environment.

## Failure is not success

A later-layer forward failure leaves committed cache lengths unchanged.
Reservations and preemptions remain in effect. The same plan is pending
for a retry, and histories and RNG states have not advanced. Arrivals
and cancellation are blocked until that pending step succeeds.

After a successful model forward, K/V has already committed. The engine
therefore retains logits until sampling and acknowledgement succeed.
If sampling fails, it restores generator states, retains unchanged token
histories and propagates the error. Retrying sampling uses those saved
scores, not another forward against an already appended cache.

Invalid saved logits are not recoverable by blindly retrying. Discard the
engine and diagnose the model or numerical settings. The retry contract
requires the model's D2/D3 whole-forward commit behavior; arbitrary hooks
that raise after cache commit do not qualify as pre-commit failures.

This is single-threaded inference, not request isolation for an exposed
service. Timeouts, transport errors and asynchronous cancellation belong
to later serving work.

## KV memory: count KV heads, not query heads

One cached token stores keys and values at every layer:

```text
bytes_per_token = 2 * layers * KV_heads * head_dim * element_bytes
bytes_per_block = block_size * bytes_per_token
num_blocks      = kv_budget_bytes // bytes_per_block
allocated_bytes = num_blocks * bytes_per_block
unused_bytes    = kv_budget_bytes - allocated_bytes
```

TinyLlama has 22 layers, four KV heads and head dimension 64:

| Model dtype | Bytes per token | KiB per token | Bytes per 16-slot block |
|---|---:|---:|---:|
| FP32 | 45,056 | 44 | 720,896 |
| FP16 | 22,528 | 22 | 360,448 |

The 32 query heads do not multiply KV storage. GQA is why only four heads
are cached. The planner derives element size from the model's embedding
weights, matching the existing cache allocation mechanism.

An FP16 KV-only budget of 1 MiB fits two 16-slot blocks:
`1,048,576 // 360,448 = 2`. The allocation is 720,896 bytes and 327,680
budget bytes remain unused. Three blocks would exceed the budget.
The remainder is budget rounding, not unused slots inside a live block.
Within each request's last block, up to `block_size-1` slots may be unused.

`plan_kv_memory()` does not allocate tensors. It rejects budgets that cannot
fit even one block. The engine verifies that the resulting pool's actual
tensor bytes equal the plan. Releasing requests returns blocks for reuse;
it does not shrink the underlying K/V tensors or return their allocation
to the CUDA driver.

## Budgeting against a live GPU

Load model weights first. Then:

```python
engine = InferenceEngine.from_available_memory(
    model, reserve_bytes=1024**3, block_size=16, max_batch_tokens=32,
)
```

The class method queries CUDA driver free memory on the model's device
and subtracts the explicit reserve before rounding to blocks. CPU models
must use an explicit KV byte budget. The shown 1 GiB reserve is an example,
not a measured safety guarantee.

Weights and any live reference model already consume GPU memory. The
reserve must cover activations, full vocabulary logits, PyTorch gather
temporaries, allocator overhead and other users. Prefill token budget and
cached context length affect peak workspace. Driver-free memory also
excludes memory already reserved by PyTorch's allocator, making this a
conservative snapshot rather than an exact reusable-memory inventory.

The snapshot can become stale immediately. Correct block arithmetic does
not prove that a future forward cannot run out of memory. Do not silently
catch OOM and claim successful inference. Measure headroom on the actual
Colab workload and report the chosen settings.

## Tests and notebook evidence

The Colab tests cover byte arithmetic and allocation, dtype scaling,
chunked prompts, mixed arrivals, EOS, zero budgets, strict admission,
cancelled/preempted requests, lifetime IDs, seeded replay, pre-commit failure
retry and post-commit sampling failure. Real small Llama models check HF/C2
greedy parity on CPU and CUDA. FP16 logits are compared numerically with
an FP32 model loaded from the same rounded weights, using `rtol=atol=1e-2`
as in C3; FP16 is not required to be bit-equal to FP32.

The full-TinyLlama notebook run reported 7,328 mismatches among 64,000
logits under that threshold, with maximum absolute difference about 0.12117.
The follow-up diagnostic on that first two-row prefill showed zero
paged/no-cache FP16 differences, identical cross-precision errors in both
FP16 paths, and matching argmax choices in all three paths. Thus the
observed first-batch error is not introduced by paging. That diagnostic
alone did not validate subsequent decode steps or establish a full-model
error ceiling; the complete run is recorded below.

The notebook now uses [compare_precision_paths](../../tinyserve/probe.py)
to separate the contracts. Paged/no-cache FP16 scores retain the `1e-2`
relative/absolute check and must have equal argmax on every scheduled row.
Each sampled greedy choice must also match the rounded-weight FP32
reference on this workload. All logits must be finite.

FP16/FP32 maximum/mean score differences and historical threshold mismatch
counts remain recorded for every row, but the uncalibrated full-model
cross-precision score gate is removed, not increased to an arbitrary number.
This intentionally changes the validation contract: engine correctness
requires paging agreement and tested greedy parity, not a claim of full
FP16/FP32 logit equivalence. The original small-model numerical regression
test remains. Additional tests accept baseline precision drift but reject
paging errors, changed sampled choices and nonfinite scores.

The notebook uses real TinyLlama FP32 references and records every planned
row, output routing, preemption, retry and final block release. FP32 logits
use `rtol=1e-5`, `atol=1e-4`: the absolute allowance matches the previously
observed small-model-backend differences, and argmax parity is checked
separately. It records numerical ranges, not a single selected difference.

## Recorded Colab validation

The complete user-supplied run on November 10, 2026 at source `d013743`
passed 422 tests; one test requiring two CUDA devices was skipped.
The original T4 environment, timestamp and test output are archived in
[D5_engine.json](../../results/D5_engine.json).

Continuous arrival finished in seven steps with 18 successful model rows.
The three-block preemption case finished in eight steps with ten original
rows plus two replay rows. The failure/retry case finished in seven successful
steps with ten original rows plus three replay rows; failed partial execution
is not included in that successful-row count. Its failure snapshot retained
identical histories, progress, resident lengths and ownership before retry.

Across all 43 successful FP32 rows, no-cache/HF argmax checks passed.
Per-row maximum absolute HF differences ranged from zero to 3.6239624e-5.
Seeded output and RNG replay, EOS, cancellation, admission errors and exact
KV pool arithmetic passed. All demonstrated pools were fully released.

The FP16 case processed five rows and generated four tokens.
Paged/no-cache FP16 per-row maximum absolute differences ranged from zero
to 0.015625, passing the combined `rtol=atol=1e-2` rule. All four sampled
choices matched rounded-weight FP32. FP16/FP32 per-row maxima ranged from
0.00964546 to 0.12116957, retained as measurements rather than a score bound.
This validates the documented workload and contract, not universal FP16/FP32
logit equivalence or OOM safety.

The archived JSON was checked against the supplied text and its internal
invariants without running models locally. Existing notebook outputs were
not modified. Local syntax/JSON/whitespace checks remain static only.

## Three interview questions

1. Why does generating `G` tokens require only `G-1` decode forwards?
   The last prompt row predicts the first output. Each later forward
   processes the preceding output. The final prediction is returned
   without being processed.
2. Why must recomputation preserve token history and RNG state?
   It rebuilds K/V for already chosen tokens. Resampling would change the
   sequence and consume random draws twice.
3. Why is a correctly sized KV pool not a total-memory guarantee?
   The pool excludes weights and variable workspace. Free-memory readings
   can become stale, and temporary allocations can still exceed headroom.

Next: E1, streaming OpenAI-compatible serving driven from Colab.
Prefix sharing, copy-on-write, Triton and CUDA graphs remain deferred.
