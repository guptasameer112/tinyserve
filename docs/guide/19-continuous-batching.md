# 19. Continuous batching and recomputation preemption (D4)

Code: [scheduler.py](../../tinyserve/engine/scheduler.py),
[block_manager.py](../../tinyserve/engine/block_manager.py),
[paged_cache.py](../../tinyserve/paged_cache.py).
Tests: [test_d4_scheduler.py](../../tests/test_d4_scheduler.py).
Notebook: [D4_scheduler.ipynb](../../notebooks/D4_scheduler.ipynb).

## What is the scheduler deciding?

D3 answers, "Which physical blocks belong to this request?" D4 answers,
"Which requests should process tokens in the next forward?"

A scheduler step builds a **batch plan**. Each item identifies one request,
the logical positions to process, and whether that work is prefill, decode
or recomputation. The scheduler reserves the necessary cache capacity.
It does not run the model, select tokens or calculate logits.

The notebook executes each plan explicitly through the existing D2 parity
helper, then acknowledges the successful step. D5 will build the automatic
engine loop and size its memory pool. The public single-request generator
does not change in D4.

## Why continuous batching?

With static batching, a fixed group starts together. A short request may
finish while a long request still runs. If new arrivals must wait for the
whole group to finish, available batch capacity goes unused.

Continuous batching revisits membership between forwards. Finished requests
release their blocks. New requests can join the next plan while older ones
continue. A batch can mix one request's prompt processing with another
request's one-token append.

This implementation has a deliberately small policy:

- Arrival order sets priority: first come, first served (FCFS).
- `max_batch_tokens` limits the sum of new token rows in a forward, not the
  number of requests or their full cached lengths.
- Prefill consumes up to the remaining token budget. Large prompts can
  therefore span several forwards.
- Decode processes at most one new token per request per step.
- Each request appears at most once in a plan. Rebuilding a prefix and
  decoding its next token happen in separate steps.

Oldest-first priority is simple, not a latency or fairness optimum. A large
older prefill can consume the whole budget while newer work waits.

## Three meanings of "length"

Suppose a request processed two tokens, then lost its blocks to preemption:

| Quantity | Before preemption | After preemption |
|---|---:|---:|
| Logical progress: tokens already processed successfully | 2 | 2 |
| Resident cache length: valid token entries currently in K/V | 2 | No cache; rebuilding starts at 0 |
| Reserved physical capacity | 2 or more slots | 0 slots |

`RequestState.processed_tokens` preserves the logical progress.
`cache.lengths` describes only resident K/V. Neither is a count of references.

The caller retains token IDs outside the KV cache. Replaying those IDs
reconstructs the prefix without choosing new tokens or adding them to the
logical history again. D4's demos retain fixed tokenized prompt slices.

## First worked example: a new arrival joins an older request

Use five blocks with two slots each and a four-token batch budget:

- Request 0 has a three-token prompt and two one-token append steps.
- Request 1 has a two-token prompt and no append steps.
- Request 2 arrives after the first forward, with a three-token prompt.

Ranges below are logical positions; the end is exclusive.

| Step | Planned work | Token rows | Finished after success |
|---|---|---:|---|
| 0 | 0: prefill `[0,3)`; 1: prefill `[0,1)` | 4 | Nobody |
| 1 | 0: decode `[3,4)`; 1: prefill `[1,2)`; 2: prefill `[0,2)` | 4 | Request 1 |
| 2 | 0: decode `[4,5)`; 2: prefill `[2,3)` | 2 | Requests 0 and 2 |

Request 2 starts in step 1, before request 0 finishes. That changing
membership is continuous batching. Request 1's prompt is chunked because
step 0 has only one token of budget left.

## What if there are not enough blocks?

First, reject any request whose complete finite workload cannot fit in the
entire pool by itself, or exceeds the model context limit. This prevents a
request from waiting forever for impossible capacity.

For feasible requests, an older request may still need a block held by a
newer resident request. The scheduler can **preempt** the newer request:

1. Keep its logical progress and the caller's token history.
2. Release all of its resident blocks through the D3 cache.
3. Give the needed blocks to higher-priority work.
4. Later reserve space again and replay the missing prefix.

This is release-and-recompute, not a GPU-to-CPU swap. Recomputing costs model
work. Copying/swapping would need a different memory and transfer policy.

Victims are considered newest first, and only among requests newer than
the current candidate. Already selected work is never evicted. The scheduler
checks that enough space can be reclaimed before releasing any victim.
If an unselected lower-priority request cannot fit, it waits without evicting
anyone merely for an unsuccessful reservation.

## Second worked example: an actual preemption

Use only three blocks with two slots each and a three-token batch budget.
Requests 0 and 1 each have a two-token prompt and three append steps. Each
can fit alone: five tokens require three blocks. Together they eventually
need more blocks than the pool has.

| Step | Planned work | Explanation |
|---|---|---|
| 0 | 0: prefill two tokens; 1: prefill one | Budget is exhausted |
| 1 | 0: decode position 2; 1: finish prefill | All three blocks are assigned |
| 2 | 0: decode position 3 | 1 cannot get its second block yet |
| 3 | Preempt 1; 0: decode position 4 | 0 needs its third block to finish |
| 4 | 1: recompute positions 0 and 1 | 0 has released its blocks |
| 5 | 1: decode position 2 | Rebuilt prefix is ready |
| 6 | 1: decode position 3 | Continue normally |
| 7 | 1: decode position 4 | 1 finishes and releases its blocks |

At step 3, request 1 still has logical progress two, but no resident cache.
At step 4, the same first two tokens are processed again. Their replay does
not advance logical progress past two. Step 5 does new work and advances it
to three.

The workload contains ten distinct logical token positions across both
requests. The model processes twelve rows: ten original rows plus two
recomputed rows. Preemption enables progress in a small pool, but is not free.

## Execution, failure and cancellation

The caller uses this sequence:

```python
plan = scheduler.schedule()
# Execute every plan.work item in one successful paged forward.
scheduler.complete_step()
```

`schedule()` reserves space but does not commit token lengths.
`complete_step()` checks the entire batch's expected cache lengths before
updating any request progress. Finished requests release their blocks only
after that check succeeds.

If forward fails, do not acknowledge it. The model leaves resident committed
lengths unchanged, and the same pending plan is returned for retry. Reservations
and any preemptions already performed remain in effect. Retrying overwrites
incomplete append slots. This is not a transaction that rolls scheduling back.

Arrivals and cancellations are accepted only between completed steps.
Changing request membership while a plan is pending raises. Cancelling
waiting, running or preempted work makes it terminal and releases any
resident blocks. IDs cannot be reused during one scheduler instance.
Request snapshots and plans are immutable.

The scheduler owns an initially empty managed cache. Do not separately
reserve, reset, release or forward requests behind its back. One scheduler
is single-threaded and permits one pending batch at a time.

## Finite teaching workloads, not a serving API

`add_request(request_id, prompt_tokens, decode_steps=0)` registers work counts.
`decode_steps` means additional token positions to process after the prompt;
it is not a `max_new_tokens` sampling budget. Sampling one output from the
prompt does not itself append that output to the KV cache.

The caller supplies and retains actual token IDs, including any previously
chosen output tokens needed for recomputation. The D4 notebook supplies fixed
suffixes to keep policy and cache correctness separate from generation.
Token selection, EOS handling, streaming, a production request lifecycle and
automatic GPU memory sizing remain outside this lesson.

## Validation and status

Tests cover exact worked timelines, chunking, mixed batches, dynamic arrivals,
newest-first eviction, deferred requests, cancellation, input validation,
immutable metadata and failed-step acknowledgement. Twenty randomized traces
check bounded plans, exclusive ownership, monotonically increasing logical
progress and termination for finite admitted workloads.

Small CPU/CUDA model tests execute the preemption scenario against no-cache
ours/HF references, inject a later-layer failure, and retry. The notebook
repeats both worked timelines with real FP32 TinyLlama. Every new or replayed
logits row uses `rtol=1e-5, atol=1e-4` plus exact argmax checks.

Local checks are static only. D4 runtime validation is pending on a fresh
Colab T4. D3's complete real-model failure/retry evidence is now archived in
[D3_block_manager.json](../../results/D3_block_manager.json); it does not
count as D4 validation.
The notebook saves `D4_scheduler.json` with environment details, test output,
both complete timelines, preemption/recomputation records and numerical ranges.
It checks required scenario records before saving rather than relying only
on success flags. Preserve the successful notebook with outputs as well.

No throughput improvement, latency bound or GPU allocation reduction is claimed.
The cache pool remains fixed and attention still uses readable PyTorch gathers.

## Three interview questions

1. **What makes batching continuous?** Batch membership can change between
   forwards; new arrivals need not wait for an entire old group to finish.
2. **What survives recomputation preemption?** Logical progress and retained
   token IDs survive; resident K/V blocks are released and later rebuilt.
3. **Why might a smaller pool increase work?** Eviction can force prefix
   recomputation, adding token rows that an ample pool would not need.

Next: D5, the engine loop and memory budgeting.
