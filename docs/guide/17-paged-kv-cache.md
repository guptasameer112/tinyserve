# 17. Block tables, slot mapping and paged KV cache (D2)

Code: [paged_cache.py](../../tinyserve/paged_cache.py),
[attention.py](../../tinyserve/models/attention.py),
[llama.py](../../tinyserve/models/llama.py),
[probe.py](../../tinyserve/probe.py).
Tests: [test_d2_paged_cache.py](../../tests/test_d2_paged_cache.py).
Notebook: [D2_paged_cache.ipynb](../../notebooks/D2_paged_cache.ipynb).

## Why change the storage layout?

C3 reserved one contiguous token array per request. D1 showed how to mix
independent token rows in a forward. D2 combines independent cached requests
with a shared pool of fixed-size physical KV blocks.

A request's context remains in logical token order, even if its physical
blocks are scattered. Attention still sees the same keys, values and causal
positions. Paging changes addressing, not the attention equation.

The user approved manually assigned, disjoint blocks for D2. This lesson has
no free list, allocation policy, reference counts, shared-prefix blocks,
copy-on-write, scheduler or performance kernel. D3 will manage allocation.

## Logical position versus physical location

For block size B and token position p:

```
logical block = p // B
offset        = p % B
physical block = request_block_table[logical block]
physical slot  = physical block * B + offset
```

With B=2 and request 7's block table `[4, 1, 5]`:

| Logical position | Logical block | Offset | Physical block | Physical slot |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 8 |
| 1 | 0 | 1 | 4 | 9 |
| 2 | 1 | 0 | 1 | 2 |
| 3 | 1 | 1 | 1 | 3 |
| 4 | 2 | 0 | 5 | 10 |
| 5 | 2 | 1 | 5 | 11 |

Sorting the physical blocks would corrupt the logical order. Advancing to
position 2 crosses from physical block 4 to block 1, not necessarily the
next physical block. A scrambled-block test catches accidental contiguous
addressing assumptions.

RoPE uses logical position p, never the physical slot. Moving a block changes
its address, not a token's meaning or rotation. Keys remain stored after RoPE;
values are not rotated.

## Shared pool and per-request lengths

Each storage tensor has shape:

```
(layers, physical_blocks, block_size, KV_heads, head_dim)
```

It still stores KV heads, not expanded query heads. Total K/V allocation is:

```
2 * layers * physical_blocks * block_size * KV_heads * head_dim * dtype_bytes
```

TinyLlama costs 22 KiB per physical token slot in FP16 or 44 KiB in FP32.
The whole pool is preallocated in D2, including unassigned blocks. Assignment
does not allocate a new tensor. This lesson does not measure a memory saving.

Every request has an independent committed length. A token batch might contain:

```
request IDs:   [7, 9, 9, 9]
positions:     [6, 0, 1, 2]
```

Request 7 decodes one token after its six-token prefix; request 9 prefills
three tokens. Another request can be paused and omitted from the batch.
The flat row index is neither logical position nor physical address.

## Readable paged attention

For every layer:

1. Project new Q/K/V and apply RoPE with logical positions.
2. Use slot mapping to write new K/V into the shared pool.
3. Group new queries by request ownership.
4. Gather that request's visible logical prefix through its block table.
5. Build `key_position <= query_position` within that request.
6. Run the existing reference GQA attention.
7. Scatter outputs back into the original flat query-row order.

For a two-token chunk after length 3:

```
query position 3 sees keys 0,1,2,3
query position 4 sees keys 0,1,2,3,4
```

The first new query must not see the later token in its own chunk. No query
sees another request's blocks. Unwritten tail slots and unassigned blocks
never participate in the gathered prefix.

This PyTorch path gathers temporary contiguous K/V copies and expands KV
heads for reference GQA. It is not a fast paged-attention kernel. Mixed
prefill/decode works correctly here, but D2 makes no throughput claim.

## Commit, retry and reset

New positions must be consecutive from each active request's committed
length, in that request's appearance order. Interleaving request rows is
allowed; skipping, duplicating or restarting committed positions is not.

All layers write uncommitted append slots. The outer model advances all
active lengths only after the entire forward and logits succeed. A failure
in a later layer leaves lengths and committed-prefix bytes unchanged.
Retry overwrites the incomplete append slots.

`reset(request_id)` sets just that request's length to zero, without zeroing
or freeing its blocks. Its old contents are hidden and overwritten on reuse.
Other request lengths remain unchanged. Freeing blocks belongs to D3.

Do not share a cache between concurrent forwards, change weights underneath
it, or alter a committed prompt. Reset/recreate after weight changes; recreate
after moving or casting the model. The cache is bound to one model, dtype
and device and requires eval mode with gradients disabled.

## Validation and interface

`new_paged_cache(num_blocks, block_size, block_tables)` constructs the pool.
`AttentionMetadata(..., paged_cache=pool)` selects the gather path, using
explicit request IDs and positions. The three-argument model interface and
checkpoint parameter names remain unchanged.

Contiguous and paged caches are mutually exclusive. Existing no-cache packed
and C3 contiguous paths stay supported. The public single-request generator
still offers contiguous caching only; D2 demos use explicit cached forwards.

Tables are copied to immutable tuples and exposed read-only. Invalid block
IDs, repeated/shared blocks, unknown request IDs, missing layout tensors,
capacity overflow, invalid append positions and context overflow raise.
Per-request table capacity and model context length are separate limits.

No shared blocks are permitted, even for common prefixes. Without reference
counts and copy-on-write, allowing them could corrupt independent requests.

## Verification and status

Tests cover hand slot values, scrambled block sizes 1-4, MHA/GQA/MQA,
contiguous/HF parity, multi-token boundary crossing, mixed new-prefill/decode,
paused requests, interleaved output routing, reset, poisoned unused/other
blocks, failure/retry and explicit boundary errors.

The notebook compares ordered and scrambled pools against separate no-cache
ours/HF on real FP32 TinyLlama. Full-stack assertions use `rtol=1e-5,
atol=1e-4`, with independent all-row argmax checks. A small CUDA FP16 test
uses the same rounded weights in FP32 with `rtol=atol=1e-2`; it does not
require token-exact FP16 parity.

Local checks are static only. Save `D2_paged_cache.json` and notebook outputs.
Next: D3, block manager free list and reference counts.

### Recorded Colab run (2026-11-03)

The user's [D2_paged_cache.json](../../results/D2_paged_cache.json) records
a Tesla T4 at commit `5f14f9c`, with PyTorch `2.11.0+cu130` and Transformers
`5.22.0`. **197 B8/C2/C3/D1/D2 tests passed; one C2 test requiring two
CUDA devices was skipped.** Pytest returned zero with empty stderr.

Both ordered and scrambled pools passed the following stages:

| Stage | Active work | Committed lengths afterward: 0, 1, 2 |
|---|---|---|
| Initial prefill | Six prompt tokens each for requests 0 and 1 | 6, 6, 0 |
| Mixed | One decode token for 0, nine prompt tokens for 2; pause 1 | 7, 6, 9 |
| Reordered decode | One token each, in request order 2, 1, 0 | 8, 7, 10 |

Across 14 request comparisons (seven per physical layout), all tested logits
rows matched the no-cache/HF argmax choices. Paged/no-cache maximum differences
range from zero to `4.58e-5`; paged/HF differences range from `8.58e-6` to
`4.58e-5`, within the stated FP32 tolerance.

The seven-block, four-slot FP32 pool costs `1,261,568` bytes per layout.
The hand mapping gives physical slots `[8, 9, 2, 3, 10, 11, 4, 5, 0, 1]`
for the demonstrated two-block-table layout.

Overlapping block ownership and restarting a committed request's positions
raise the expected errors. Reset reuses pool storage and gives zero logits
difference against no-cache in the demonstrated comparison.

The original pasted JSON and environment metadata were preserved. Static
record checks verified slots, lengths, ranges and byte accounting; no local
runtime tests were run. These observations validate the tested gather path,
not allocation, shared blocks, scheduling or a performance kernel.

## Interview questions

1. **What does a block table translate?** A request's logical block number
   into a physical block ID; the within-block offset is unchanged.
2. **Why scramble the physical blocks in a test?** Correctness must not
   depend on physically contiguous or sorted storage.
3. **Why does D2 reject shared blocks?** Independent writes require ownership
   management and copy-on-write rules, which this lesson does not implement.
