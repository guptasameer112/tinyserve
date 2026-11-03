# 18. Block ownership, reference counts and reuse (D3)

Code: [block_manager.py](../../tinyserve/engine/block_manager.py),
[paged_cache.py](../../tinyserve/paged_cache.py),
[llama.py](../../tinyserve/models/llama.py).
Tests: [test_d3_block_manager.py](../../tests/test_d3_block_manager.py).
Notebook: [D3_block_manager.ipynb](../../notebooks/D3_block_manager.ipynb).

## What changes after D2?

D2 translated logical positions through manually assigned, disjoint block
tables. D3 assigns those tables from a free list, grows them when a request
needs more capacity, and returns blocks when that request finishes.
Attention, logical RoPE positions and successful-forward length commits
remain the D2 implementation.

This is a single-threaded ownership layer, not a scheduler. It does not
choose which request runs, evict live requests, preempt, share model prefixes
or implement copy-on-write. The public generator still uses the C3 contiguous
cache. D2's manual `new_paged_cache(...)` interface remains supported.

## The allocator contract

`BlockAllocator(num_blocks)` starts with every physical block free.
`allocate(count)` returns distinct block IDs, each with reference count one.
It either satisfies the whole reservation or raises `BlockPoolExhausted`
without changing state. Zero/negative/noninteger sizes raise `ValueError`.

`retain(block)` adds a reference to a live block. `release(block)` removes
one; only the transition from one reference to zero returns it to the free
list. Invalid IDs, retaining a free block and releasing a free block raise.

For a three-block pool:

```
operation              references     next-allocation order
start                  [0, 0, 0]      [0, 1, 2]
allocate(2)            [1, 1, 0]      [2]
retain(0)              [2, 1, 0]      [2]
release(0)             [1, 1, 0]      [2]
release(0)             [0, 1, 0]      [0, 2]
allocate(1) -> (0,)    [1, 1, 0]      [2]
```

The free list is a stack: allocation initially visits ascending IDs, then
prefers recently released blocks. Correctness does not depend on that order.
The sum of references need not equal the number of live blocks.

At every operation:

```
free blocks = exactly the IDs whose reference count is zero
number of free blocks + number of live physical blocks = pool size
```

The allocator tests exercise counts above one. Managed model requests remain
disjoint, with exactly one reference per assigned block. Reference counts alone
do not make shared-prefix writes safe: an independently writable shared tail
would still need copy-on-write and compatible prefix metadata.

## Reservation is not committed length

Create a fixed tensor pool with
`model.new_managed_paged_cache(num_blocks, block_size)`. It starts without
requests. Before forwarding, call `pool.reserve(request_id, total_tokens)`.
The argument is the total processed-token capacity, not the append count.

For B=2, a request committed at length 3 already owns two blocks. Reserving
capacity 5 adds one block:

```
before: table [0, 1],    committed length 3, capacity 4
reserve(0, 5)
after:  table [0, 1, 3], committed length 3, capacity 6
successful append of positions [3, 4]
after:  table [0, 1, 3], committed length 5, capacity 6
```

Required blocks are `ceil(total_tokens / B)`. Existing blocks are retained;
reservation does not shrink a table. A reservation below the committed length
or above the model context limit raises. Tables and lengths are read-only
views; block lists are immutable tuples.

Admission and growth allocate all missing blocks before updating a table.
On exhaustion, no partial table, length or reference-count change is published.
Batch callers reserve requests explicitly; reservations are atomic per request,
not a transaction across a whole batch. Forward never allocates implicitly.

## Failure, reset and release

If a later decoder layer fails, append slots may contain incomplete K/V, but
the committed lengths and prefix K/V remain unchanged. Reserved blocks stay
owned for retry. Retry writes the same uncommitted positions again.

`reset(request_id)` hides the old prefix by setting its length to zero and
keeps all its blocks. `release_request(request_id)` removes its table and length,
then returns its blocks to the allocator. Unknown request IDs raise.

Release does not zero K/V bytes or shrink GPU tensor storage. A new request
can reuse those physical IDs because attention gathers only its written
logical prefix. The notebook and tests poison unused storage with NaN and
compare reused requests with fresh no-cache/HF forwards.

Do not use one pool concurrently, change weights under committed prefixes,
or externally mutate its tensors. Recreate after moving/casting the model.
This educational cache is not a secure memory-erasure mechanism.

## Memory and performance boundaries

Both K and V retain shape:

```
(layers, num_blocks, block_size, KV_heads, head_dim)
```

The whole pool is allocated once. Reserving and releasing blocks change
ownership, not GPU allocated bytes. TinyLlama uses 44 KiB per FP32 physical
token slot (22 KiB in FP16). D3 measures no memory saving or throughput gain.
The reference attention still gathers temporary K/V copies.

## Validation

Randomized property tests compare allocator reference counts with an independent
counter model across 20 seeds and 400 operations each. Ten additional seeded
200-operation traces check disjoint managed tables, growth and conservation.
Boundary tests check invalid input, exhaustion without partial allocation,
reset versus release, read-only metadata and fixed storage addresses.

Small CPU/CUDA model tests check mixed growth/reuse against separate FP32
ours/HF references, plus injected later-layer failure and retry. The Colab
notebook repeats lifecycle checks with real FP32 TinyLlama and saves
`D3_block_manager.json` through the existing environment recorder.
All-row logits use D2's `rtol=1e-5, atol=1e-4` tolerance, with separate exact
argmax checks; this is numerical correctness, not a timing benchmark.

Local validation is static only. Runtime tests and demos are **pending on a
fresh Colab T4**. Save both the JSON and successful notebook outputs before
replacing the runtime. Do not mark D3 complete until that evidence is available.

## Three interview questions

1. **Why reference counts, not just a free flag?** A block remains live until
   its final owner releases it; multiple leases cannot free it prematurely.
2. **Why separate reservation from committed length?** Capacity can grow
   before execution, while only a complete successful forward makes tokens
   visible. A failed forward can retry without exposing incomplete K/V.
3. **Does releasing a request free GPU memory?** It returns physical blocks
   for reuse within the fixed pool; it does not deallocate the tensor.

Next: D4, continuous batching and preemption policy.
