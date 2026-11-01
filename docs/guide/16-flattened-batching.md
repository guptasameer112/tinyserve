# 16. Flattened batching and attention metadata (D1)

Code: [batching.py](../../tinyserve/batching.py),
[attention.py](../../tinyserve/models/attention.py),
[probe.py](../../tinyserve/probe.py).
Tests: [test_d1_batching.py](../../tests/test_d1_batching.py).
Notebook: [D1_flattened_batching.ipynb](../../notebooks/D1_flattened_batching.ipynb).

## From one request to several

C3/C4 generate one request at a time. D1 prepares the model to process several
independent prompts in one forward, without changing its flattened interface.
The model already accepted positions and request IDs in B4/B8; this lesson
adds a validated pack/split helper and makes the contract explicit.

This is **no-cache packed prefill**, not continuous batching, batched cached
generation or paged storage. The single-request generator remains unchanged.

## Three pieces of layout information

For request A with three tokens and request B with two:

```
flat index:    0   1   2   3   4
token IDs:    A0  A1  A2  B0  B1
positions:     0   1   2   0   1
sequence IDs:  0   0   0   1   1
lengths:      (3, 2)
```

- Flat index locates a row in the concatenated tensor.
- Position locates a token within its own request, for RoPE and causality.
- Sequence ID tells attention which request owns that row.
- Lengths route output rows back to requests, in input order.

Sequence IDs are batch-local labels, not token IDs or permanent scheduler
handles. Reordering requests can change labels without changing the requests.
Positions reset per prompt. Cached decode will later continue each request's
own positions, not the batch's flat index.

## Why resetting positions is not enough

The existing causal condition is:

```
key_position <= query_position
```

If A0 and B0 both have position zero, this condition lets A0 see B0.
The correct packed condition adds request ownership:

```
(key_sequence_id == query_sequence_id)
AND
(key_position <= query_position)
```

For lengths `(3, 2)`, the mask is:

```
              A0 A1 A2 B0 B1
query A0       1  0  0  0  0
query A1       1  1  0  0  0
query A2       1  1  1  0  0
query B0       0  0  0  1  0
query B1       0  0  0  1  1
```

This is block-diagonal causal attention. Each request has its own triangular
block, with zero visibility outside it. This mask isolates request data;
omitting ownership would allow one prompt to influence another's output.

## Packing is not padding

A padded batch often has shape `(requests, longest_length)`. With lengths
3 and 2, it reserves six token slots, including one padding slot.

Our packed IDs have shape `(sum(lengths),)`, here five real tokens. Hidden
states have shape `(5, hidden_size)`; logits have shape `(5, vocab_size)`.
All tokenwise projections, normalization and MLP work use real token rows.

However, our reference attention still materializes dense scores and masks
over the total token count. With N total tokens, it computes N-by-N scores
per query head, including cross-request pairs later masked out.

Do not call this a block-sparse kernel. Packing avoids padding rows, but this
reference implementation does not skip every masked attention multiplication.
No speedup is claimed in D1.

## Output routing and next tokens

`pack_prompts(prompts)` returns IDs, `AttentionMetadata` and a lengths tuple.
`split_packed(logits, lengths)` returns token-major views in request order.
The returned views share the packed tensor's storage; do not mutate them if
the packed result must remain unchanged.

For lengths `(3, 2)`:

```
request A logits = packed_logits[0:3]
request B logits = packed_logits[3:5]
A next token     = argmax(packed_logits[2])
B next token     = argmax(packed_logits[4])
```

Using only `packed_logits[-1]` would predict for B and omit A. Each request's
last row predicts its own next token. A single packed forward predicts one
next token per request; it is not a complete multi-request generation loop.

## Why the model need not be rewritten

Embedding, RMSNorm, projections and SwiGLU already operate on flattened token
rows. They do not mix independent requests. Attention is the operation that
mixes tokens; positions and sequence IDs control that mixing at every layer.

Correct packed outputs should match separate requests, within floating-point
tolerance. Changing one request's token content while retaining the same
layout must not change any other request's outputs. Permuting requests should
permute the routed outputs, with possible small changes from GEMM rounding.

All requests share this model's weights, device and dtype. The helper requires
nonempty integer prompts on one device with one integer dtype. It preserves
input tensors; the model checks vocabulary bounds.

The context limit applies to each request's length/positions, not the sum of
unrelated prompt lengths. Large total batches can still exceed available
memory because the reference attention is dense.

## Boundaries with C3 and D2

C3's contiguous cache tracks one committed length for one request. It cannot
represent several requests advancing independently; packed sequence IDs
still cause that cached path to raise explicitly.

D2 introduces per-request block tables and slot mappings. Later scheduling
decides which requests contribute prefill/decode tokens to a batch. D1 only
establishes the no-cache ownership, position and output-routing contract.

## Verification and status

Tests cover exact layout/mask values, round trips, dtype/device checks, output
views, last-row routing, invalid inputs, MHA/GQA/MQA, separate/HF full-model
parity, cross-request isolation, within-request causality, permutation,
individual context limits and continued rejection of packed contiguous cache.

The notebook demonstrates deliberately missing request IDs with tiny hand
values, then compares real FP32 TinyLlama packed logits with separate ours/HF
for every row. Full-stack assertions use B8's `rtol=1e-5, atol=1e-4`.
Argmax equality is required separately. A small CUDA FP16 test compares the
same rounded weights in FP32 with `rtol=atol=1e-2`, without token-exact claims.

**Runtime validation is pending the user's fresh Colab T4 run.**
Local checks are static only. Save `D1_flattened_batching.json` and the
notebook with outputs. This lesson makes no batching performance claim.

## Interview questions

1. **Why are positions and sequence IDs separate?** Positions control RoPE
   and causal order; IDs prevent attention between independent requests.
2. **Why not select only the final packed logits row?** That row belongs
   only to the last request. Each request needs its own last row.
3. **Does a dense block-diagonal mask skip cross-request computation?**
   No. It removes visibility, but our dense reference still computes masked scores.
