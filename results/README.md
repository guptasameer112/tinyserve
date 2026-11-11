# Recorded lesson results

These JSON files contain saved Colab observations, not fresh local runs.
Each result includes the original UTC timestamp, GPU/library environment,
git SHA and lesson data.

## E2 status

[E2_engine_vs_hf.ipynb](../notebooks/E2_engine_vs_hf.ipynb) is prepared,
not runtime-validated. No E2 result is archived yet.
Its direct comparison saves tests, a short FP32 parity gate and four FP16
fixed-work cases, including all trials, token-ready timestamps, exact IDs,
TTFT/ITL/last-ID latency, throughput, memory observations and timing boundaries.
It uses no HTTP server and does not require a speedup or universal FP16 parity.
Archive the complete user-supplied `E2_engine_vs_hf.json` before changing status.

## E1 status

[E1_streaming_server.json](E1_streaming_server.json) was archived from the
complete user-supplied Colab output on November 11, 2026. The original timestamp
`2026-11-11T08:31:24Z`, T4 environment, source `cee6f49` and test log are preserved:
all 114 D5/E1 tests passed in 22.90 seconds, with no skips.

The FP32 live-loopback demo records health/model discovery, JSON/SSE text
and token histories matching direct generation, and eight forwards shared
by the two concurrent clients. Completed requests each generated eight tokens;
the streamed text arrived in three content chunks of 2-11 characters.
The disconnected request was cancelled after three tokens. Unsupported
options returned HTTP 400. All 16 blocks were released, block tables were
empty, and shutdown closed the engine.

The fixed KV pool held 11,534,336 bytes (11 MiB). The archive was checked
for byte equality with the supplied JSON after removing only Markdown fences.
Counts, schemas, parity, ownership, memory arithmetic and release were checked
without executing model code locally. Preserve the successful
[E1 notebook](../notebooks/E1_streaming_server.ipynb) outputs separately.
This validates the documented loopback-only subset, not public deployment,
full OpenAI compatibility or performance. E2 owns timing measurements.

## D5 status

[D5_engine.json](D5_engine.json) was archived from the complete user-supplied
Colab output on November 10, 2026. The original timestamp
`2026-11-10T09:35:14Z`, T4 environment, source `d013743` and test log are preserved:
422 tests passed and one two-GPU test was skipped.

The three FP32 scenarios record 18, 12 and 13 successful scheduled rows.
The latter two include two and three recomputed rows respectively.
All 43 rows passed reference argmax checks; HF absolute differences ranged
from zero to 3.6239624e-5. The injected failure snapshot is unchanged before
retry. Seeded outputs/RNG states, EOS, zero budgets, invalid admission,
cancellation, block conservation and final release are recorded.

Five FP16 rows passed same-precision paging checks, with per-row maximum
absolute differences ranging from zero to 0.015625 under the relative/absolute
tolerance. All four sampled greedy choices matched rounded-weight FP32.
FP16/FP32 per-row maxima ranged from 0.00964546 to 0.12116957; this run does
not claim a universal cross-precision score bound or logit equivalence.

The archive was checked for equality with the supplied JSON after removing
only its Markdown fence. Timelines, counts, pool bytes and recorded summaries
were checked without executing model code locally. Preserve the successful
[D5 notebook](../notebooks/D5_engine.ipynb) outputs separately.
Memory planning is not an OOM guarantee. E1's separate run is archived above.

## D4 status

[D4_scheduler.json](D4_scheduler.json) was archived from the user's complete
Colab output on November 4, 2026. It preserves timestamp
`2026-11-04T12:54:48Z`, source commit `47ea839`, T4 environment and test output.
The run records 304 passed B8/C2/C3/D1/D2/D3/D4 tests and one skipped two-GPU test.

Both scenarios contain their complete timelines: three continuous-batching
steps and eight preemption/recomputation steps. The latter records request 1
being preempted at logical progress two, rebuilding its two-token prefix, and
resuming decode. Its ten logical token positions require twelve model rows,
including two replay rows. Both pools finish with every block free.

Seventeen request comparisons cover 22 new/replayed logits rows, all with
matching no-cache/HF argmax choices. Numerical ranges, token budgets, retained
token histories, slot mappings, ownership snapshots and pool bytes were checked
against recorded fields without local model execution. The archived payload
equals the supplied JSON, with no regenerated timestamps or environment data.
Retain the successful [D4 notebook](../notebooks/D4_scheduler.ipynb) with outputs
separately; no notebook outputs were fabricated.

## D3 status

[D3_block_manager.json](D3_block_manager.json) was archived from the user's
complete Colab output on November 4, 2026. It preserves timestamp
`2026-11-04T05:02:37Z`, source commit `e176350`, T4 environment and test output.
The run records 253 passed B8/C2/C3/D1/D2/D3 tests and one skipped two-GPU test.

All four demo stages are present, including the previously missing failure
snapshot and boundary-crossing retry. Request 0 stayed at committed length
three after the injected error, then reached five after retry. Reused blocks,
mixed work, reset and final release are recorded. Six request comparisons
cover fourteen logits rows, all with equal argmax choices.

The JSON payload was checked for equality with the supplied text. State
conservation, slot mappings, pool bytes and numerical ranges were checked
without running models or tests locally. No notebook outputs were fabricated;
retain the successful [D3 notebook](../notebooks/D3_block_manager.ipynb) with
outputs separately.

## Recovered archive

The following files were recovered from complete JSON text printed in the
saved notebook outputs. The archive preserves the original payloads and
metadata; it does not regenerate results with today's environment.

| JSON | Source notebook |
|---|---|
| [B3_rope.json](B3_rope.json) | [B3_rope.ipynb](../notebooks/B3_rope.ipynb) |
| [B4_attention.json](B4_attention.json) | [B4_attention.ipynb](../notebooks/B4_attention.ipynb) |
| [B5_gqa.json](B5_gqa.json) | [B5_gqa.ipynb](../notebooks/B5_gqa.ipynb) |
| [B6_mlp.json](B6_mlp.json) | [B6_mlp.ipynb](../notebooks/B6_mlp.ipynb) |
| [B7_decoder_layer.json](B7_decoder_layer.json) | [B7_decoder_layer.ipynb](../notebooks/B7_decoder_layer.ipynb) |
| [B8_full_model.json](B8_full_model.json) | [B8_full_model.ipynb](../notebooks/B8_full_model.ipynb) |
| [C1_sampling.json](C1_sampling.json) | [C1_sampling.ipynb](../notebooks/C1_sampling.ipynb) |
| [C2_generation.json](C2_generation.json) | [C2_generation.ipynb](../notebooks/C2_generation.ipynb) |

Each recovered JSON was parsed and checked for equality with the notebook's
printed payload. Existing notebook outputs were not modified. B3's JSON
does not contain pytest stdout; its 16-test result is in the notebook.
Test counts for B4-C2 describe each notebook's selected tests, not independent
counts that can be added together.

## C3 status

[C3_kv_cache.json](C3_kv_cache.json) was archived from the user's pasted
Colab output on November 1, 2026. It preserves the original timestamp,
T4 environment and source commit `7b3969f`.

The result records 139 passed B8/C1/C2/C3 tests, one two-GPU test skipped,
20 greedy parity steps and successful reset, replay and boundary checks.
This is runtime evidence supplied by the user, not a local test run.
It does not claim that the repository notebook contains the complete
successful run's outputs. Preserve the successful Colab notebook separately.

Save both the JSON and notebook with outputs before replacing a Colab runtime.

## C4 status

[C4_cache_curve.json](C4_cache_curve.json) was archived from the user's complete
Colab result on November 1, 2026, preserving timestamp, T4 environment and
source commit `e50ae86`. It records 101 passed tests and one skipped test,
eight unique measured cases, and three paired trials per case.

Trial ordering, token counts, tokens/s, speedup ratios, summary ranges,
theoretical KV bytes and the shared sweep point were checked against the
raw records without running the benchmark locally.

The JSON reports both plot paths. Their PNG contents were not supplied in
the pasted result and are not archived here. Retain the Colab PNGs and the
successful notebook with outputs separately.

## D2 status

[D2_paged_cache.json](D2_paged_cache.json) was archived from the user's Colab
output on November 3, 2026. It preserves the original timestamp, T4 environment,
Transformers `5.22.0` and source commit `5f14f9c`.

The record contains 197 passed tests and one skipped two-GPU test, with
ordered/scrambled parity, mixed cached batches, pause/resume and reset evidence.
Physical slot mappings, per-request stage lengths, numerical ranges and pool
bytes were checked against recorded fields without model execution locally.
Retain the successful notebook with outputs separately. This is correctness
evidence, not a paged-attention performance claim.

## D1 status

[D1_flattened_batching.json](D1_flattened_batching.json) was archived from the
user's Colab output on November 1, 2026. Original timestamp, T4 environment
and source commit `0044311` are preserved.

The record contains 140 passed tests with no skips, three-request packed
parity, last-row routing, zero cross-request isolation differences and zero
reordering differences. Layout, mask, numerical ranges and token routing
were checked against its recorded fields without executing models locally.
This is correctness evidence, not a batching performance claim. Retain the
successful notebook with outputs separately.
