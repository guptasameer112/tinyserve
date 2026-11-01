# Recorded lesson results

These JSON files contain saved Colab observations, not fresh local runs.
Each result includes the original UTC timestamp, GPU/library environment,
git SHA and lesson data.

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

[C4_cache_curve.ipynb](../notebooks/C4_cache_curve.ipynb) is ready for the
user's fresh Colab T4 run. Its runtime tests and measured curves are pending.
It saves `C4_cache_curve.json` after each point with `status="partial"` and
marks it complete only after both sweeps and plots succeed. No placeholder
benchmark result or performance claim is committed.
