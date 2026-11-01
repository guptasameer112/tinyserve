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

[C3_kv_cache.ipynb](../notebooks/C3_kv_cache.ipynb) is ready for the user's
fresh Colab T4 run. Runtime validation is pending. Its final save cell will
create `C3_kv_cache.json`; no placeholder success result is committed.

Save both the JSON and notebook with outputs before replacing a Colab runtime.
