import json

from tinyserve.envinfo import collect_env, save_result


def test_collect_env_has_expected_keys():
    env = collect_env()
    assert {"python", "platform", "git_sha", "gpu", "torch", "cuda", "transformers"} <= set(env)


def test_save_result_roundtrip(tmp_path):
    path = save_result("demo", {"tokens_per_s": [10.0, 12.5]}, results_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "demo.json"
    assert payload["data"] == {"tokens_per_s": [10.0, 12.5]}
    assert "env" in payload and "timestamp_utc" in payload
