"""Environment capture and result saving. Every benchmark result goes through save_result."""
import json
import platform
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_env():
    """Describe the machine and library versions. torch/transformers are optional imports."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "gpu": None,
        "torch": None,
        "cuda": None,
        "transformers": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def save_result(name, data, results_dir=None):
    """Write data plus environment info to results/<name>.json and return the path."""
    results_dir = Path(results_dir) if results_dir is not None else REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": collect_env(),
        "data": data,
    }
    path = results_dir / (name + ".json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
