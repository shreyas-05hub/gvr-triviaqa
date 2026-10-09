import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gvr.manifests import package_versions  # noqa: E402


def env_versions():
    v = package_versions()
    return {k: v[k] for k in ("torch", "transformers", "accelerate", "bitsandbytes", "pyserini", "numpy")}


def new_run_id(prefix):
    return f"{prefix}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
