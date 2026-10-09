import datetime
import json
import platform
import shutil
import subprocess
from importlib import metadata

from .config import sha256_file
from .paths import CONFIGS, CONFIG_YAML, DATA, FREEZE_MANIFEST
from . import prompts as P

PACKAGES = ["torch", "transformers", "accelerate", "bitsandbytes", "datasets", "huggingface_hub",
            "pyserini", "pyjnius", "faiss-cpu", "faiss-gpu", "numpy", "pyyaml", "pyarrow", "tokenizers",
            "safetensors", "pytest"]


def package_versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"unavailable: {type(e).__name__}"


def hardware():
    hw = dict(platform=platform.platform(), python=platform.python_version(),
              cpu=platform.processor() or _cmd(["sh", "-c", "lscpu | grep 'Model name'"]),
              disk_free_gb=round(shutil.disk_usage("/").free / 1e9, 1))
    try:
        hw["ram_gb"] = round(int(open("/proc/meminfo").read().split()[1]) / 1e6, 1)
    except Exception:
        pass
    try:
        import torch
        hw["torch_cuda_build"] = torch.version.cuda
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                               vram_gb=round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                               capability=list(torch.cuda.get_device_capability(i)))
                          for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    hw["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                             "--format=csv,noheader"])
    hw["java"] = _cmd(["java", "-version"])
    import os
    hw["JAVA_HOME"] = os.environ.get("JAVA_HOME")
    hw["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    for k in ("PYSERINI_CACHE", "HF_HOME"):
        hw[k] = os.environ.get(k)
    return hw


def write_env_manifest(path=CONFIGS / "env_manifest.json", role="laptop-cpu"):
    audit = CONFIGS / "kaggle_audit_stage0.txt"
    rec = dict(role=role, kaggle_dataset=_kaggle_dataset(),
               kaggle_audit_stage0=dict(file="configs/kaggle_audit_stage0.txt", sha256=sha256_file(audit),
                                        kind="re-run captured by runner env stage") if audit.exists() else None,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               hardware=hardware(), packages=package_versions())
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def _kaggle_dataset():
    p = CONFIGS / "kaggle_dataset.json"
    return json.load(open(p)) if p.exists() else None


def _device():
    import os
    d = dict(CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        import torch
        if torch.cuda.is_available():
            d.update(n_visible=torch.cuda.device_count(), name=torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return d


def run_manifest(cfg, extra=None):
    def h(p):
        return sha256_file(p) if p.exists() else None
    man = dict(model_id=cfg["model"]["id"], model_revision=cfg["model"]["revision"],
               precision=cfg["model"]["precision"], dataset_repo=cfg["dataset"]["repo_id"],
               dataset_revision=cfg["dataset"]["revision"], prompts=P.registry(),
               config_sha256=h(CONFIG_YAML), env_manifest_sha256=h(CONFIGS / "env_manifest.json"),
               pilot_ids_sha256=h(DATA / "pilot_ids.json"), calib_ids_sha256=h(DATA / "calib_ids.json"),
               freeze_manifest=json.load(open(FREEZE_MANIFEST)),
               gate_json_sha256=h(CONFIGS / "gate.json"),
               index=cfg["retrieval"].get("index_record"),
               environment_decisions=cfg["environment_decisions"],
               interpretations=dict(**cfg["parser"]["interpretations"], recall_text=cfg["retrieval"]["recall_text"]),
               device=_device(), kaggle_dataset=_kaggle_dataset(), seeds={k: v for k, v in cfg["sampling"].items() if k.startswith("seed")})
    man.update(extra or {})
    return man
