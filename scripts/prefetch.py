"""Timed setup download of model weights (counted as setup, never per-item cost).
Default prints sizes only; --confirm-download required to download."""
import argparse
import json
import os
import time
from huggingface_hub import HfApi, snapshot_download
from _common import ROOT
from gvr.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--what", choices=["qwen", "judge"], required=True)
ap.add_argument("--confirm-download", action="store_true")
a = ap.parse_args()
cfg = load_config()
if a.what == "qwen":
    rid, rev, pats = cfg["model"]["id"], cfg["model"]["revision"], None
else:
    j = cfg["scoring"]["judge"]
    rid, rev, pats = j["model_id"], j["revision"], j["allow_patterns"]
info = HfApi().model_info(rid, revision=rev, files_metadata=True)
import fnmatch
files = [s for s in info.siblings if pats is None or any(fnmatch.fnmatch(s.rfilename, p) for p in pats)]
size = sum(s.size or 0 for s in files)
print(f"{rid}@{rev}: {len(files)} files, {size / 1e9:.2f} GB -> HF_HOME={os.environ.get('HF_HOME')}")
if not a.confirm_download:
    raise SystemExit(0)
t0 = time.time()
snapshot_download(rid, revision=rev, allow_patterns=pats)
dt = time.time() - t0
(ROOT / "runs").mkdir(exist_ok=True)
with open(ROOT / "runs" / "setup_timing.jsonl", "a") as f:
    f.write(json.dumps(dict(step=f"download_{a.what}", repo=rid, revision=rev, bytes=size, seconds=dt,
                            t_end=time.time())) + "\n")
print(f"downloaded in {dt:.0f} s")
