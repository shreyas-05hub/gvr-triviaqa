"""Measures download bandwidth with small ranged GETs (default 128 MB each) to estimate setup time."""
import json
import time
import urllib.request
from huggingface_hub import hf_hub_url
from _common import ROOT
from gvr.config import load_config

N = 128 * 2**20
cfg = load_config()
urls = {"hf_qwen_shard": hf_hub_url(cfg["model"]["id"], "model-00001-of-00004.safetensors", revision=cfg["model"]["revision"]),
        "uwaterloo_index": "https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz"}
sizes = {"hf_qwen_shard": 15.24e9, "uwaterloo_index": cfg["retrieval"]["index_download"]["compressed_bytes"]}
out = {}
for k, u in urls.items():
    req = urllib.request.Request(u, headers={"Range": f"bytes=0-{N - 1}"})
    t0 = time.time(); got = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        while chunk := r.read(1 << 20):
            got += len(chunk)
    dt = time.time() - t0
    out[k] = dict(bytes=got, seconds=dt, MBps=got / dt / 1e6, est_full_download_min=sizes[k] / (got / dt) / 60)
print(json.dumps(out, indent=2))
(ROOT / "runs").mkdir(exist_ok=True)
with open(ROOT / "runs" / "setup_timing.jsonl", "a") as f:
    f.write(json.dumps(dict(step="bw_probe", **out)) + "\n")
