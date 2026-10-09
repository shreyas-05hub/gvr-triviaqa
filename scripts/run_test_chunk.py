"""Main-study test chunk (test_execution_rule). Aborts on any lock/hash mismatch BEFORE downloads or model load."""
import argparse
import json
import time
from _common import ROOT, env_versions, write_json
from gvr import prompts as P
from gvr.calllog import CallLogger
from gvr.config import load_config, sha256_file
from gvr.data import load_questions, read_ids
from gvr.guard import test_unlocked
from gvr.test_runner import run_chunk, progress

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", required=True, choices=["c1", "c2", "c3"])
a = ap.parse_args()
cfg = load_config()
man = json.load(open(ROOT / "configs" / "freeze_manifest.json"))
lock = man.get("test_lock")


def need(cond, msg):
    if not cond:
        raise SystemExit(f"ABORT before any download/model load: {msg}")


need(test_unlocked(), "test_unlocked is not true")
need(a.chunk in {x["chunk"] for x in man.get("test_chunk_approvals", [])}, f"chunk {a.chunk} not approved by PI")
need(lock and lock.get("status") == "LOCKED", "no LOCKED test_lock in freeze_manifest")
P.assert_frozen()
need(sha256_file(ROOT / "configs/gate.json") == lock["gate_sha256"] == man["gate"]["sha256"], "gate.json hash mismatch")
need(sha256_file(ROOT / "configs/config.yaml") == lock["config_sha256"], "config.yaml hash mismatch")
need(sha256_file(ROOT / "data/test_ids.json") == lock["test_ids_sha256"] == man["test_ids_sha256"], "test_ids hash mismatch")
need({k: v["sha256"] for k, v in man["frozen_prompts"]["prompts"].items()} == lock["prompt_sha256"], "frozen prompt set != lock")
need(cfg["model"]["revision"] == lock["model_revision"], "model revision mismatch")
need(json.load(open(ROOT / "configs/gate.json"))["provisional"] is False, "gate is provisional")
lo, hi = man["test_ids"]["chunks"][a.chunk]
ids = read_ids("test_ids")[lo:hi]
assert len(ids) == 1000
qs = load_questions(cfg, "validation", ids)      # guard permits only because test_unlocked is true; no gold loaded
run_id = f"test-{a.chunk}"
run_dir = ROOT / "runs" / run_id
lg = CallLogger(run_dir, run_id, f"validation:test_ids:{a.chunk}", env_versions(), fixture=False)

import torch
from gvr.arms import Runner
from gvr.model import QAModel
from gvr.retrieval import load_bm25_guarded, smoke_regression
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
bm, bm_rec = load_bm25_guarded(cfg["retrieval"], timing_log=ROOT / "runs" / "setup_timing.jsonl")
ref = json.load(open(ROOT / "configs" / "bm25_smoke_reference.json"))
rows, problems = smoke_regression(bm, load_questions(cfg, "train", ref["item_order"]), ref, cfg["retrieval"])
bm_rec.update(smoke_regression=dict(rows=rows, problems=problems))
with open(run_dir / "bm25_session_records.jsonl", "a") as f:
    f.write(json.dumps(bm_rec) + "\n")
if problems or bm_rec["openai_guard_calls"]:
    raise SystemExit(f"STOP before any arm: BM25 problems {problems} / OpenAI calls {bm_rec['openai_guard_calls']}")
qa = QAModel(cfg, fixture=False)
res = run_chunk(Runner(cfg, qa, lg, bm25=bm), run_dir, ids, qs, cfg)
meta = dict(run_id=run_id, chunk=a.chunk, session_end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **res,
            peak_gpu_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()), gpu=torch.cuda.get_device_name(0),
            n_visible_gpus=torch.cuda.device_count(), wall_s=time.time() - t0, kaggle_dataset=man.get("kaggle_dataset"),
            **qa.meta())
with open(run_dir / "run_meta.jsonl", "a") as f:
    f.write(json.dumps(meta) + "\n")
write_json(run_dir / "progress.json", progress(run_dir, ids, cfg, qs))   # errors/invalid counts only; NO gold
print(json.dumps(meta, indent=1))
