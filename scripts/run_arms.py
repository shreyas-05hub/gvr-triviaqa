"""Stages 3/5/6/12: run E0, then E1 (on LOGGED E0 answers), then E3. Batch size 1."""
import argparse
import json
import time
from _common import ROOT, env_versions, new_run_id, write_json
from gvr.arms import Runner
from gvr.calllog import CallLogger, read_jsonl
from gvr.config import load_config
from gvr.data import load_questions, read_ids
from gvr.derive import derive_items
from gvr.model import QAModel

ap = argparse.ArgumentParser()
ap.add_argument("--ids", default="pilot_ids", choices=["pilot_ids", "judgedev_ids", "calib_ids"])  # test never via this flag
ap.add_argument("--arms", default="E0,E1,E3")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--run-id", default=None)
ap.add_argument("--calib-rerun-approved", default=None, metavar="REASON",
                help="required (with PI approval) for any calibration run after calib-01; logged")
ap.add_argument("--a1-from", default=None, help="run_id whose LOGGED E0 answers E1 consumes (no E0 generated here)")
a = ap.parse_args()

cfg = load_config()
from gvr import prompts as _P
_P.assert_frozen()
if a.ids == "calib_ids":   # calib-01 is the one and only calibration run
    _cr = ROOT / "configs" / "calibration_runs.json"
    _done = json.load(open(_cr))["runs"] if _cr.exists() else []
    if _done and not a.calib_rerun_approved:
        raise SystemExit(f"calibration already run ({[r['run_id'] for r in _done]}); rerun needs PI approval "
                         "(--calib-rerun-approved REASON), which is logged")
    if _done:
        _log = json.load(open(_cr)); _log.setdefault("rerun_approvals", []).append(dict(run_id=a.run_id, reason=a.calib_rerun_approved))
        json.dump(_log, open(_cr, "w"), indent=2)   # abort before any download/model load if prompts differ from the frozen set
arms = a.arms.split(",")
ids = read_ids(a.ids)[: a.limit]
qs = load_questions(cfg, "train", ids)
run_id = a.run_id or new_run_id("pilot")
run_dir = ROOT / "runs" / run_id
lg = CallLogger(run_dir, run_id, "train:" + a.ids, env_versions(), fixture=False)
import torch
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
bm = None
if "E3" in arms:   # guarded, timed load + smoke regression BEFORE any arm runs; any problem -> STOP
    from gvr.retrieval import load_bm25_guarded, smoke_regression
    bm, bm_rec = load_bm25_guarded(cfg["retrieval"], timing_log=ROOT / "runs" / "setup_timing.jsonl")
    reference = json.load(open(ROOT / "configs" / "bm25_smoke_reference.json"))
    rows, problems = smoke_regression(bm, load_questions(cfg, "train", reference["item_order"]), reference,
                                      cfg["retrieval"])
    bm_rec.update(smoke_regression=dict(rows=rows, problems=problems))
    write_json(run_dir / "bm25_session_record.json", bm_rec)
    if problems or bm_rec["openai_guard_calls"]:
        raise SystemExit(f"STOP before any arm: BM25 problems {problems} / OpenAI calls {bm_rec['openai_guard_calls']}")
qa = QAModel(cfg, fixture=False)
r = Runner(cfg, qa, lg, bm25=bm)
if "E0" in arms:
    for i in ids:
        r.e0(i, qs[i])
if "E1" in arms:
    src = ROOT / "runs" / a.a1_from / "calls.jsonl" if a.a1_from else lg.path
    assert a.a1_from is None or "E0" not in arms, "--a1-from excludes running E0 in the same process"
    items = derive_items(read_jsonl(src), cfg, qs)  # A1 = logged E0 answer, never regenerated
    for i in ids:
        e0 = items[(i, "E0")]
        r.e1(i, qs[i], e0["answer"], e0["invalid"])
if "E3" in arms:
    for i in ids:
        r.e3(i, qs[i])
write_json(run_dir / "run_meta.json", dict(
    run_id=run_id, ids_file=a.ids, a1_from=a.a1_from, n_items=len(ids), arms=arms, batch_size=cfg["model"]["batch_size"],
    peak_gpu_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()),
    peak_gpu_mem_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    gpu=torch.cuda.get_device_name(0), cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    n_visible_gpus=torch.cuda.device_count(),
    kaggle_dataset=json.load(open(ROOT / "configs/kaggle_dataset.json")) if (ROOT / "configs/kaggle_dataset.json").exists() else None,
    total_wall_s=time.time() - t0, **qa.meta()))
print("done", run_id)
