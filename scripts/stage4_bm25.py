"""Stage 4. Default: print Pyserini registry metadata for the index (no download).
--load --confirm-download: guarded, timed load via gvr.retrieval.load_bm25_guarded + smoke regression."""
import argparse
import json
from _common import ROOT, write_json
from gvr.config import load_config
from gvr.data import load_questions
from gvr.retrieval import index_metadata, load_bm25_guarded, smoke_regression

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="store_true")
ap.add_argument("--confirm-download", action="store_true")
a = ap.parse_args()
cfg = load_config()
r = cfg["retrieval"]
if not a.load:
    reg, meta = index_metadata(r["prebuilt_index"])
    print("registry:", reg)
    print(json.dumps({k: meta.get(k) for k in ("description", "filename", "md5", "size compressed (bytes)",
                                               "documents", "total_terms", "unique_terms", "urls")}, indent=2))
    raise SystemExit(0)
if not a.confirm_download:
    raise SystemExit("refusing to download without --confirm-download (user confirmation required)")
(ROOT / "runs").mkdir(exist_ok=True)
bm, rec = load_bm25_guarded(r, timing_log=ROOT / "runs" / "setup_timing.jsonl")
reference = json.load(open(ROOT / "configs" / "bm25_smoke_reference.json"))
qs = load_questions(cfg, "train", reference["item_order"])
rows, problems = smoke_regression(bm, qs, reference, r)
for q in rows:
    print(q["item_id"], "|", q["n_hits"], "hits |", q["k1_b_readback"], "|", f"{q['latency_s']:.3f}s |",
          " ; ".join(q["titles"]))  # titles only
rec.update(smoke_regression=dict(rows=rows, problems=problems), openai_guard_calls=list(rec["openai_guard_calls"]))
rec["bm25_ok"] = not problems and not rec["openai_guard_calls"]
write_json(ROOT / "runs" / "bm25_session_record.json", rec)
if not rec["bm25_ok"]:
    raise SystemExit(f"STOP: BM25 regression/guard problems: {problems or rec['openai_guard_calls']}")
