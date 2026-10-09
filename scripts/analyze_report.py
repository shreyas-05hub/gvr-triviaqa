"""Stages 7-10 + 13: derive items, score, provisional gate, compose, random gate, regenerate the report."""
import argparse
import csv
import json
import numpy as np
from _common import ROOT, write_json
from gvr import prompts as P
from gvr.calllog import read_jsonl
from gvr.config import load_config, sha256_file
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.gate import fit_gate, save_gate
from gvr.manifests import run_manifest
from gvr.parsing import PARSER_VERSION
from gvr.report import render
from gvr.scoring.analyze import analyze, judge_agreement, judge_requests, score_items, write_handlabel_csv
from gvr.scoring.gold import audit_no_gold, load_gold

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--determinism", nargs="*", default=[], help="determinism_*.json files")
ap.add_argument("--checks", default=None, help="json with rescore_pass, bm25_ok, no_oom_2048 from GPU tests")
ap.add_argument("--n-test-placeholder", default="N_test")
ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
ap.add_argument("--judge", choices=["on", "off"], default=None,
                help="default from config scoring.judge.enabled_in_main_study (off after judge_rule_v2 fallback); "
                     "'on' only to regenerate the pilot appendix")
ap.add_argument("--out-dir", default=str(ROOT), help="root for configs/gate.json and reports/ (tests use a tmp dir)")
a = ap.parse_args()
from pathlib import Path
OUT = Path(a.out_dir)
(OUT / "reports").mkdir(parents=True, exist_ok=True)
(OUT / "configs").mkdir(parents=True, exist_ok=True)
cfg = load_config()
run_dir = Path(a.runs_dir) / a.run_id
calls = read_jsonl(run_dir / "calls.jsonl")
assert not any(c.get("fixture") for c in calls), "fixture calls must never be reported"
ids = sorted({c["item_id"] for c in calls})
qs = load_questions(cfg, "train", ids)
gold = load_gold(cfg, "train", ids)
audit_no_gold(calls, gold)

items = derive_items(calls, cfg, qs)
verdicts, unparsed = {}, []
jpath = run_dir / "judge.jsonl"
if jpath.exists():
    for j in read_jsonl(jpath):
        verdicts[(j["item_id"], j["arm"])] = bool(j["verdict"])  # unparsed -> False (pending user confirmation)
        if j.get("judge_unparsed"):
            unparsed.append((j["item_id"], j["arm"]))
reqs = judge_requests(items, gold, qs, cfg["scoring"]["abstain_list"])
write_handlabel_csv(OUT / "reports" / f"judge_handlabel_blind_{a.run_id}.csv", reqs)
JUDGE_ON = (a.judge == "on") if a.judge else bool(cfg["scoring"]["judge"].get("enabled_in_main_study", False))
if JUDGE_ON:
    missing = [r for r in reqs if (r["item_id"], r["arm"]) not in verdicts]
    if missing:
        raise SystemExit(f"{len(missing)} judge verdicts missing; run scripts/judge.py first")
else:
    verdicts, unparsed = {}, []

scored = score_items(items, gold, verdicts, cfg["scoring"]["abstain_list"], unparsed, judge_enabled=JUDGE_ON)
with open(run_dir / "items.jsonl", "w") as f:
    for (iid, arm), it in sorted(scored.items()):
        f.write(json.dumps(dict(it, item_id=iid, arm=arm), ensure_ascii=False) + "\n")
res = analyze(scored, cfg)

e0 = [it for (i, arm), it in scored.items() if arm == "E0"]
tau, dec, rates = fit_gate(e0, cfg["gate"])
prov = dict(ids_file="data/pilot_ids.json", ids_sha256=sha256_file(ROOT / "data/pilot_ids.json"),
            prompt_sha={k: v["sha256"] for k, v in P.registry().items()}, model_id=cfg["model"]["id"],
            model_revision=cfg["model"]["revision"], precision=cfg["model"]["precision"],
            parser_version=PARSER_VERSION, run_id=a.run_id,
            u_values={it["item_id"]: (None if it["invalid"] else it["u"]) for it in e0})
gate_path = OUT / "configs" / "gate.json"
save_gate(gate_path, tau, rates, cfg["gate"], prov, provisional=True)
g = json.load(open(gate_path))
re_items = [dict(item_id=k, invalid=v is None, u=v) for k, v in g["u_values"].items()]
tau2, _, rates2 = fit_gate(re_items, cfg["gate"])
tau_ok = tau2 == g["tau"] and rates2 == g["realized"]

gen = [c for c in calls if c["step"] != "retrieve"]
final_fail = 0
by_key = {}
for c in calls:
    by_key.setdefault((c["item_id"], c["arm"], c["step"], c["step_idx"]), []).append(c)
for cs in by_key.values():
    if all(c.get("error") for c in cs):
        final_fail += 1
retried = sum(c.get("retry_of") is not None for c in calls)
checks = dict(unrecovered=final_fail, retried_rate=retried / len(by_key),
              u_all_valid=all(it["u"] is not None for it in e0 if not it["invalid"]),
              tau_reproducible=tau_ok, judge_unparsed=len(unparsed), judge_unparsed_ids=unparsed)
if a.checks:
    checks.update(json.load(open(a.checks)))
ha = OUT / "reports" / f"judge_handlabel_blind_{a.run_id}_LABELED.csv"
if ha.exists():
    ag, n_lab = judge_agreement(list(csv.DictReader(open(ha))), verdicts)
    checks["judge_agreement"], checks["judge_agreement_n"] = ag, n_lab
# BM25 checks recorded before any arm in this run
br = run_dir / "bm25_session_record.json"
if br.exists():
    b = json.load(open(br))
    checks["bm25_ok"] = (not b["smoke_regression"]["problems"] and not b["openai_guard_calls"]
                         and b["num_docs"] == cfg["retrieval"]["index_download"]["documents"])
    checks["bm25_detail"] = dict(num_docs=b["num_docs"], lucene=b["lucene_commit_version"], analyzer=b["analyzer"],
                                 similarity=b["similarity_readback"], smoke_problems=b["smoke_regression"]["problems"],
                                 openai_guard_calls=b["openai_guard_calls"], timing_s=b["timing_s"])
# OOM / 2048-token cap from this run's own calls
e3_in = [c["input_tokens"] for c in calls if c["arm"] == "E3" and c["step"] == "answer" and c.get("input_tokens")]
oom = [c["error"] for c in calls if c.get("error") and "out of memory" in c["error"].lower()]
checks["no_oom_2048"] = (not oom) and (max(e3_in, default=0) <= cfg["retrieval"]["rag_max_input_tokens"])
checks["e3_max_input_tokens"] = max(e3_in, default=None)
checks["e3_truncated_count"] = sum(bool(c.get("truncated")) for c in calls if c["arm"] == "E3" and c["step"] == "answer")
# GPU-hours from the Kaggle quota log (includes setup); baseline = first push of the pilot programme
import re as _re
qrows = [json.loads(l) for l in open(ROOT / "runs" / "setup_timing.jsonl") if '"kaggle_quota_' in l] \
    if (ROOT / "runs" / "setup_timing.jsonl").exists() else []
used = [float(m.group(1)) for r in qrows for m in [_re.search(r"GPU\s+([0-9.]+)h", r["quota"])] if m]
checks["gpu_hours_quota"] = (used[-1] - used[0]) if len(used) >= 2 else None
checks["gpu_hours_quota_note"] = "Kaggle quota 'used' delta from first pilot-programme push to latest pull (incl. setup and failed attempts)"
# alignment + same-path reproducibility (from unit_main / diag sessions)
chk_local = json.load(open(ROOT / "runs" / "checks.json")) if (ROOT / "runs" / "checks.json").exists() else {}
for k in ("alignment_pass", "same_path_repro_pass"):
    checks.setdefault(k, chk_local.get(k))
checks["test_history"] = json.load(open(ROOT / "configs" / "test_history.json")) if (ROOT / "configs" / "test_history.json").exists() else None
jd = run_dir / "judge_decision.json"
if jd.exists():
    checks["judge_decision"] = json.load(open(jd))
metas = [json.load(open(m)) for m in Path(a.runs_dir).glob("*/run_meta.json")]
checks["gpu_hours_scripts"] = sum(m["total_wall_s"] for m in metas) / 3600 if metas else None
dets = [json.load(open(p)) for p in a.determinism]
if dets:
    checks["determinism"] = dets
    flags = [d["any_primary_mismatch"] for d in dets if "any_primary_mismatch" in d]
    checks["determinism_ok"] = (not any(flags)) if flags else None   # pre-registered target: 0 primary mismatches

warm = cfg["reproducibility"]["warmup_calls_excluded"]
ok_gen = sorted([c for c in gen if c.get("error") is None], key=lambda c: c["t_start"])[warm:]
timing = {}
for (arm, step) in sorted({(c["arm"], c["step"]) for c in ok_gen}):
    t = [c["t_end"] - c["t_start"] for c in ok_gen if (c["arm"], c["step"]) == (arm, step)]
    timing[f"{arm}/{step}"] = dict(n=len(t), mean_s=float(np.mean(t)), median_s=float(np.median(t)), max_s=float(np.max(t)))
ret = [c["retrieval"]["latency_s"] for c in calls if c["step"] == "retrieve" and c.get("error") is None]
if ret:
    timing["E3/retrieve"] = dict(n=len(ret), mean_s=float(np.mean(ret)))
timing["batch_size"] = cfg["model"]["batch_size"]
timing["warmup_calls_excluded"] = warm
meta = run_dir / "run_meta.json"
if meta.exists():
    timing["run_meta"] = json.load(open(meta))

def per_item(arm):
    return sum(v["mean_s"] * (v["n"] / len(ids)) for k, v in timing.items()
               if isinstance(v, dict) and "mean_s" in v and k.startswith(arm + "/"))
t_e0, t_e1, t_e3 = per_item("E0"), per_item("E1"), per_item("E3")
jm = run_dir / "judge_meta.json"
t_judge = json.load(open(jm))["latency_s_excl_warmup"] if jm.exists() else None
judged_per_item = len(reqs) / len(ids) if ids else None
projection = dict(
    formula=f"GPU_h = [500*t_E0 + {a.n_test_placeholder}*(t_E0 + t_E1 + t_E3 + j*t_judge)]/3600 + setup_h per session; "
            "t_arm = measured mean seconds per item (warm-up excluded); j = judged calls per item in this pilot; "
            "t_judge = measured judge seconds per call; setup (venv ~1 min, Qwen ~1.2 min, index ~6 min) per session",
    t_judge_s_per_call=t_judge, judge_calls_per_item=judged_per_item,
    per_test_item_gpu_h_incl_judge=((t_e0 + t_e1 + t_e3 + judged_per_item * t_judge) / 3600) if t_judge else None,
    t_E0_s_per_item=t_e0, t_E1_s_per_item=t_e1, t_E3_s_per_item=t_e3,
    calib_part_gpu_h=500 * t_e0 / 3600, per_test_item_gpu_h=(t_e0 + t_e1 + t_e3) / 3600)

manifest = run_manifest(cfg, dict(run_id=a.run_id, gate_json_sha256=sha256_file(gate_path), index_record=json.load(open(ROOT / "configs/index_record.json"))
                                  if (ROOT / "configs/index_record.json").exists() else None,
                                  env_manifest=json.load(open(ROOT / "configs/env_manifest.json"))
                                  if (ROOT / "configs/env_manifest.json").exists() else None))
res_out = {k: v for k, v in res.items() if k != "decisions"}
write_json(run_dir / "results.json", dict(results=res_out, checks=checks, timing=timing, projection=projection))
md = render(res, manifest, checks, cfg["go_no_go"], timing, projection)
(OUT / "reports" / "pilot_report.md").write_text(md)
print("wrote", OUT / "reports" / "pilot_report.md")
