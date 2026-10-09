"""Fit the FINAL gate on calibration E0 (generate-path u; u_decision_rule case 1). NO gold, NO EM, NO accuracy.
tau = numpy.quantile(u[valid], 0.60, method='higher'); flag = u > tau. Archives the provisional pilot gate."""
import hashlib
import json
import shutil
import numpy as np
from _common import ROOT, write_json
from gvr import prompts as P
from gvr.calllog import read_jsonl
from gvr.config import load_config, sha256_file
from gvr.data import load_questions, read_ids
from gvr.derive import derive_items
from gvr.gate import fit_gate, save_gate
from gvr.parsing import PARSER_VERSION
from gvr.scoring.normalize import normalize_answer

RUN = "calib-01"
cfg = load_config()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
P.assert_frozen()
ids = read_ids("calib_ids")
assert len(ids) == cfg["sampling"]["n_calib"] == 500
assert sha256_file(ROOT / "data/calib_ids.json") == man["calib_ids_sha256"], "calib ids changed"
calls = read_jsonl(ROOT / "runs" / RUN / "calls.jsonl")
assert {c["arm"] for c in calls} == {"E0"} and not any(c.get("fixture") for c in calls), "E0-only real run expected"
assert sorted({c["item_id"] for c in calls}) == sorted(ids)
items = {iid: it for (iid, arm), it in derive_items(calls, cfg, load_questions(cfg, "train", ids)).items()}
e0 = [dict(item_id=i, invalid=items[i]["invalid"], u=items[i]["u"]) for i in sorted(ids)]
abst = {normalize_answer(x) for x in cfg["scoring"]["abstain_list"]}
n_abstain = sum((not items[i]["invalid"]) and normalize_answer(items[i]["answer"]) in abst for i in ids)
tau, dec, rates = fit_gate(e0, cfg["gate"])

gate_path = ROOT / "configs" / "gate.json"
arch = ROOT / "configs" / "gate_provisional_pilot.json"
if gate_path.exists() and json.load(open(gate_path)).get("provisional"):
    shutil.copy(gate_path, arch)
prov = dict(ids_file="data/calib_ids.json", ids_sha256=man["calib_ids_sha256"], seed=cfg["sampling"]["seed_b_calib"],
            prompt_sha={k: v["sha256"] for k, v in man["frozen_prompts"]["prompts"].items()},
            model_id=cfg["model"]["id"], model_revision=cfg["model"]["revision"], precision=cfg["model"]["precision"],
            parser_version=PARSER_VERSION, u_definition="generate-path (u_decision_rule case 1)", run_id=RUN,
            calls_sha256=sha256_file(ROOT / "runs" / RUN / "calls.jsonl"),
            n_abstain=n_abstain, u_values={r["item_id"]: (None if r["invalid"] else r["u"]) for r in e0},
            archived_provisional_gate=str(arch.relative_to(ROOT)) if arch.exists() else None)
save_gate(gate_path, tau, rates, cfg["gate"], prov, provisional=False)
g = json.load(open(gate_path))
re_items = [dict(item_id=k, invalid=v is None, u=v) for k, v in g["u_values"].items()]
tau2, _, rates2 = fit_gate(re_items, cfg["gate"])
assert tau2 == g["tau"] and rates2 == g["realized"], "tau not reproducible"
U = np.array([r["u"] for r in e0 if not r["invalid"]])
summary = dict(n=len(ids), n_valid=rates["n_valid"], n_invalid=rates["n_invalid"], n_abstain=n_abstain, tau=tau,
               realized_nominal_flag_rate=rates["nominal_rate"], overall_mitigation_rate=rates["overall_mitigation_rate"],
               k_flagged_valid=rates["k_flagged_valid"],
               u_summary={q: float(np.percentile(U, p)) for q, p in (("min", 0), ("p10", 10), ("p50", 50), ("p90", 90), ("max", 100))},
               tau_reproducible=True, gate_json_sha256=sha256_file(gate_path))
write_json(ROOT / "runs" / RUN / "calibration_summary.json", summary)
print(json.dumps(summary, indent=1))
