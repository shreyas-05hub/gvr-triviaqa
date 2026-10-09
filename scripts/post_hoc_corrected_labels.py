"""POST-HOC SENSITIVITY ANALYSIS (corrected audit labels; generated from post_freeze_adjusted_v2.py by fixed string replacements)
POST-FREEZE ADDITION v2 (PI-approved 2026-10-08): audit-adjusted accuracy = EM sensitivity-adjusted accuracy based
on the blinded audit. SECONDARY SENSITIVITY ANALYSIS ONLY; the frozen EM result (runs/analysis_main) is the PRIMARY.
Supersedes post_freeze_adjusted_e2e4.py (v1), whose interval plugged each arm's Wilson FN bounds in simultaneously as
fixed extremes: that ignored item-sampling variance and is not a confidence interval.

Point estimate (unchanged from v1): per item, adjusted correctness = EM_i + FN_src * 1[source output valid,
non-abstain, EM=0]; E0/E1/E3 use their own outputs; E2 = E0 on unflagged + E1 on flagged (invalid -> mitigation),
E4 likewise with E3 (final gate). FN_a = share of the 50 audited EM-failed outputs of arm a labelled CORRECT.

Interval: joint paired bootstrap, B = 10,000, numpy PCG64(20261008). Each replicate (i) resamples the 3,000 test items
with replacement (paired across arms) and (ii) resamples each arm's 50 audit labels with replacement, independently
per arm, re-estimating FN_E0, FN_E1, FN_E3. Percentile 95% CIs for every arm and for E2 - E4.

TRANSFERABILITY ASSUMPTION: the audit FN rates were estimated on a random sample of ALL valid, non-abstain EM-failed
outputs of each arm (E0, E1, E3), whereas E2/E4 invoke E1/E3 only on the gated subset (and E0 on the rest). Applying
the per-arm rates to E2/E4 assumes each arm's EM false-negative rate is the same on the gated and ungated subsets.
The bootstrap does not test or relax this assumption. The audit covers EM-failed outputs only, so EM false positives
are not estimated (assumed 0)."""
import csv
import json
import numpy as np
from _common import ROOT, write_json
from gvr.config import load_config, sha256_file
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.gate import apply_gate
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import exact_match, is_abstain
from gvr.test_runner import effective_calls, done_units

B, SEED = 10_000, 20261008
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--labels", required=True); _ap.add_argument("--labels-sha256", required=True)
_ap.add_argument("--out", required=True); _ap.add_argument("--status", required=True); _A = _ap.parse_args()
ARMS = ("E0", "E1", "E3")
cfg = load_config()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
assert sha256_file(ROOT / "runs/analysis_main/results.json") == man["analysis_run"]["outputs_sha256"]["runs/analysis_main/results.json"]
frozen = json.load(open(ROOT / "runs/analysis_main/results.json"))
key_p, lab_p = ROOT / "runs/analysis_main/em_audit_key.json", ROOT / _A.labels
assert sha256_file(key_p) == man["analysis_run"]["outputs_sha256"]["runs/analysis_main/em_audit_key.json"]
assert sha256_file(lab_p) == _A.labels_sha256
key = json.load(open(key_p))["rows"]
labels = {r["row_id"]: r["human_label(CORRECT/INCORRECT/UNSURE)"].strip().upper()
          for r in csv.DictReader(open(lab_p, newline="", encoding="utf-8"))}
audit = {a: np.array([labels[r["row_id"]] == "CORRECT" for r in key if r["arm"] == a and labels[r["row_id"]] != "UNSURE"]) for a in ARMS}
assert all(len(audit[a]) == 50 for a in ARMS)
gate = json.load(open(ROOT / "configs/gate.json"))
calls = []
for r in ("test-c1", "test-c2", "test-c3"):
    d = ROOT / "runs" / r
    done = done_units(d)
    calls += [c for c in effective_calls(d) if (c["item_id"], c["arm"]) in done]
ids = sorted({c["item_id"] for c in calls}); n = len(ids)
assert n == 3000
items = derive_items(calls, cfg, load_questions(cfg, "validation", ids))
gold = load_gold(cfg, "validation", ids)
abst = cfg["scoring"]["abstain_list"]
dec = apply_gate([dict(item_id=i, invalid=items[(i, "E0")]["invalid"], u=items[(i, "E0")]["u"]) for i in ids], gate["tau"])
mit = np.array([dec[i]["mitigate"] for i in ids])
EM, EL = {}, {}
for a in ARMS:
    em = np.array([(not items[(i, a)]["invalid"]) and exact_match(items[(i, a)]["answer"], ground_truths(gold[i])) for i in ids])
    el = np.array([(not items[(i, a)]["invalid"]) and not is_abstain(items[(i, a)]["answer"], abst) for i in ids]) & ~em
    EM[a], EL[a] = em.astype(float), el.astype(float)
# composed arms: source arm per item
SRC = {"E0": {a: (np.ones(n, bool) if a == "E0" else np.zeros(n, bool)) for a in ARMS},
       "E1": {a: (np.ones(n, bool) if a == "E1" else np.zeros(n, bool)) for a in ARMS},
       "E3": {a: (np.ones(n, bool) if a == "E3" else np.zeros(n, bool)) for a in ARMS},
       "E2": {"E0": ~mit, "E1": mit, "E3": np.zeros(n, bool)},
       "E4": {"E0": ~mit, "E1": np.zeros(n, bool), "E3": mit}}


def adjusted(arm, fn, idx=None):
    sl = slice(None) if idx is None else idx
    return float(sum((SRC[arm][a][sl] * (EM[a][sl] + fn[a] * EL[a][sl])).sum() for a in ARMS) / (n if idx is None else len(idx)))


def em_acc(arm):
    return float(sum((SRC[arm][a] * EM[a]).sum() for a in ARMS) / n)


fn_hat = {a: float(audit[a].mean()) for a in ARMS}
point = {arm: adjusted(arm, fn_hat) for arm in SRC}
for arm in SRC:                                   # consistency with frozen outputs
    assert abs(em_acc(arm) - frozen["descriptives"]["em_accuracy"][arm]) < 1e-12, arm
rng = np.random.Generator(np.random.PCG64(SEED))
bs = {arm: np.empty(B) for arm in SRC}; bs["E2-E4"] = np.empty(B); bs_fn = {a: np.empty(B) for a in ARMS}
for b in range(B):
    idx = rng.integers(0, n, n)
    fn_b = {a: float(audit[a][rng.integers(0, 50, 50)].mean()) for a in ARMS}
    for a in ARMS:
        bs_fn[a][b] = fn_b[a]
    for arm in SRC:
        bs[arm][b] = adjusted(arm, fn_b, idx)
    bs["E2-E4"][b] = bs["E2"][b] - bs["E4"][b]
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = dict(status=_A.status, labels_file=_A.labels, labels_sha256=_A.labels_sha256,
           metric="audit-adjusted accuracy (EM sensitivity-adjusted accuracy based on the blinded audit)",
           interval_method=f"joint paired bootstrap B={B}, PCG64({SEED}): items resampled (paired) + per-arm audit labels resampled; percentile 95%",
           transferability_assumption=("audit FN rates estimated on audited EM-failed outputs of each arm (all such outputs, n=50 sample per arm); "
                                       "E2/E4 invoke E1/E3 only on the gated subset; applying the rates to E2/E4 assumes equal FN rates on gated and ungated subsets"),
           em_false_positives="not estimated (audit covers EM-failed outputs only); assumed 0",
           audit_fn={a: dict(correct=int(audit[a].sum()), n=len(audit[a]), fn=fn_hat[a], boot_ci95=ci(bs_fn[a])) for a in ARMS},
           em_accuracy={arm: em_acc(arm) for arm in SRC},
           audit_adjusted={arm: dict(point=point[arm], ci95=ci(bs[arm])) for arm in SRC},
           audit_adjusted_E2_minus_E4=dict(point=point["E2"] - point["E4"], ci95=ci(bs["E2-E4"])),
           supersedes=dict(script="scripts/post_freeze_adjusted_e2e4.py", reason="v1 interval plugged Wilson FN bounds as fixed simultaneous extremes; ignored item-sampling variance; not a confidence interval"))
write_json(ROOT / _A.out, out)
print(json.dumps(out, indent=1))
