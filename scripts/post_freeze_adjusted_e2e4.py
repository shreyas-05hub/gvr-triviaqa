"""POST-FREEZE ADDITION (PI-approved 2026-10-08; logged in freeze_manifest.post_freeze_additions).
Reason: em_validity_plan item 3 specified E2/E4 human-adjusted accuracy via constituent outputs; the frozen
analyze_main.py omitted it. Frozen outputs are NOT modified.

E2 = E0 on unflagged + E1 on flagged (invalid -> mitigation); E4 likewise with E3 (final gate tau).
Per item: adjusted correctness = EM_i + FN_src * 1[source output valid, non-abstain, EM=0], where FN_src is the
frozen item-4 per-arm FN rate of the item's source arm (E0/E1/E3). Interval: same Wilson lower/upper FN bounds
applied to all items (as the frozen script does per arm). Adjusted E2-E4: point = adj(E2)-adj(E4); interval =
[lo(E2)-hi(E4), hi(E2)-lo(E4)] (conservative bound combination).
ASSUMPTION: FN rates were estimated on a sample of ALL valid non-abstain EM=0 outputs per arm, and are applied here to
the flagged / unflagged subsets of those outputs (i.e., FN assumed homogeneous across the gate split within an arm).
Consistency check: the same method reproduces the frozen E0/E1/E3 adjusted values exactly."""
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

cfg = load_config()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
item4_path = ROOT / "runs/analysis_main_item4/results.json"
assert sha256_file(item4_path) == man["analysis_run_item4"]["outputs_sha256"]["runs/analysis_main_item4/results.json"]
res4 = json.load(open(item4_path))
fn = res4["em_audit"]["item4"]["fn"]
gate = json.load(open(ROOT / "configs/gate.json"))
assert sha256_file(ROOT / "configs/gate.json") == man["gate"]["sha256"]
calls = []
for r in ("test-c1", "test-c2", "test-c3"):
    d = ROOT / "runs" / r
    done = done_units(d)
    calls += [c for c in effective_calls(d) if (c["item_id"], c["arm"]) in done]
ids = sorted({c["item_id"] for c in calls})
assert len(ids) == 3000
items = derive_items(calls, cfg, load_questions(cfg, "validation", ids))
gold = load_gold(cfg, "validation", ids)
abst = cfg["scoring"]["abstain_list"]
em, elig = {}, {}
for (i, arm), it in items.items():
    v = not it["invalid"]
    e = bool(v and exact_match(it["answer"], ground_truths(gold[i])))
    em[(i, arm)] = e
    elig[(i, arm)] = v and not is_abstain(it["answer"], abst) and not e
dec = apply_gate([dict(item_id=i, invalid=items[(i, "E0")]["invalid"], u=items[(i, "E0")]["u"]) for i in ids], gate["tau"])


def adj(src_of):
    out = {}
    for tag, key in (("point", "fn_rate"), ("lo", 0), ("hi", 1)):
        f = (lambda a: fn[a]["fn_rate"]) if tag == "point" else (lambda a, k=key: fn[a]["wilson95"][k])
        out[tag] = float(np.mean([em[(i, src_of(i))] + f(src_of(i)) * elig[(i, src_of(i))] for i in ids]))
    return out


res = {a: adj(lambda i, a=a: a) for a in ("E0", "E1", "E3")}
for a in ("E0", "E1", "E3"):   # consistency with frozen item-4 output
    assert abs(res[a]["point"] - fn[a]["adj_accuracy"]) < 1e-12, a
    assert abs(res[a]["lo"] - fn[a]["adj_interval"][0]) < 1e-12 and abs(res[a]["hi"] - fn[a]["adj_interval"][1]) < 1e-12, a
res["E2"] = adj(lambda i: "E1" if dec[i]["mitigate"] else "E0")
res["E4"] = adj(lambda i: "E3" if dec[i]["mitigate"] else "E0")
em_acc = {a: float(np.mean([em[(i, "E1" if dec[i]["mitigate"] else "E0") if a == "E2" else (i, "E3" if dec[i]["mitigate"] else "E0") if a == "E4" else (i, a)] for i in ids])) for a in ("E0", "E1", "E3", "E2", "E4")}
assert all(abs(em_acc[a] - res4["descriptives"]["em_accuracy"][a]) < 1e-12 for a in em_acc)
diff = dict(point=res["E2"]["point"] - res["E4"]["point"], interval=[res["E2"]["lo"] - res["E4"]["hi"], res["E2"]["hi"] - res["E4"]["lo"]],
            method="conservative bound combination [lo(E2)-hi(E4), hi(E2)-lo(E4)]")
out = dict(status="POST-FREEZE ADDITION (PI-approved)", reason="item 3 specified E2/E4 adjustment; frozen script omitted it",
           assumption="FN rates estimated on all valid non-abstain EM=0 outputs per arm (n=50 sample each), applied to flagged-subset outputs",
           fn_source="runs/analysis_main_item4/results.json (frozen item-4 output)", em_accuracy=em_acc, adjusted=res,
           adjusted_E2_minus_E4=diff, consistency_check="E0/E1/E3 adjusted reproduce frozen item-4 values exactly; EM accuracies match frozen output",
           n_eligible={a: int(sum(elig[(i, (("E1" if dec[i]["mitigate"] else "E0") if a == "E2" else ("E3" if dec[i]["mitigate"] else "E0") if a == "E4" else a))] for i in ids)) for a in ("E0", "E1", "E3", "E2", "E4")})
write_json(ROOT / "runs/post_freeze/adjusted_e2e4.json", out)
print(json.dumps(out, indent=1))
