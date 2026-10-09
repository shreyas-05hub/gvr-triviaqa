"""Test-size planning (dev data only; pilot-01 + judgedev-01, prompts v1). Final gate tau applied to their E0 u.
Discordance rates for McNemar sizing. Nothing here feeds any arm-choice decision."""
import json
import math
import numpy as np
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.gate import apply_gate, rates
from gvr.random_gate import random_gate  # noqa: F401  (same RNG convention)
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import exact_match

cfg = load_config()
gate = json.load(open(ROOT / "configs/gate.json"))
assert gate["provisional"] is False
TAU = gate["tau"]


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - w), min(1.0, c + w)


em = {}; e0_items = []
for run in ("pilot-01", "judgedev-01"):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
    gold = load_gold(cfg, "train", ids)
    for (iid, arm), it in items.items():
        em[(iid, arm)] = (not it["invalid"]) and exact_match(it["answer"], ground_truths(gold[iid]))
        if arm == "E0":
            e0_items.append(dict(item_id=iid, invalid=it["invalid"], u=it["u"]))
dec = apply_gate(e0_items, TAU)
r = rates(dec)
ids = [x["item_id"] for x in e0_items]
E = {a: {i: em[(i, a)] for i in ids} for a in ("E0", "E1", "E3")}
E2 = {i: E["E1"][i] if dec[i]["mitigate"] else E["E0"][i] for i in ids}
E4 = {i: E["E3"][i] if dec[i]["mitigate"] else E["E0"][i] for i in ids}
N = len(ids)


def disc(A, B):
    p10 = sum(A[i] and not B[i] for i in ids); p01 = sum(B[i] and not A[i] for i in ids)
    return dict(n=N, n10=p10, n01=p01, p10=p10 / N, p01=p01 / N, p_d=wilson(p10 + p01, N))


# random gate: same k valid items chosen uniformly (seeds 0..999); invalid forced
valid = sorted(x["item_id"] for x in e0_items if not x["invalid"]); inv = {x["item_id"] for x in e0_items if x["invalid"]}
k = r["k_flagged_valid"]
rg = {"E2": [], "E4": []}
for s in range(1000):
    rng = np.random.Generator(np.random.PCG64(s))
    mit = set(rng.choice(valid, size=k, replace=False).tolist()) | inv
    for name, A, mitarm in (("E2", E2, "E1"), ("E4", E4, "E3")):
        R = {i: E[mitarm][i] if i in mit else E["E0"][i] for i in ids}
        rg[name].append(sum(A[i] != R[i] for i in ids) / N)
res = dict(note="DEV DATA ONLY (train split; pilot-01 + judgedev-01; prompts v1); sizing only", tau=TAU, n_items=N,
           realized=r, E2_vs_E4=disc(E2, E4), E2_vs_E0=disc(E2, E["E0"]), E4_vs_E0=disc(E4, E["E0"]),
           E2_vs_random_gate_mean_p_d=float(np.mean(rg["E2"])), E4_vs_random_gate_mean_p_d=float(np.mean(rg["E4"])),
           random_gate_p_d_sd=dict(E2=float(np.std(rg["E2"])), E4=float(np.std(rg["E4"]))))
write_json(ROOT / "runs" / "sizing_dev.json", res)
print(json.dumps(res, indent=1))
