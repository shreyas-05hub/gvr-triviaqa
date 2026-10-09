"""MAIN ANALYSIS (pre-registered plan; frozen as ANALYSIS_FROZEN in freeze_manifest before c3 approval).

Inputs: one or more run dirs (test: test-c1, test-c2, test-c3), the FINAL gate (configs/gate.json), official EM gold.
For the test split, gold is reachable only after test_completeness passed (enforced in gvr.scoring.gold.load_gold).

1. Compose E2/E4 from logs with the final gate: flag = u > tau (strict); invalid E0 -> forced mitigation;
   unflagged items take the E0 answer verbatim (asserted).
2. Primary: exact two-sided McNemar, E2 vs E4, on EM (official TriviaQA gold set).
   Secondary (Holm over 4): exact McNemar E2 vs E0, E4 vs E0; E2 vs random gate, E4 vs random gate
   (1,000 seeds 0..999, numpy PCG64; k = realized number of flagged VALID items; invalid always mitigated).
   Random-gate p = min(1, 2*min((1+#{null>=obs})/(B+1), (1+#{null<=obs})/(B+1))); percentile of observed in the null
   reported as 100*#{null<obs}/B + 50*#{null==obs}/B.
3. Paired bootstrap 95% percentile CIs for all accuracy differences: B=10,000 item resamples, PCG64(20261006), one index
   draw per replicate shared across all differences. Random-gate reference per item = exact expectation under uniform
   k-subset gating (valid: p=k/|V| mitigated; invalid: mitigated). Token F1 (official, max over gold) as secondary,
   same bootstrap.
4. Cost accounting per arm (arm-only and pipeline): GPU-s (sum of logged model-call wall time), generated tokens,
   input tokens, model calls, retrieved passages; per item and total; accuracy-per-cost table.
   Pipelines: E0 = E0; E1 = E0 + E1 (E1 consumes A1); E3 = E3; E2 = E0 + E1 on mitigated; E4 = E0 + E3 on mitigated.
5. Descriptives: realized flag rates, invalid/abstain counts per arm, BM25 answer-string recall@5 (passage text, no title).
6. em_validity_plan item 3: blind EM-audit sample, 50 valid non-abstain EM=0 outputs per arm E0, E1, E3 (all if fewer),
   PCG64(20261005) over sorted item_ids per arm, arms in order E0, E1, E3, then shuffled with the same generator;
   blind CSV (arm hidden, gold aliases shown) + separate key. Item 4 (EM-bias flag, |FN(E1) - FN(E3)| > 10 pp) is
   computed ONLY by --em-audit-labels after the PI returns labels.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from _common import ROOT, write_json
from gvr.config import load_config, sha256_file
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.gate import apply_gate, rates
from gvr.random_gate import random_gate
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import answer_string_recall, exact_match, is_abstain, max_f1
from gvr.test_runner import effective_calls, done_units

BOOT_B, BOOT_SEED, RG_SEEDS, AUDIT_SEED, AUDIT_N = 10_000, 20261006, range(1000), 20261005, 50
ARMS = ("E0", "E1", "E3")


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired booleans a, b. Returns (n10, n01, p)."""
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b)); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, p)


def holm(pvals):
    names = sorted(pvals, key=pvals.get); m = len(names); out, run = {}, 0.0
    for i, nm in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[nm]))
        out[nm] = run
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None, None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - w), min(1.0, c + w)


def load(runs, split, cfg, require_units):
    calls = []
    for r in runs:
        d = ROOT / "runs" / r
        cs = effective_calls(d)
        if require_units:
            done = done_units(d)
            cs = [c for c in cs if (c["item_id"], c["arm"]) in done]
        calls += cs
    ids = sorted({c["item_id"] for c in calls})
    qs = load_questions(cfg, split, ids)
    return calls, ids, qs, derive_items(calls, cfg, qs)


def costs(calls, ids):
    per = {a: {i: dict(gpu_s=0.0, gen_tokens=0, in_tokens=0, calls=0, passages=0) for i in ids} for a in ARMS}
    for c in calls:
        if c.get("arm") not in per:
            continue
        r = per[c["arm"]][c["item_id"]]
        if c["step"] == "retrieve":
            r["passages"] += len((c.get("retrieval") or {}).get("passage_ids") or [])
            continue
        r["gpu_s"] += (c.get("t_end") or 0) - (c.get("t_start") or 0)
        r["gen_tokens"] += c.get("output_tokens") or 0
        r["in_tokens"] += c.get("input_tokens") or 0
        r["calls"] += 1
    return per


def add(*ds):
    return {k: sum(d[k] for d in ds) for k in ds[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True, choices=["train", "validation"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="dev data; output marked NOT A RESULT")
    ap.add_argument("--em-audit-labels", default=None, help="labelled audit CSV (after PI labelling) -> item-4 flag")
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    gate = json.load(open(ROOT / "configs/gate.json"))
    assert gate["provisional"] is False
    if a.split == "validation":
        assert not a.dry_run and all(r.startswith("test-") for r in a.runs)
    calls, ids, qs, items = load(a.runs, a.split, cfg, require_units=a.split == "validation")
    for i in ids:
        for arm in ARMS:
            assert (i, arm) in items, f"missing {arm} for {i}"
    gold = load_gold(cfg, a.split, ids)
    GT = {i: ground_truths(gold[i]) for i in ids}
    abst = cfg["scoring"]["abstain_list"]

    def lab(arm):
        em, f1, inv, ab = {}, {}, {}, {}
        for i in ids:
            it = items[(i, arm)]
            v = not it["invalid"]
            inv[i] = not v
            ab[i] = v and is_abstain(it["answer"], abst)
            em[i] = bool(v and exact_match(it["answer"], GT[i]))
            f1[i] = max_f1(it["answer"], GT[i]) if v else 0.0
        return em, f1, inv, ab
    L = {arm: lab(arm) for arm in ARMS}

    # 1. gate + composition
    e0 = [dict(item_id=i, invalid=items[(i, "E0")]["invalid"], u=items[(i, "E0")]["u"]) for i in ids]
    dec = apply_gate(e0, gate["tau"])
    R = rates(dec)
    mit = {i: dec[i]["mitigate"] for i in ids}
    comp = {}
    for name, src in (("E2", "E1"), ("E4", "E3")):
        ans = {i: items[(i, src)]["answer"] if mit[i] else items[(i, "E0")]["answer"] for i in ids}
        for i in ids:
            if not mit[i]:
                assert ans[i] == items[(i, "E0")]["answer"]
        comp[name] = dict(em={i: L[src][0][i] if mit[i] else L["E0"][0][i] for i in ids},
                          f1={i: L[src][1][i] if mit[i] else L["E0"][1][i] for i in ids})
    EM = {arm: np.array([L[arm][0][i] for i in ids]) for arm in ARMS}
    F1 = {arm: np.array([L[arm][1][i] for i in ids]) for arm in ARMS}
    for nm in ("E2", "E4"):
        EM[nm] = np.array([comp[nm]["em"][i] for i in ids]); F1[nm] = np.array([comp[nm]["f1"][i] for i in ids])

    # 2. tests
    k = R["k_flagged_valid"]
    n10, n01, p_primary = mcnemar_exact(EM["E2"], EM["E4"])
    sec = {}
    for nm, (x, y) in {"E2_vs_E0": ("E2", "E0"), "E4_vs_E0": ("E4", "E0")}.items():
        s10, s01, p = mcnemar_exact(EM[x], EM[y])
        sec[nm] = dict(n10=s10, n01=s01, p=p, diff=float(EM[x].mean() - EM[y].mean()))
    nulls = {}
    for nm, src in (("E2", "E1"), ("E4", "E3")):
        accs, rr = random_gate(e0, {i: L["E0"][0][i] for i in ids}, {i: L[src][0][i] for i in ids}, k, RG_SEEDS)
        assert all(x == R["nominal_rate"] for x in rr)
        obs = float(EM[nm].mean()); B = len(accs)
        p = min(1.0, 2 * min((1 + np.sum(accs >= obs)) / (B + 1), (1 + np.sum(accs <= obs)) / (B + 1)))
        pct = 100 * (np.sum(accs < obs) + 0.5 * np.sum(accs == obs)) / B
        nulls[nm] = accs
        sec[f"{nm}_vs_random_gate"] = dict(observed=obs, null_mean=float(accs.mean()), null_sd=float(accs.std(ddof=1)),
                                          null_p2_5=float(np.percentile(accs, 2.5)), null_p97_5=float(np.percentile(accs, 97.5)),
                                          percentile_of_observed=float(pct), p=float(p), diff=obs - float(accs.mean()))
    adj = holm({nm: v["p"] for nm, v in sec.items()})
    for nm in sec:
        sec[nm]["p_holm"] = adj[nm]

    # 3. paired bootstrap
    valid = np.array([not items[(i, "E0")]["invalid"] for i in ids])
    pm = np.where(valid, k / max(int(valid.sum()), 1), 1.0)
    RGE = {"E2": pm * EM["E1"] + (1 - pm) * EM["E0"], "E4": pm * EM["E3"] + (1 - pm) * EM["E0"]}
    RGF = {"E2": pm * F1["E1"] + (1 - pm) * F1["E0"], "E4": pm * F1["E3"] + (1 - pm) * F1["E0"]}
    diffs = {"E2-E4": (EM["E2"], EM["E4"]), "E2-E0": (EM["E2"], EM["E0"]), "E4-E0": (EM["E4"], EM["E0"]),
             "E2-RG": (EM["E2"], RGE["E2"]), "E4-RG": (EM["E4"], RGE["E4"]),
             "E1-E0": (EM["E1"], EM["E0"]), "E3-E0": (EM["E3"], EM["E0"])}
    fdiffs = {"E2-E4": (F1["E2"], F1["E4"]), "E2-E0": (F1["E2"], F1["E0"]), "E4-E0": (F1["E4"], F1["E0"]),
              "E2-RG": (F1["E2"], RGF["E2"]), "E4-RG": (F1["E4"], RGF["E4"])}
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED)); n = len(ids)
    bs_em = {d: np.empty(BOOT_B) for d in diffs}; bs_f1 = {d: np.empty(BOOT_B) for d in fdiffs}
    for b in range(BOOT_B):
        idx = rng.integers(0, n, n)
        for d, (x, y) in diffs.items():
            bs_em[d][b] = x[idx].mean() - y[idx].mean()
        for d, (x, y) in fdiffs.items():
            bs_f1[d][b] = x[idx].mean() - y[idx].mean()
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    boot = dict(em={d: dict(diff=float(x.mean() - y.mean()), ci95=ci(bs_em[d])) for d, (x, y) in diffs.items()},
                f1={d: dict(diff=float(x.mean() - y.mean()), ci95=ci(bs_f1[d])) for d, (x, y) in fdiffs.items()})

    # 4. costs
    per = costs(calls, ids)
    pipe = {"E0": {i: per["E0"][i] for i in ids}, "E1": {i: add(per["E0"][i], per["E1"][i]) for i in ids},
            "E3": {i: per["E3"][i] for i in ids},
            "E2": {i: add(per["E0"][i], per["E1"][i]) if mit[i] else per["E0"][i] for i in ids},
            "E4": {i: add(per["E0"][i], per["E3"][i]) if mit[i] else per["E0"][i] for i in ids}}
    cost = {}
    for arm, d in pipe.items():
        tot = {m: float(sum(v[m] for v in d.values())) for m in ("gpu_s", "gen_tokens", "in_tokens", "calls", "passages")}
        acc = float(EM[arm].mean())
        cost[arm] = dict(total=tot, per_item={m: v / n for m, v in tot.items()}, em_accuracy=acc,
                         em_per_gpu_s=acc / (tot["gpu_s"] / n) if tot["gpu_s"] else None,
                         em_per_1k_tokens=acc / ((tot["gen_tokens"] + tot["in_tokens"]) / n / 1000) if tot["in_tokens"] else None)
    arm_only = {arm: {m: float(sum(per[arm][i][m] for i in ids)) for m in ("gpu_s", "gen_tokens", "in_tokens", "calls", "passages")} for arm in ARMS}

    # 5. descriptives
    rec5 = float(np.mean([answer_string_recall(items[(i, "E3")].get("passages_full") or [], GT[i]) for i in ids]))
    desc = dict(n_items=n, gate=dict(tau=gate["tau"], **R), invalid={arm: int(sum(L[arm][2].values())) for arm in ARMS},
                abstain={arm: int(sum(L[arm][3].values())) for arm in ARMS}, bm25_answer_string_recall5=rec5,
                em_accuracy={arm: float(EM[arm].mean()) for arm in EM}, f1_mean={arm: float(F1[arm].mean()) for arm in F1})

    # 6. EM-audit sample (item 3)
    rng_a = np.random.Generator(np.random.PCG64(AUDIT_SEED)); sample = []
    for arm in ARMS:
        frame = sorted(i for i in ids if not L[arm][2][i] and not L[arm][3][i] and not L[arm][0][i])
        pick = frame if len(frame) <= AUDIT_N else sorted(rng_a.choice(frame, AUDIT_N, replace=False).tolist())
        sample += [(i, arm) for i in pick]
    order = rng_a.permutation(len(sample))
    key = []
    with open(out / "em_audit_blind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "gold_aliases", "prediction", "human_label(CORRECT/INCORRECT/UNSURE)"])
        for j, oi in enumerate(order):
            i, arm = sample[oi]
            w.writerow([f"a{j:03d}", qs[i], " | ".join(sorted(set(gold[i]["aliases"]))), items[(i, arm)]["answer"], ""])
            key.append(dict(row_id=f"a{j:03d}", item_id=i, arm=arm))
    write_json(out / "em_audit_key.json", dict(rows=key, seed=AUDIT_SEED, n_per_arm={arm: sum(r["arm"] == arm for r in key) for arm in ARMS},
                                               blind_csv_sha256=sha256_file(out / "em_audit_blind.csv")))
    item4 = None
    if a.em_audit_labels:
        labels = {r["row_id"]: r["human_label(CORRECT/INCORRECT/UNSURE)"].strip().upper()
                  for r in csv.DictReader(open(a.em_audit_labels, newline="", encoding="utf-8"))}
        fn = {}
        for arm in ARMS:
            ls = [labels[r["row_id"]] for r in key if r["arm"] == arm]
            kk, nn = sum(x == "CORRECT" for x in ls), sum(x in ("CORRECT", "INCORRECT") for x in ls)
            p, lo, hi = wilson(kk, nn)
            p_em0 = float(np.mean([not L[arm][2][i] and not L[arm][3][i] and not L[arm][0][i] for i in ids]))
            fn[arm] = dict(fn_rate=p, wilson95=[lo, hi], n=nn, unsure=sum(x == "UNSURE" for x in ls),
                           adj_accuracy=float(EM[arm].mean()) + (p or 0) * p_em0,
                           adj_interval=[float(EM[arm].mean()) + (lo or 0) * p_em0, float(EM[arm].mean()) + (hi or 0) * p_em0])
        item4 = dict(fn=fn, e1_minus_e3_pp=100 * (fn["E1"]["fn_rate"] - fn["E3"]["fn_rate"]),
                     em_bias_flag=abs(100 * (fn["E1"]["fn_rate"] - fn["E3"]["fn_rate"])) > 10,
                     labels_sha256=sha256_file(a.em_audit_labels))

    res = dict(status="DRY RUN - dev data, NOT A RESULT" if a.dry_run else "MAIN RESULT",
               runs=a.runs, split=a.split, script_sha256=sha256_file(Path(__file__)), gate_sha256=sha256_file(ROOT / "configs/gate.json"),
               primary=dict(test="exact McNemar E2 vs E4 (EM)", n10=n10, n01=n01, p=p_primary, diff=float(EM["E2"].mean() - EM["E4"].mean())),
               secondary_holm=sec, bootstrap=dict(B=BOOT_B, seed=BOOT_SEED, **boot), cost_pipeline=cost, cost_arm_only=arm_only,
               descriptives=desc, em_audit=dict(sample_rows=len(key), item4=item4))
    write_json(out / "results.json", res)
    print(json.dumps(dict(status=res["status"], n=n, primary=res["primary"], flag_rate=R["nominal_rate"],
                          audit_rows=len(key)), indent=1))


if __name__ == "__main__":
    main()
