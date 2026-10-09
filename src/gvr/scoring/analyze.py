"""Score derived items, fit (provisional) gate, compose E2/E4, random gate. Descriptive only."""
import csv
import numpy as np

from ..compose import compose
from ..gate import fit_gate
from ..random_gate import random_gate, summarize
from .gold import ground_truths
from .metrics import label_item, flips, needs_judge, is_abstain, exact_match


def judge_requests(items, gold, questions, abstain_list):
    """Items needing a judge verdict: valid AND not abstain AND EM=0 (per arm E0/E1/E3)."""
    reqs = []
    for (iid, arm), it in sorted(items.items()):
        gts = ground_truths(gold[iid])
        valid = not it["invalid"]
        ab = valid and is_abstain(it["answer"], abstain_list)
        em = valid and exact_match(it["answer"], gts)
        if needs_judge(valid, ab, em):
            reqs.append(dict(item_id=iid, arm=arm, question=questions[iid], prediction=it["answer"],
                             gold_aliases=sorted(set(gold[iid]["aliases"]))))
    return reqs


def write_handlabel_csv(path, reqs, verdicts=None):
    """BLIND CSV for the USER to hand-label: no judge verdict column. Verdicts stay in judge.jsonl.
    Claude never fills the human_label column."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "arm", "question", "gold_aliases", "prediction", "human_label(CORRECT/INCORRECT)"])
        for r in reqs:
            w.writerow([r["item_id"], r["arm"], r["question"], " | ".join(r["gold_aliases"]), r["prediction"], ""])


def judge_agreement(labeled_rows, verdicts):
    """labeled_rows: dicts from the user's labeled CSV; verdicts: {(item_id, arm): bool}. -> (agreement, n)."""
    pairs = []
    for r in labeled_rows:
        lab = r["human_label(CORRECT/INCORRECT)"].strip().upper()
        if lab not in ("CORRECT", "INCORRECT"):
            continue
        k = (r["item_id"], r["arm"])
        if k in verdicts:
            pairs.append((lab == "CORRECT") == verdicts[k])
    return (sum(pairs) / len(pairs) if pairs else None), len(pairs)


def score_items(items, gold, verdicts, abstain_list, unparsed=(), judge_enabled=True):
    out = {}
    for (iid, arm), it in items.items():
        gts = ground_truths(gold[iid])
        lab = label_item(it["answer"], not it["invalid"], gts, abstain_list,
                         judge_correct=verdicts.get((iid, arm)),
                         passages=it.get("passages_full") if arm == "E3" else None, judge_enabled=judge_enabled)
        lab["judge_unparsed"] = lab["judged"] and (iid, arm) in set(unparsed)
        out[(iid, arm)] = {**it, **lab}
    return out


def by_arm(scored, arm):
    return {iid: it for (iid, a), it in scored.items() if a == arm}


def arm_summary(arm_items):
    v = list(arm_items.values())
    n = len(v)
    s = lambda k: sum(bool(x.get(k)) for x in v)
    return dict(N=n, em=s("em") / n, final=s("final") / n, f1=float(np.mean([x["f1"] for x in v])),
                invalid=s("invalid") / n, abstain=s("abstain") / n, judged=s("judged"), judge_unparsed=s("judge_unparsed"),
                hallucination_proxy=s("hallucination_proxy") / n, failed=s("failed"),
                cost_calls=sum(x["cost"]["n_calls"] for x in v),
                cost_retrievals=sum(x["cost"].get("n_retrievals", 0) for x in v),
                cost_in_tokens=sum(x["cost"]["input_tokens"] for x in v),
                cost_out_tokens=sum(x["cost"]["output_tokens"] for x in v),
                cost_wall_s=sum(x["cost"]["wall_s"] for x in v))


def flip_table(e0, arm):
    f = [flips(e0[i]["final"], arm[i]["final"]) for i in e0]
    return dict(right_to_wrong=sum(x["right_to_wrong"] for x in f),
                wrong_to_right=sum(x["wrong_to_right"] for x in f))


def analyze(scored, cfg, seeds=None):
    e0, e1, e3 = by_arm(scored, "E0"), by_arm(scored, "E1"), by_arm(scored, "E3")
    tau, dec, rates = fit_gate(list(e0.values()), cfg["gate"])
    e2 = compose(e0, e1, dec, "E2")
    e4 = compose(e0, e3, dec, "E4")
    lo, hi = cfg["random_gate"]["seeds"]
    seeds = seeds if seeds is not None else range(lo, hi + 1)
    k = rates["k_flagged_valid"]
    rg = {}
    for name, mit in (("E2", e1), ("E4", e3)):
        accs, rr = random_gate(list(e0.values()), {i: x["final"] for i, x in e0.items()},
                               {i: x["final"] for i, x in mit.items()}, k, seeds)
        assert all(r == rates["nominal_rate"] for r in rr), "random rate != real-gate rate"
        rg[name] = summarize(accs)
    arms = {"E0": e0, "E1": e1, "E3": e3, "E2": e2, "E4": e4}
    paraphrase = [v for x in e1.values() for v in x.get("verif", [])]
    return dict(
        tau=tau, gate_rates=rates, decisions=dec, random_gate=rg,
        summary={a: arm_summary(v) for a, v in arms.items()},
        flips={a: flip_table(e0, arms[a]) for a in ("E1", "E3", "E2", "E4")},
        recall5=float(np.mean([x["answer_string_recall5"] for x in e3.values()])) if e3 else None,
        e3_truncated=sum(bool(x.get("truncated")) for x in e3.values()),
        u_valid=[x["u"] for x in e0.values() if not x["invalid"]],
        paraphrase=dict(n=len(paraphrase), flagged=sum(v["paraphrase_flag"] for v in paraphrase)),
        sv_plan_invalid=sum(bool(x.get("sv_plan_invalid")) for x in e1.values()),
    )


def evaluate_judge(labeled_rows, blind_rows, verdicts, em_by_key, threshold):
    """Applies the pre-registered judge_decision_rule. labeled_rows must equal blind_rows except the label column.
    verdicts / em_by_key: {(item_id, arm): bool}. UNSURE rows are reported and excluded from agreement."""
    LAB = "human_label(CORRECT/INCORRECT)"
    assert len(labeled_rows) == len(blind_rows), "row count changed"
    for a_, b_ in zip(labeled_rows, blind_rows):
        assert {k: v for k, v in a_.items() if k != LAB} == {k: v for k, v in b_.items() if k != LAB}, \
            f"labeled CSV differs from blind CSV outside the label column at {a_['item_id']}/{a_['arm']}"
    counts = dict(CORRECT=0, INCORRECT=0, UNSURE=0, blank_or_other=0)
    fa = fr = agree = em_agree = n = 0
    per_arm = {}
    for r in labeled_rows:
        lab = r[LAB].strip().upper()
        k = (r["item_id"], r["arm"])
        if lab not in ("CORRECT", "INCORRECT", "UNSURE"):
            counts["blank_or_other"] += 1
            continue
        counts[lab] += 1
        if lab == "UNSURE":
            continue
        human, judge, em = lab == "CORRECT", verdicts[k], em_by_key[k]
        n += 1
        agree += judge == human
        em_agree += em == human
        fa += judge and not human
        fr += (not judge) and human
        pa = per_arm.setdefault(r["arm"], dict(n=0, agree=0, false_accept=0, false_reject=0, em_agree=0,
                                               human_correct=0))
        pa["em_agree"] += em == human; pa["human_correct"] += human
        pa["n"] += 1; pa["agree"] += judge == human; pa["false_accept"] += judge and not human
        pa["false_reject"] += (not judge) and human
    agreement = agree / n if n else None
    case = None if agreement is None else (1 if agreement >= threshold else 2)
    return dict(case=case, agreement=agreement, n_compared=n, label_counts=counts, false_accept=fa,
                false_reject=fr, em_vs_labels_agreement=em_agree / n if n else None, per_arm=per_arm,
                threshold=threshold)
