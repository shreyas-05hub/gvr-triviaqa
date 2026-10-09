"""Applies judge_rule_v2 exactly (local). Primary labels 5b911ae0...; sensitivity check vs original f05aaa35..."""
import csv
import hashlib
import json
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config

cfg = load_config()
jr = cfg["judge_rule_v2"]
assert hashlib.sha256(jr["text"].rstrip("\n").encode()).hexdigest() == jr["sha256"], "rule hash"
acc = jr["acceptance"]
man = json.load(open(ROOT / "configs" / "freeze_manifest.json"))
LAB = "human_label(CORRECT/INCORRECT/UNSURE)"
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
key = json.load(open(ROOT / "runs/judgedev-01/blind_key.json"))
kmap = {r["row_id"]: r for r in key["rows"]}
judge = {j["row_id"]: j for j in read_jsonl(ROOT / "runs/judgedev-01/judge_v2.jsonl")}
assert len(judge) == 170 and set(judge) == {r["row_id"] for r in key["rows"] if not r["em"]}


def labels(which):
    rec = man["judgedev_labels"][which]
    p = ROOT / rec["file"]
    assert h(p) == rec["sha256"], f"{which} labels hash changed"
    return {r["row_id"]: r[LAB] for r in csv.DictReader(open(p, newline="", encoding="utf-8"))}, rec["sha256"]


def evaluate(lab):
    unsure = sum(lab[r] == "UNSURE" for r in judge)
    n = agree = fa = fr = n_inc = n_cor = unparsed = 0
    arms = {}
    for rid, j in judge.items():
        if lab[rid] == "UNSURE":
            continue
        human = lab[rid] == "CORRECT"
        jv = bool(j["verdict"])          # unparsed -> False (counted)
        unparsed += j["verdict"] is None
        n += 1; agree += jv == human
        n_inc += not human; n_cor += human
        fa += jv and not human; fr += (not jv) and human
        a = arms.setdefault(j["arm"], dict(n=0, agree=0, false_accept=0, false_reject=0))
        a["n"] += 1; a["agree"] += jv == human; a["false_accept"] += jv and not human; a["false_reject"] += (not jv) and human
    for a in arms.values():
        a["agreement"] = a["agree"] / a["n"]
    res = dict(n_compared=n, unsure_excluded=unsure, judge_unparsed=unparsed, agreement=agree / n,
               false_accept=fa, human_incorrect=n_inc, false_accept_rate=fa / n_inc if n_inc else None,
               false_reject=fr, human_correct=n_cor, false_reject_rate=fr / n_cor if n_cor else None, per_arm=arms)
    res["criteria"] = dict(overall_agreement=res["agreement"] >= acc["overall_agreement_min"],
                           false_accept_rate=res["false_accept_rate"] <= acc["false_accept_rate_max"],
                           per_arm_agreement=all(a["agreement"] >= acc["per_arm_agreement_min"] for a in arms.values()))
    res["pass"] = all(res["criteria"].values())
    em1 = [r["row_id"] for r in key["rows"] if r["em"]]
    res["em_sanity"] = dict(n_em1_rows=len(em1), labelled_INCORRECT=sum(lab[r] == "INCORRECT" for r in em1),
                            labelled_UNSURE=sum(lab[r] == "UNSURE" for r in em1),
                            incorrect_row_ids=[r for r in em1 if lab[r] == "INCORRECT"])
    return res


prim, sp = labels("primary")
orig, so = labels("original")
out = dict(rule_sha256=jr["sha256"], acceptance=acc, candidate="A", judge_model=jr["candidate_A"]["model_id"],
           judge_revision=jr["candidate_A"]["revision"], prompt_sha=man["judge_v2_prompt"]["sha256"],
           judge_outputs_sha256=h(ROOT / "runs/judgedev-01/judge_v2.jsonl"),
           primary=dict(labels_sha256=sp, **evaluate(prim)),
           sensitivity_original_labels=dict(labels_sha256=so, **evaluate(orig)))
out["decision_basis"] = "primary labels 5b911ae0 (pre-committed); original-label results are sensitivity only"
out["sensitivity_pass_fail_disagrees"] = out["primary"]["pass"] != out["sensitivity_original_labels"]["pass"]
out["decision"] = "candidate A ACCEPTED (judge = secondary metric)" if out["primary"]["pass"] else \
    "candidate A FAILED -> stop; candidate B requires user approval of the model"
write_json(ROOT / "runs" / "judgedev" / "judge_decision_v2.json", out)
print(json.dumps(out, indent=2))
