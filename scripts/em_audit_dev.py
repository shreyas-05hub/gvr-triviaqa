"""em_validity_plan item 2: dev EM audit from existing human labels (DESCRIPTIVE; train-split dev data only)."""
import csv
import hashlib
import json
import math
import yaml
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import exact_match, is_abstain

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--final-labels", default="primary", help="freeze_manifest judgedev_labels key for the final labels")
ap.add_argument("--out", default="em_audit_dev.json")
args = ap.parse_args()
cfg = load_config()
plan = cfg["em_validity_plan"]
assert hashlib.sha256(plan["text"].rstrip("\n").encode()).hexdigest() == plan["sha256"], "plan hash"
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
h = lambda p: hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest()
ARMS = ("E0", "E1", "E3")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - w), min(1.0, c + w)


def unlabelled_em0(run):
    """Invalid or abstain EM=0 outputs per arm (not labelled; not correct by definition)."""
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
    gold = load_gold(cfg, "train", ids)
    out = {a: 0 for a in ARMS}
    em_lookup = {}
    for (iid, arm), it in items.items():
        em = (not it["invalid"]) and exact_match(it["answer"], ground_truths(gold[iid]))
        em_lookup[(iid, arm)] = em
        if not em and (it["invalid"] or is_abstain(it["answer"], cfg["scoring"]["abstain_list"])):
            out[arm] += 1
    return out, em_lookup


# pilot-01 final labels (only labels for pilot); arm column present
P = "reports/judge_handlabel_blind_pilot-01_LABELED.csv"
pj = json.load(open(ROOT / "runs/pilot-01/judge_decision.json"))
assert h(P) == pj["labels_sha256"], "pilot labels hash"
pil_un, pil_em = unlabelled_em0("pilot-01")
pilot = [(r["arm"], r["human_label(CORRECT/INCORRECT)"].strip().upper(), r["item_id"])
         for r in csv.DictReader(open(ROOT / P, newline="", encoding="utf-8"))]
assert all(not pil_em[(i, a)] for a, _, i in pilot), "pilot labelled row with EM=1"

# judge-dev: key gives arm + EM
key = json.load(open(ROOT / "runs/judgedev-01/blind_key.json"))
kmap = {r["row_id"]: r for r in key["rows"]}
jd_un, jd_em = unlabelled_em0("judgedev-01")
assert all(jd_em[(r["item_id"], r["arm"])] == r["em"] for r in key["rows"]), "key EM mismatch"


def jd_labels(which):
    rec = man["judgedev_labels"][which]
    assert h(rec["file"]) == rec["sha256"], f"judge-dev {which} labels hash"
    lab = {r["row_id"]: r["human_label(CORRECT/INCORRECT/UNSURE)"]
           for r in csv.DictReader(open(ROOT / rec["file"], newline="", encoding="utf-8"))}
    em0 = [(kmap[r]["arm"], lab[r], kmap[r]["item_id"]) for r in lab if not kmap[r]["em"]]
    em1 = [lab[r] for r in lab if kmap[r]["em"]]
    return em0, em1, rec["sha256"]


def fn_table(rows):
    t = {}
    for arm in ARMS + ("ALL",):
        sel = [l for a, l, _ in rows if arm == "ALL" or a == arm]
        k, n = sum(l == "CORRECT" for l in sel), sum(l in ("CORRECT", "INCORRECT") for l in sel)
        p, lo, hi = wilson(k, n)
        t[arm] = dict(human_correct=k, n_labelled=n, unsure=sum(l == "UNSURE" for l in sel),
                      fn_rate=p, wilson95=[lo, hi])
    return t


def fp(em1):
    k, n = sum(l == "INCORRECT" for l in em1), sum(l in ("CORRECT", "INCORRECT") for l in em1)
    p, lo, hi = wilson(k, n)
    return dict(human_incorrect=k, n_labelled=n, unsure=sum(l == "UNSURE" for l in em1), fp_rate=p, wilson95=[lo, hi])


res = dict(note="DESCRIPTIVE; train-split development data only (pilot-01, judge-dev); not test evidence",
           plan_sha256=plan["sha256"], unlabelled_invalid_or_abstain_em0=dict(pilot=pil_un, judgedev=jd_un))
res["judgedev_final_labels_key"] = args.final_labels
res["em_gold_set"] = cfg["scoring"].get("em_gold_set", "protocol_stage7")
for name, which in (("final", args.final_labels), ("sensitivity_judgedev_original", "original")):
    em0, em1, sha = jd_labels(which)
    res[name] = dict(judgedev_labels_sha256=sha, pilot_labels_sha256=pj["labels_sha256"],
                     fn_pilot=fn_table(pilot), fn_judgedev=fn_table(em0), fn_pooled=fn_table(pilot + em0),
                     fp_judgedev_em1=fp(em1))
write_json(ROOT / "runs" / args.out, res)
print(json.dumps(res, indent=1))
