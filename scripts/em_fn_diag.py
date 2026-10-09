"""EM false-negative diagnostic (dev data only, descriptive). Deterministic category rules; rows no rule catches are
written out for manual categorization (logged as such). No accuracy is computed."""
import csv
import difflib
import hashlib
import json
import re
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.normalize import normalize_answer, contains_token_seq

cfg = load_config()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
h = lambda p: hashlib.sha256(open(ROOT / p, "rb").read()).hexdigest()


def load_run(run):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    qs = load_questions(cfg, "train", ids)
    return derive_items(calls, cfg, qs), load_gold(cfg, "train", ids), qs


rows = []
P = "reports/judge_handlabel_blind_pilot-01_LABELED.csv"
assert h(P) == json.load(open(ROOT / "runs/pilot-01/judge_decision.json"))["labels_sha256"]
items, gold, qs = load_run("pilot-01")
for r in csv.DictReader(open(ROOT / P, newline="", encoding="utf-8")):
    if r["human_label(CORRECT/INCORRECT)"].strip().upper() == "CORRECT":
        k = (r["item_id"], r["arm"])
        rows.append(dict(source="pilot-01", item_id=k[0], arm=k[1], question=qs[k[0]], prediction=items[k]["answer"],
                         aliases=sorted(set(gold[k[0]]["aliases"])), gts=sorted(ground_truths(gold[k[0]]))))
rec = man["judgedev_labels"]["primary"]
assert h(rec["file"]) == rec["sha256"]
key = {r["row_id"]: r for r in json.load(open(ROOT / "runs/judgedev-01/blind_key.json"))["rows"]}
items, gold, qs = load_run("judgedev-01")
for r in csv.DictReader(open(ROOT / rec["file"], newline="", encoding="utf-8")):
    kr = key[r["row_id"]]
    if r["human_label(CORRECT/INCORRECT/UNSURE)"] == "CORRECT" and not kr["em"]:
        k = (kr["item_id"], kr["arm"])
        rows.append(dict(source="judgedev-01", item_id=k[0], arm=k[1], question=qs[k[0]], prediction=items[k]["answer"],
                         aliases=sorted(set(gold[k[0]]["aliases"])), gts=sorted(ground_truths(gold[k[0]]))))
assert len(rows) == 65, len(rows)

NUM = re.compile(r"\d")
for r in rows:
    npred = normalize_answer(r["prediction"])
    verbose = [g for g in r["gts"] if contains_token_seq(r["prediction"], g) and len(npred.split()) > len(g.split())]
    squash = lambda s: re.sub(r"[\s\-]", "", s)
    ratio = max(difflib.SequenceMatcher(None, npred, g).ratio() for g in r["gts"])
    if verbose:
        r.update(category="VERBOSE", method="rule: alias is normalized token-subsequence of longer prediction",
                 evidence=verbose[0])
    elif any(squash(npred) == squash(g) for g in r["gts"]):
        r.update(category="VARIANT", method="rule: equal after removing spaces/hyphens", evidence="")
    elif ratio >= 0.8:
        r.update(category="VARIANT", method=f"rule: char similarity {ratio:.2f} >= 0.80 to an alias", evidence="")
    else:
        r.update(category=None, method="manual", evidence=f"max char similarity {ratio:.2f}")
    r["max_char_ratio"] = ratio
write_json(ROOT / "runs/em_fn_diag/rows_auto.json", rows)
for r in rows:
    if r["category"] is None:
        print(json.dumps(dict(src=r["source"][:5], item=r["item_id"], arm=r["arm"], q=r["question"][:90],
                              pred=r["prediction"], aliases=r["aliases"][:8]), ensure_ascii=False))
print("auto:", {c: sum(r["category"] == c for r in rows) for c in ("VERBOSE", "VARIANT", None)})
