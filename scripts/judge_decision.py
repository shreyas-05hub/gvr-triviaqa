"""Applies the pre-registered judge_decision_rule to the user's labeled CSV. Never edits labels or prompts."""
import csv
import hashlib
import json
import yaml
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.scoring.analyze import evaluate_judge
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import exact_match

cfg = load_config()
rule = cfg["judge_decision_rule"]
assert hashlib.sha256(rule["text"].rstrip("\n").encode()).hexdigest() == rule["sha256"], "rule text hash mismatch"
labeled = list(csv.DictReader(open(ROOT / rule["labels_file"])))
blind = list(csv.DictReader(open(ROOT / "reports" / "judge_handlabel_blind_pilot-01.csv")))
verdicts = {(j["item_id"], j["arm"]): bool(j["verdict"]) for j in read_jsonl(ROOT / rule["verdicts_file"])}
calls = read_jsonl(ROOT / "runs/pilot-01/calls.jsonl")
ids = sorted({c["item_id"] for c in calls})
items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
gold = load_gold(cfg, "train", ids)
em = {k: exact_match(it["answer"], ground_truths(gold[k[0]])) for k, it in items.items()}
res = evaluate_judge(labeled, blind, verdicts, em, rule["threshold_agreement"])
res["rule_sha256"] = rule["sha256"]
res["labels_sha256"] = hashlib.sha256(open(ROOT / rule["labels_file"], "rb").read()).hexdigest()
write_json(ROOT / "runs" / "pilot-01" / "judge_decision.json", res)
print(json.dumps(res, indent=2))
