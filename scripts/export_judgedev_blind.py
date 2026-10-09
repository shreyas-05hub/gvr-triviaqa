"""Blind hand-label CSV for judge-dev (judge_rule_v2): all valid, non-abstain, EM=0 outputs of E0/E1/E3 plus
20 random EM=1 rows; shuffled (seed 20261004); arm hidden; key file kept separately. No judge involved."""
import csv
import hashlib
import json
import numpy as np
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.scoring.gold import load_gold, ground_truths
from gvr.scoring.metrics import exact_match, is_abstain

cfg = load_config()
jr = cfg["judge_rule_v2"]
assert hashlib.sha256(jr["text"].rstrip("\n").encode()).hexdigest() == jr["sha256"], "judge_rule_v2 hash mismatch"
seed, n_em1 = jr["judgedev"]["blind_seed"], jr["judgedev"]["n_em1_sanity"]
calls = read_jsonl(ROOT / "runs" / "judgedev-01" / "calls.jsonl")
ids = sorted({c["item_id"] for c in calls})
assert ids == sorted(json.load(open(ROOT / "data" / "judgedev_ids.json")))
qs = load_questions(cfg, "train", ids)
gold = load_gold(cfg, "train", ids)
items = derive_items(calls, cfg, qs)
em0, em1 = [], []
for (iid, arm), it in sorted(items.items()):
    if it["invalid"] or is_abstain(it["answer"], cfg["scoring"]["abstain_list"]):
        continue
    (em1 if exact_match(it["answer"], ground_truths(gold[iid])) else em0).append((iid, arm))
rng = np.random.Generator(np.random.PCG64(seed))
em1_pick = [em1[i] for i in sorted(rng.choice(len(em1), size=min(n_em1, len(em1)), replace=False))]
rows = [(k, False) for k in em0] + [(k, True) for k in em1_pick]
order = rng.permutation(len(rows))
key, out = [], []
for n, idx in enumerate(order):
    (iid, arm), em = rows[idx]
    rid = f"r{n:03d}"
    key.append(dict(row_id=rid, item_id=iid, arm=arm, em=em))
    out.append([rid, qs[iid], " | ".join(sorted(set(gold[iid]["aliases"]))), items[(iid, arm)]["answer"], ""])
path = ROOT / "reports" / "judge_handlabel_blind_judgedev.csv"
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "question", "gold_aliases", "prediction", "human_label(CORRECT/INCORRECT/UNSURE)"])
    w.writerows(out)
write_json(ROOT / "runs" / "judgedev-01" / "blind_key.json",
           dict(rows=key, n_em0=len(em0), n_em1_sampled=len(em1_pick), n_em1_total=len(em1), seed=seed,
                blind_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(f"blind CSV rows: {len(out)} (EM=0: {len(em0)}, EM=1 sanity: {len(em1_pick)}) -> {path}")
