"""Item 2: does our EM ground-truth set match the official TriviaQA evaluation? (train split only; no test access)."""
import json
import pyarrow.parquet as pq
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import download_split_file, load_questions
from gvr.derive import derive_items
from gvr.scoring.gold import ground_truths
from gvr.scoring.normalize import normalize_answer

cfg = load_config()
rows = pq.read_table(download_split_file(cfg, "train"), columns=["question_id", "answer"]).to_pylist()


def official_gts(a):   # triviaqa_evaluation.get_ground_truths (HF has no HumanAnswers field)
    assert "human_answers" not in a and "HumanAnswers" not in a
    return list(a["normalized_aliases"])


def official_em(pred, a):
    return any(normalize_answer(pred) == normalize_answer(g) for g in official_gts(a))


n = len(rows); extra_items = 0; extra_strings = 0; nonidem = 0; examples = []
for r in rows:
    a = r["answer"]
    na = set(a["normalized_aliases"])
    ours = ground_truths(a)
    off = {normalize_answer(g) for g in na} - {""}
    if ours != off:
        extra_items += 1
        d = sorted(ours ^ off)
        extra_strings += len(d)
        if len(examples) < 8:
            examples.append(dict(qid=r["question_id"], ours_minus_official=sorted(ours - off), official_minus_ours=sorted(off - ours)))
    nonidem += sum(normalize_answer(g) != g for g in na)

dev = dict(n=0, disagree=[])
for run in ("pilot-01", "judgedev-01"):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    gold = {r["question_id"]: r["answer"] for r in rows if r["question_id"] in set(ids)}
    for (iid, arm), it in derive_items(calls, cfg, load_questions(cfg, "train", ids)).items():
        if it["invalid"]:
            continue
        dev["n"] += 1
        ours = normalize_answer(it["answer"]) in ground_truths(gold[iid])
        off = official_em(it["answer"], gold[iid])
        if ours != off:
            dev["disagree"].append(dict(run=run, item_id=iid, arm=arm, prediction=it["answer"], ours=ours, official=off))

scap = {"aliases": ["Examplebone (example blade)"], "normalized_aliases": ["examplebone example blade"]}   # synthetic annotated alias
res = dict(official_source="mandarjoshi90/triviaqa master: evaluation/triviaqa_evaluation.py get_ground_truths + exact_match_score",
           train_items=n, items_where_our_gt_set_differs=extra_items, differing_strings=extra_strings,
           normalized_aliases_not_idempotent=nonidem, examples=examples, dev_outputs_compared=dev["n"],
           dev_em_disagreements=dev["disagree"],
           scapula_example=dict(aliases=scap["aliases"], normalized_aliases=scap["normalized_aliases"],
                                official_em_for_Examplebone=official_em("Examplebone", scap)))
write_json(ROOT / "runs" / "em_official_check.json", res)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1)); print("examples:", json.dumps(examples[:5], indent=1))
