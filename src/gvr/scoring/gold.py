"""Gold loading. Only gvr.scoring may import this module."""
import pyarrow.parquet as pq

from ..data import download_split_file
from ..guard import check_split
from .normalize import normalize_answer


def load_gold(cfg, split, ids, manifest_path=None):
    check_split(split)
    if split != "train":   # test_execution_rule: no gold for test until all 3 chunks complete + completeness check passed
        import json
        from ..paths import FREEZE_MANIFEST
        tc = json.load(open(manifest_path or FREEZE_MANIFEST)).get("test_completeness", {})
        if tc.get("passed") is not True:
            raise PermissionError("test gold locked: completeness check has not passed")
    path = download_split_file(cfg, split)
    rows = pq.read_table(path, columns=["question_id", "answer"]).to_pylist()
    want = set(ids)
    out = {r["question_id"]: r["answer"] for r in rows if r["question_id"] in want}
    missing = want - set(out)
    assert not missing, f"gold missing for {len(missing)} ids"
    return out


def ground_truths(answer: dict) -> set:
    """Official TriviaQA EM gold set (evaluation/triviaqa_evaluation.py get_ground_truths + exact_match_score):
    NormalizedAliases as-is plus HumanAnswers (absent in the HF dataset), each passed through normalize_answer at
    comparison time. Spec correction 2026-10-07 (previously also unioned normalize(aliases) and skipped re-normalizing)."""
    human = answer.get("human_answers") or answer.get("HumanAnswers") or []
    return {normalize_answer(g) for g in list(answer["normalized_aliases"]) + list(human)}


def gold_reprs(answer: dict):
    """String forms of gold field objects that must never appear in any prompt."""
    import json
    forms = [repr(answer), json.dumps(answer, ensure_ascii=False), json.dumps(answer)]
    for k in ("aliases", "normalized_aliases"):
        v = answer.get(k) or []
        if len(v) > 1:
            forms += [repr(v), json.dumps(v, ensure_ascii=False), ", ".join(v), "\n".join(v)]
    return forms


def audit_no_gold(calls, gold_by_id):
    """Assert no logged prompt contains any gold alias field object."""
    for c in calls:
        ans = gold_by_id.get(c["item_id"])
        if ans is None:
            continue
        for form in gold_reprs(ans):
            assert form not in (c.get("input_text") or ""), \
                f"gold field object leaked into prompt: item {c['item_id']} step {c['step']}"
