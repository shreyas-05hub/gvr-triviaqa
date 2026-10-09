import pytest
from gvr.scoring.analyze import evaluate_judge

LAB = "human_label(CORRECT/INCORRECT)"


def rows(labels):
    return [dict(item_id=f"q{i}", arm="E0", question="q", gold_aliases="a", prediction="p", **{LAB: l})
            for i, l in enumerate(labels)]


def test_evaluate_judge_cases_and_counts():
    labels = ["CORRECT"] * 8 + ["INCORRECT", "INCORRECT", "UNSURE"]
    v = {(f"q{i}", "E0"): True for i in range(11)}       # judge says CORRECT everywhere
    em = {(f"q{i}", "E0"): False for i in range(11)}     # judged rows all have EM=0
    r = evaluate_judge(rows(labels), rows([""] * 11), v, em, 0.9)
    assert r["n_compared"] == 10 and r["label_counts"]["UNSURE"] == 1
    assert r["agreement"] == 0.8 and r["case"] == 2
    assert r["false_accept"] == 2 and r["false_reject"] == 0 and r["em_vs_labels_agreement"] == 0.2
    r1 = evaluate_judge(rows(["CORRECT"] * 10), rows([""] * 10), v, em, 0.9)
    assert r1["case"] == 1 and r1["agreement"] == 1.0


def test_labeled_csv_must_match_blind():
    a, b = rows(["CORRECT"]), rows([""])
    a[0]["prediction"] = "edited"
    with pytest.raises(AssertionError):
        evaluate_judge(a, b, {("q0", "E0"): True}, {("q0", "E0"): False}, 0.9)


def test_parse_check_is_parse_only():
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "judge_parse_check.py").read_text()
    assert "LABELED" not in src and "judge_decision" not in src and "evaluate_judge" not in src
    assert "judgedev-01\" / \"calls" not in src and '"pilot-01" / "calls.jsonl"' in src
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "judge_agreement" not in names


def test_judgedev_judge_never_reads_labels():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "judge_judgedev.py").read_text()
    assert "LABELED" not in src and "human_label" not in src and "reports/" not in src
    assert 'targets = [r for r in key["rows"] if not r["em"]]' in src and "len(targets) == 170" in src


def test_judgedev_judge_never_reads_labels():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "judge_judgedev.py").read_text()
    for banned in ("LABELED", "human_label", "judge_decision", "evaluate_judge", "reports/"):
        assert banned not in src, banned
    assert 'not r["em"]' in src and "== 170" in src   # EM=0 targets only
