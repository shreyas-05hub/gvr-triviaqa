import pytest
from gvr.parsing import parse_answer, parse_plan


@pytest.mark.parametrize("raw,ans,inv", [
    ("Paris", "Paris", False), ("Answer: Paris.", "Paris", False), ("ANSWER:Paris!", "Paris", False),
    ("\n\n  Paris  \nbecause...", "Paris", False), ("answer: ", "", True), ("", "", True),
    ("...", "", True), ("one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten", False),
    ("one two three four five six seven eight nine ten eleven", "one two three four five six seven eight nine ten eleven", True),
    ("Washington, D.C.", "Washington, D.C", False), ("1066", "1066", False),
    ("Answer: answer: x", "answer: x", False),
])
def test_parse_answer(raw, ans, inv):
    assert parse_answer(raw, 10) == (ans, inv)


def test_parse_plan():
    assert parse_plan("1. Q one?\n2) Q two?\n\n- Q three?\nQ four?") == ["Q one?", "Q two?", "Q three?"]
    assert parse_plan("\n  \n") == []
    assert parse_plan("Only one?") == ["Only one?"]


@pytest.mark.parametrize("raw,ans", [
    ("The Example Picture (1957 film)", "The Example Picture (1957 film)"),
    ("Battle of Exampleford (1777).", "Battle of Exampleford (1777)"),
    ("Examplebone (example blade)!", "Examplebone (example blade)"),
    ("List [a, b]", "List [a, b]"),
    ("unbalanced)", "unbalanced"),
    ("x ((a)", "x ((a)"),
    ("Paris.)", "Paris"),
])
def test_balanced_closing_bracket_kept(raw, ans):
    assert parse_answer(raw, 10)[0] == ans
