"""Gold may be loaded only in gvr.scoring; generation code never receives gold objects."""
import ast
import pathlib
import pytest
from gvr import prompts as P

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gvr"
GENERATION = ["arms.py", "model.py", "prompts.py", "data.py", "retrieval.py", "derive.py", "calllog.py",
              "parsing.py", "uncertainty.py"]


@pytest.mark.parametrize("fname", GENERATION)
def test_no_gold_imports(fname):
    tree = ast.parse((SRC / fname).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            assert "gold" not in mod and "gold" not in names and "analyze" not in mod, (fname, mod, names)
        if isinstance(node, ast.Constant) and node.value in ("aliases", "normalized_aliases"):
            pytest.fail(f"{fname} references gold field {node.value!r}")


def test_builders_reject_non_str():
    gold = {"aliases": ["Paris"], "normalized_aliases": ["paris"]}
    for fn, args in [(P.qa_messages, (gold,)), (P.plan_messages, ("q", gold)),
                     (P.revise_messages, ("q", "a", [("vq", gold)])), (P.rag_messages, (["Paris"], []))]:
        with pytest.raises(TypeError):
            fn(*args)


def test_prompt_texts():
    m = P.qa_messages("Who wrote Hamlet?")
    assert m[0]["content"] == ("You are a factual question-answering assistant. Reply with only the answer: "
                               "a short name, number, date or phrase. Do not explain.")
    assert m[1]["content"] == "Question: Who wrote Hamlet?\nReply with only the shortest standard name, number, date or phrase — no extra words, qualifiers or sentence.\nAnswer:"
    r = P.rag_messages("Q?", [dict(title="T1", text="x1"), dict(title="T2", text="x2")])
    assert r[0]["content"].endswith("Use the passages below if they contain the answer; otherwise answer "
                                    "from your own knowledge.")
    assert r[1]["content"] == "[1] Title: T1\nText: x1\n\n[2] Title: T2\nText: x2\n\nQuestion: Q?\nReply with only the shortest standard name, number, date or phrase — no extra words, qualifiers or sentence.\nAnswer:"
    assert "Proposed answer" not in r[1]["content"]  # A1 not shown in E3
    rv = P.revise_messages("Q?", "A", [("vq1", "va1"), ("vq2", "va2")])[1]["content"]
    assert "Verification question 1: vq1\nVerification answer 1: va1\nVerification question 2: vq2" in rv
    assert "Q1:" not in rv
    reg = P.registry()
    assert all(len(v["sha256"]) == 64 for v in reg.values())


def test_v2_format_line_identical_in_all_answer_turns():
    line = "Reply with only the shortest standard name, number, date or phrase — no extra words, qualifiers or sentence.\nAnswer:"
    turns = [P.qa_messages("q")[1]["content"], P.rag_messages("q", [dict(title="t", text="x")])[1]["content"],
             P.revise_messages("q", "a", [("vq", "va")])[1]["content"]]
    assert all(t.endswith(line) for t in turns)
    assert all(t.count("Reply with only") == 1 for t in turns)   # single, identical format instruction per turn
    assert "Answer:" not in P.plan_messages("q", "a")[1]["content"]          # plan turn unchanged (v1)
    assert P.qa_messages("q")[0]["content"] == P.text("system_qa")           # system prompts unchanged
    assert P.ACTIVE["system_qa"] == "system_qa_v1" and P.ACTIVE["system_rag"] == "system_rag_v1"
