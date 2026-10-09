import pytest
from gvr.scoring.normalize import normalize_answer as N, normalize_answer, f1_score, contains_token_seq, token_jaccard
from gvr.scoring.metrics import exact_match, max_f1, is_abstain, label_item, answer_string_recall, flips
from gvr.scoring.gold import ground_truths, audit_no_gold
from gvr.scoring.judge import check_judge_model, parse_verdict, Judge

ABS = ["unknown", "i don't know", "i do not know", "not sure", "cannot answer"]


@pytest.mark.parametrize("raw,norm", [
    ("The Beatles", "beatles"), ("A Tale of Two Cities", "tale of two cities"),
    ("an apple", "apple"), ("Theatre", "theatre"), ("Anne", "anne"),
    ("New_York_City", "new york city"), ("U.S.A.", "u s"), ("Rock 'n' Roll", "rock n roll"),
    ("‘Quoted’ ´x` ", "quoted x"), ("  Multiple   spaces ", "multiple spaces"),
    ("Paris, Texas", "paris texas"), ("O’Neill", "o neill"), ("", ""),
])
def test_normalize(raw, norm):
    assert N(raw) == norm


def test_unicode_quotes_beyond_list_kept():
    assert N("“x”") == "“x”"   # official list: only ‘ ’ ´ ` beyond string.punctuation


ANS = {"aliases": ["Paris", "Paris, France", "City of Light"],
       "normalized_aliases": ["paris", "paris france", "city of light"], "value": "Paris"}


def test_ground_truths():
    assert ground_truths(ANS) == {"paris", "paris france", "city of light"}
    a = {"aliases": ["The_Who"], "normalized_aliases": []}
    assert ground_truths(a) == set()          # official: Aliases are NOT used, only NormalizedAliases


def test_ground_truths_official_definition():
    # NormalizedAliases are re-normalized at comparison time (official exact_match_score normalizes both sides)
    a = {"aliases": ["X"], "normalized_aliases": ["Ꞵeta", "paris"]}
    assert ground_truths(a) == {normalize_answer("Ꞵeta"), "paris"}
    a = {"aliases": [], "normalized_aliases": ["paris"], "human_answers": ["The City of Light"]}
    assert ground_truths(a) == {"paris", "city of light"}
    s = {"aliases": ["Scapula (shoulder blade)"], "normalized_aliases": ["scapula shoulder blade"]}
    assert not exact_match("Scapula", ground_truths(s)) and exact_match("Scapula (shoulder blade)", ground_truths(s))


def test_em_cases():
    gts = ground_truths(ANS)
    assert exact_match("paris", gts) and exact_match("PARIS.", gts) and exact_match("the City of Light", gts)
    assert not exact_match("Paris, Texas", gts)          # must be EM=0
    assert not exact_match("It is Paris", gts)           # no substring/contains rule
    assert not exact_match("Pari", gts)


def test_f1():
    assert f1_score("Paris Texas", "Paris") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert f1_score("London", "Paris") == 0.0
    assert max_f1("paris", ground_truths(ANS)) == 1.0


def test_abstain():
    assert is_abstain("I don't know.", ABS) and is_abstain("Unknown", ABS) and is_abstain("not sure", ABS)
    assert not is_abstain("I think Paris", ABS)


def test_token_seq_and_recall():
    assert contains_token_seq("He lived in Paris, France.", "paris france")
    assert not contains_token_seq("Parisian life", "paris")
    assert not contains_token_seq("france paris", "paris france")
    assert answer_string_recall(["nothing here", "visit Paris now"], {"paris"})
    assert not answer_string_recall(["nothing here"], {"paris"})


def test_label_logic():
    gts = ground_truths(ANS)
    l = label_item("Paris", True, gts, ABS)
    assert l["em"] and l["final"] and not l["judged"] and not l["hallucination_proxy"]
    with pytest.raises(ValueError):
        label_item("Lutetia", True, gts, ABS)  # needs judge
    l = label_item("Lutetia", True, gts, ABS, judge_correct=False)
    assert l["incorrect"] and l["hallucination_proxy"]
    l = label_item("Lutetia", True, gts, ABS, judge_correct=True)
    assert l["final"] and not l["em"]
    l = label_item("I don't know", True, gts, ABS, judge_correct=True)  # judge ignored for abstain
    assert l["abstain"] and l["incorrect"] and not l["hallucination_proxy"] and not l["judged"]
    l = label_item("", False, gts, ABS)
    assert l["incorrect"] and not l["hallucination_proxy"]
    l = label_item("Rome", True, gts, ABS, judge_correct=False, passages=["Rome is in Italy"])
    assert l["unsupported"] and not l["answer_string_recall5"]


def test_flips():
    assert flips(True, False) == dict(right_to_wrong=True, wrong_to_right=False)
    assert flips(False, True) == dict(right_to_wrong=False, wrong_to_right=True)


def test_judge_never_qwen():
    for m in ["Qwen/Qwen2.5-7B-Instruct", "qwen/anything", "", None]:
        with pytest.raises(ValueError):
            check_judge_model(m)
    check_judge_model("mistralai/Mistral-7B-Instruct-v0.3")


def test_parse_verdict():
    assert parse_verdict(" CORRECT") is True
    assert parse_verdict("INCORRECT.") is False
    assert parse_verdict("incorrect") is False
    assert parse_verdict("maybe") is None
    assert parse_verdict("Correct, because") is True and parse_verdict("") is None
    assert parse_verdict("The answer is CORRECT") is None   # first word only
    assert parse_verdict("CORRECTLY") is None


def test_judge_interface():
    seen = {}
    def backend(p):
        seen["p"] = p
        return "CORRECT"
    j = Judge("mistralai/Mistral-7B-Instruct-v0.3", backend, "Q:{question}\n{aliases}\nP:{prediction}")
    raw, v = j("q?", ["Paris"], "paris")
    assert v is True and "- Paris" in seen["p"]


def test_audit_no_gold():
    calls = [{"item_id": "a", "step": "answer", "input_text": "Question: capital of France?"}]
    audit_no_gold(calls, {"a": ANS})
    leaked = [{"item_id": "a", "step": "answer", "input_text": "x " + repr(ANS["aliases"])}]
    with pytest.raises(AssertionError):
        audit_no_gold(leaked, {"a": ANS})


def test_jaccard():
    assert token_jaccard("Who wrote Hamlet?", "who wrote hamlet") == 1.0
    assert token_jaccard("a b", "c d") == 0.0


def test_label_item_judge_disabled_em_only():
    gts = ground_truths(ANS)
    l = label_item("Lutetia", True, gts, ABS, judge_enabled=False)          # no verdict needed
    assert not l["judged"] and l["final"] is False and l["final"] == l["em"]
    l = label_item("Lutetia", True, gts, ABS, judge_correct=True, judge_enabled=False)  # verdict ignored
    assert l["final"] is False and l["judge_correct"] is None
    l = label_item("Paris", True, gts, ABS, judge_enabled=False)
    assert l["final"] and l["em"]
