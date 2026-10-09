"""Official TriviaQA normalize_answer and token F1 (triviaqa utils/dataset_utils.py)."""
import re
import string
from collections import Counter

_EXCLUDE = set(string.punctuation + "".join(["‘", "’", "´", "`"]))


def normalize_answer(s: str) -> str:
    def replace_underscore(text):
        return text.replace("_", " ")

    def lower(text):
        return text.lower()

    def handle_punc(text):
        return "".join(ch if ch not in _EXCLUDE else " " for ch in text)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def contains_token_seq(haystack: str, needle: str) -> bool:
    """Whole-token contiguous match of normalized needle inside normalized haystack."""
    h, n = normalize_answer(haystack).split(), normalize_answer(needle).split()
    if not n:
        return False
    L = len(n)
    return any(h[i:i + L] == n for i in range(len(h) - L + 1))
