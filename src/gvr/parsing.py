import re
import string

PARSER_VERSION = "1.1"   # 1.1: keep a closing ) or ] that balances an opening bracket
_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)
_TRAIL = string.punctuation + string.whitespace
_ENUM = re.compile(r"^\s*(?:\d+\s*[.):]|[-*•])\s*")


def parse_answer(text: str, max_words: int = 10):
    """-> (answer, invalid). First non-empty line, strip 'Answer:' prefix and trailing punct/ws."""
    line = next((l for l in text.splitlines() if l.strip()), "")
    ans = _strip_trailing(_PREFIX.sub("", line).strip())
    invalid = ans == "" or len(ans.split()) > max_words
    return ans, invalid


def _strip_trailing(s):
    """Strip trailing punctuation/whitespace, except a closing ')' or ']' that balances an opening bracket."""
    while s and s[-1] in _TRAIL:
        c = s[-1]
        if (c == ")" and s.count("(") >= s.count(")")) or (c == "]" and s.count("[") >= s.count("]")):
            break
        s = s[:-1]
    return s


def parse_plan(text: str, n: int = 3):
    """Up to n non-empty lines; leading list markers ('1.', '2)', '-', '*') removed."""
    qs = []
    for l in text.splitlines():
        q = _ENUM.sub("", l).strip()
        if q:
            qs.append(q)
        if len(qs) == n:
            break
    return qs
