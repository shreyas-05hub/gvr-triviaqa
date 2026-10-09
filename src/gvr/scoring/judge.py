"""Model-agnostic judge interface. Never Qwen. Runs in a separate process after Qwen is unloaded."""
import string

from ..paths import PROMPTS

FORBIDDEN_SUBSTR = "qwen"


def check_judge_model(model_id: str):
    if not model_id or FORBIDDEN_SUBSTR in model_id.lower():
        raise ValueError(f"invalid judge model: {model_id!r} (Qwen is never allowed as judge)")


def load_judge_template(version="judge_v1"):
    return (PROMPTS / f"{version}.txt").read_text()


def build_judge_prompt(template: str, question: str, gold_aliases, prediction: str) -> str:
    aliases = "\n".join(f"- {a}" for a in gold_aliases)
    return template.format(question=question, aliases=aliases, prediction=prediction)


def parse_verdict(text: str):
    """First word, case-insensitive. None -> judge_unparsed=True (logged, counted in go/no-go)."""
    words = text.split()
    w = words[0].strip(string.punctuation).upper() if words else ""
    return {"CORRECT": True, "INCORRECT": False}.get(w)


class Judge:
    """backend: callable(prompt:str) -> raw str, greedy. Concrete HF backend lives in the GPU scripts."""

    def __init__(self, model_id, backend, template):
        check_judge_model(model_id)
        self.model_id, self.backend, self.template = model_id, backend, template

    def __call__(self, question, gold_aliases, prediction):
        raw = self.backend(build_judge_prompt(self.template, question, gold_aliases, prediction))
        return raw, parse_verdict(raw)
