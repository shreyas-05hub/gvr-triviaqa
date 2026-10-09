"""Prompt templates (versioned files). Builders accept plain strings only, never gold objects."""
import json
from .config import sha256_file
from .paths import PROMPTS

ACTIVE = {  # prompt_id -> file stem; bump version here when a file changes
    "system_qa": "system_qa_v1", "user_qa": "user_qa_v2",
    "system_rag": "system_rag_v1", "user_rag": "user_rag_v2", "passage": "passage_v1",
    "system_cove_plan": "system_cove_plan_v1", "user_cove_plan": "user_cove_plan_v1",
    "user_cove_revise": "user_cove_revise_v2", "judge": "judge_v1",
    "judge_v2": "judge_v2",   # FROZEN for judge-dev validation (freeze_manifest.judge_v2_prompt)
}


def path(pid):
    return PROMPTS / f"{ACTIVE[pid]}.txt"


def text(pid):
    return path(pid).read_text()


def sha(pid):
    return sha256_file(path(pid))


def registry():
    return {pid: {"file": f"prompts/{stem}.txt", "sha256": sha(pid)} for pid, stem in ACTIVE.items()}


def write_registry():
    with open(PROMPTS / "registry.json", "w") as f:
        json.dump(registry(), f, indent=2)


def _str(*xs):
    for x in xs:
        if not isinstance(x, str):
            raise TypeError(f"prompt builders accept str only, got {type(x).__name__} (gold guard)")


def qa_messages(question):
    _str(question)
    return [{"role": "system", "content": text("system_qa")},
            {"role": "user", "content": text("user_qa").format(question=question)}]


def format_passages(passages):
    for p in passages:
        _str(p["title"], p["text"])
    t = text("passage")
    return "\n\n".join(t.format(i=i, title=p["title"], text=p["text"]) for i, p in enumerate(passages, 1))


def rag_messages(question, passages):
    _str(question)
    return [{"role": "system", "content": text("system_rag")},
            {"role": "user", "content": text("user_rag").format(passages=format_passages(passages),
                                                                question=question)}]


def plan_messages(question, a1):
    _str(question, a1)
    return [{"role": "system", "content": text("system_cove_plan")},
            {"role": "user", "content": text("user_cove_plan").format(question=question, answer=a1)}]


def revise_messages(question, a1, qa_pairs):
    _str(question, a1, *[x for p in qa_pairs for x in p])
    pairs = "\n".join(f"Verification question {i}: {q}\nVerification answer {i}: {a}" for i, (q, a) in enumerate(qa_pairs, 1))
    return [{"role": "system", "content": text("system_qa")},
            {"role": "user", "content": text("user_cove_revise").format(question=question, answer=a1,
                                                                        qa_pairs=pairs)}]


GENERATION_PROMPTS = ("system_qa", "user_qa", "system_rag", "user_rag", "passage",
                      "system_cove_plan", "user_cove_plan", "user_cove_revise")


class PromptNotFrozenError(RuntimeError):
    pass


def assert_frozen(manifest_path=None):
    """Abort if any active generation prompt (file or hash) differs from freeze_manifest.frozen_prompts."""
    from .paths import FREEZE_MANIFEST
    man = json.load(open(manifest_path or FREEZE_MANIFEST))
    fp = man.get("frozen_prompts")
    if not fp or fp.get("status") != "FROZEN":
        raise PromptNotFrozenError("no FROZEN prompt set in freeze_manifest")
    for pid in GENERATION_PROMPTS:
        rec = fp["prompts"].get(pid)
        if rec is None:
            raise PromptNotFrozenError(f"{pid} not in frozen set")
        if rec["file"] != f"prompts/{ACTIVE[pid]}.txt" or rec["sha256"] != sha(pid):
            raise PromptNotFrozenError(f"{pid}: active {ACTIVE[pid]} sha {sha(pid)[:12]} != frozen {rec['file']} {rec['sha256'][:12]}")
    return True
