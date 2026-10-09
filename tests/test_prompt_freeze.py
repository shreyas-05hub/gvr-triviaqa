import json
import pytest
from gvr import prompts as P
from gvr.paths import FREEZE_MANIFEST


def test_active_prompts_match_frozen():
    assert P.assert_frozen() is True


def test_guard_aborts_on_hash_mismatch(tmp_path):
    m = json.load(open(FREEZE_MANIFEST))
    m["frozen_prompts"]["prompts"]["user_rag"]["sha256"] = "0" * 64
    p = tmp_path / "m.json"; p.write_text(json.dumps(m))
    with pytest.raises(P.PromptNotFrozenError):
        P.assert_frozen(p)
    m2 = json.load(open(FREEZE_MANIFEST))
    m2["frozen_prompts"]["prompts"]["user_qa"]["file"] = "prompts/user_qa_v1.txt"
    p.write_text(json.dumps(m2))
    with pytest.raises(P.PromptNotFrozenError):
        P.assert_frozen(p)


def test_frozen_set_is_exactly_generation_prompts():
    fp = json.load(open(FREEZE_MANIFEST))["frozen_prompts"]
    assert set(fp["prompts"]) == set(P.GENERATION_PROMPTS) and "judge" not in fp["prompts"]
    assert fp["prompts"]["user_qa"]["file"].endswith("user_qa_v2.txt") and fp["prompts"]["system_qa"]["file"].endswith("system_qa_v1.txt")


def test_calibration_gate_script_never_touches_gold():
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fit_calibration_gate.py").read_text()
    tree = ast.parse(src)
    mods = {getattr(n, "module", None) or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m.endswith(("gold", "metrics", "analyze")) for m in mods), mods
    for banned in ("exact_match", "ground_truths", "load_gold", "f1"):
        assert banned not in src, banned
    assert "provisional=False" in src and '"higher"' not in src   # method comes from config


def test_calibration_blocked_after_calib01():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1]
    m = re.search(r'case "\$STAGE" in ([a-z_|]+)\)', (root / "scripts/kaggle_push.sh").read_text())
    assert m and {"calibration", "test"} <= set(m.group(1).split("|"))
    push = (root / "scripts/kaggle_push.sh").read_text(); runner = (root / "notebooks/src/runner.py").read_text()
    assert "test_chunk_approvals" in push and "test_chunk_approvals" in runner      # test_chunk gated per approved chunk
    assert 'if STAGE in ("calibration", "test"):' in runner
    src = (root / "scripts/run_arms.py").read_text()
    assert "calib_rerun_approved" in src and "calibration_runs.json" in src
