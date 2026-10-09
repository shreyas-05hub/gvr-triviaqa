"""Resume / idempotency of the chunk runner with a fake (CPU) runner that can crash mid-unit."""
import pytest
from gvr.calllog import CallLogger, read_jsonl
from gvr.test_runner import run_chunk, effective_calls, done_units, progress


class Boom(RuntimeError):
    pass


class FakeRunner:
    """Mimics gvr.arms.Runner call logging; crash_at=(item, arm, step) raises mid-unit after earlier steps logged."""
    def __init__(self, lg, crash_at=None):
        self.lg, self.crash_at, self.seen_a1 = lg, crash_at, {}

    def _log(self, iid, arm, step, idx, raw):
        if self.crash_at == (iid, arm, step):
            raise Boom(f"crash at {iid} {arm} {step}")
        self.lg.log(item_id=iid, arm=arm, step=step, step_idx=idx, model_id="fake", model_revision="x", precision="x",
                    prompt_id="p", prompt_sha={}, input_text="q", raw_output=raw, input_tokens=1, output_tokens=1,
                    token_logprobs=[-0.1], t_start=0.0, t_end=0.1, u=0.1, gen_ids=[1, 151645], parser_version="1.1")

    def e0(self, iid, q):
        self._log(iid, "E0", "answer", 0, f"ans-{iid}")

    def e1(self, iid, q, a1, a1_invalid):
        self.seen_a1[iid] = a1
        self._log(iid, "E1", "plan", 0, "Q a?\nQ b?\nQ c?")
        for k in (1, 2, 3):
            self._log(iid, "E1", "verify", k, "x")
        self._log(iid, "E1", "revise", 4, f"rev-{iid}")

    def e3(self, iid, q):
        self.lg.log(item_id=iid, arm="E3", step="retrieve", step_idx=0, input_text=q,
                    retrieval=dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=[], scores=[], latency_s=0.0), passages=[])
        self._log(iid, "E3", "answer", 1, f"rag-{iid}")


def test_resume_is_idempotent(tmp_path, cfg):
    ids = [f"t{i}" for i in range(5)]
    qs = {i: f"question {i}?" for i in ids}
    lg = CallLogger(tmp_path, "test-cX", "validation:test", {}, fixture=True)
    with pytest.raises(Boom):                                    # session 1 dies mid-unit (t2, E1 verify)
        run_chunk(FakeRunner(lg, crash_at=("t2", "E1", "verify")), tmp_path, ids, qs, cfg)
    assert ("t2", "E0") in done_units(tmp_path) and ("t2", "E1") not in done_units(tmp_path)
    fr = FakeRunner(lg)
    r2 = run_chunk(fr, tmp_path, ids, qs, cfg)                   # session 2 resumes
    assert r2["calls_superseded_at_start"] == 1                  # the orphan t2/E1 plan call
    assert r2["units_skipped"] == 7 and r2["units_run"] == 8      # t0,t1 x3 + t2/E0 skipped
    assert fr.seen_a1["t2"] == "ans-t2"                           # E1 consumed the LOGGED E0 answer
    r3 = run_chunk(FakeRunner(lg), tmp_path, ids, qs, cfg)        # session 3: nothing left to do
    assert r3 == dict(units_run=0, units_skipped=15, calls_superseded_at_start=0)
    eff = effective_calls(tmp_path)
    for i in ids:                                                  # exactly one complete unit per (item, arm)
        assert sum(c["item_id"] == i and c["arm"] == "E1" and c["step"] == "plan" for c in eff) == 1
        assert sum(c["item_id"] == i and c["arm"] == "E0" for c in eff) == 1
    assert len(read_jsonl(tmp_path / "calls.jsonl")) == len(eff) + 1   # append-only: superseded call kept on disk
    p = progress(tmp_path, ids, cfg, qs)
    assert p["complete"] and p["units_done"] == {"E0": 5, "E1": 5, "E3": 5} and set(p) >= {"errors", "invalid"}


def test_unit_order_and_no_duplicate_work(tmp_path, cfg):
    ids = ["a", "b"]
    lg = CallLogger(tmp_path, "test-cY", "validation:test", {}, fixture=True)
    run_chunk(FakeRunner(lg), tmp_path, ids, {i: i for i in ids}, cfg)
    order = [(r["item_id"], r["arm"]) for r in read_jsonl(tmp_path / "units.jsonl")]
    assert order == [("a", "E0"), ("a", "E1"), ("a", "E3"), ("b", "E0"), ("b", "E1"), ("b", "E3")]
    n_before = len(read_jsonl(tmp_path / "calls.jsonl"))
    run_chunk(FakeRunner(lg), tmp_path, ids, {i: i for i in ids}, cfg)
    assert len(read_jsonl(tmp_path / "calls.jsonl")) == n_before


def test_test_gold_locked_until_completeness(cfg, tmp_path, completeness_unpassed_manifest):
    import json
    from gvr.scoring.gold import load_gold
    from gvr.guard import TestSplitLockedError
    with pytest.raises((TestSplitLockedError, PermissionError)):  # locked: split guard; unlocked: completeness lock
        load_gold(cfg, "validation", ["x"])
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"test_unlocked": True}))
    import gvr.guard as G
    orig = G.test_unlocked
    G.test_unlocked = lambda manifest_path=None: True
    try:
        with pytest.raises(PermissionError):                      # unlocked but completeness not passed
            load_gold(cfg, "validation", ["x"], manifest_path=p)
    finally:
        G.test_unlocked = orig


def test_chunk_script_aborts_before_downloads():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_test_chunk.py").read_text()
    head = src.split("import torch")[0]
    for chk in ("test_unlocked()", "assert_frozen()", "gate.json hash", "config.yaml hash", "test_ids hash", "model revision"):
        assert chk in head, chk
    assert "load_gold" not in src and "exact_match" not in src


def test_validation_questions_blocked_outside_kaggle_session(cfg, monkeypatch, completeness_unpassed_manifest):
    import gvr.data as D
    monkeypatch.setattr(D, "check_split", lambda s: None)             # simulate test_unlocked: true
    monkeypatch.delenv("GVR_SESSION", raising=False)
    with pytest.raises(PermissionError):
        D._read(cfg, "validation", ["question_id", "question"])         # local: refused before any file is opened
    monkeypatch.setattr(D, "download_split_file", lambda c, s: (_ for _ in ()).throw(RuntimeError("would open file")))
    monkeypatch.setenv("GVR_SESSION", "kaggle-runner")
    with pytest.raises(RuntimeError, match="would open file"):          # Kaggle runner session: allowed through
        D._read(cfg, "validation", ["question_id", "question"])


def test_real_completeness_state_is_consistent():
    """Real state: test_completeness may be passed only with 9,000/9,000 units (3,000 per chunk, 0 extra)."""
    import json
    from gvr.paths import FREEZE_MANIFEST
    tc = json.load(open(FREEZE_MANIFEST)).get("test_completeness")
    if tc and tc.get("passed") is True:
        pc = tc["per_chunk"]
        assert set(pc) == {"c1", "c2", "c3"}
        assert all(v["units_done"] == v["units_expected"] == 3000 and v["extra_units"] == 0 for v in pc.values())
        assert sum(v["units_done"] for v in pc.values()) == 9000
