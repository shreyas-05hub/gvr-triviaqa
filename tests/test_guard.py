import json
import pytest
from gvr import guard, data
from gvr.guard import TestSplitLockedError
from gvr.paths import FREEZE_MANIFEST


def test_temp_manifest_locked(locked_manifest):
    assert json.load(open(locked_manifest))["test_unlocked"] is False
    assert guard.test_unlocked() is False                       # guard default now points at the locked temp copy


def test_real_manifest_state_is_consistent():
    """test_unlocked may be true only with a LOCKED test_lock, >= 1 approved chunk, and lock hashes matching files."""
    import hashlib
    from gvr.paths import ROOT
    m = json.load(open(FREEZE_MANIFEST))
    if m["test_unlocked"] is True:
        lock = m.get("test_lock", {})
        assert lock.get("status") == "LOCKED" and m.get("test_chunk_approvals")
        h = lambda f: hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest()
        assert h("configs/config.yaml") == lock["config_sha256"]
        assert h("configs/gate.json") == lock["gate_sha256"] and h("data/test_ids.json") == lock["test_ids_sha256"]
        assert {k: v["sha256"] for k, v in m["frozen_prompts"]["prompts"].items()} == lock["prompt_sha256"]
    else:
        assert m["test_unlocked"] is False


@pytest.mark.parametrize("split", ["validation", "test", "VALIDATION", "test[:10]"])
def test_split_access_raises(split, locked_manifest):
    with pytest.raises(TestSplitLockedError):
        guard.check_split(split)


@pytest.mark.parametrize("f", ["unfiltered.nocontext/validation-00000-of-00001.parquet",
                               "unfiltered.nocontext/test-00000-of-00001.parquet"])
def test_file_access_raises(f, locked_manifest):
    with pytest.raises(TestSplitLockedError):
        guard.check_file(f)


def test_train_allowed():
    guard.check_split("train")
    guard.check_file("unfiltered.nocontext/train-00000-of-00001.parquet")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_data_layer_raises(cfg, split, locked_manifest):
    with pytest.raises(TestSplitLockedError):
        data.load_questions(cfg, split)
    with pytest.raises(TestSplitLockedError):
        data.download_split_file(cfg, split)


def test_sample_test_ids_raises(cfg, locked_manifest):
    with pytest.raises(TestSplitLockedError):
        data.sample_test_ids(cfg, 10)


def test_gold_layer_raises(cfg, locked_manifest):
    from gvr.scoring.gold import load_gold
    with pytest.raises((TestSplitLockedError, PermissionError)):   # split guard (locked) or completeness lock (unlocked)
        load_gold(cfg, "validation", ["x"])


def test_unlock_flag_semantics(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"test_unlocked": "true"}))  # string is NOT true
    with pytest.raises(TestSplitLockedError):
        guard.check_split("test", p)
    p.write_text(json.dumps({"test_unlocked": True}))
    guard.check_split("test", p)  # tmp manifest only; real manifest stays false


def test_answer_columns_blocked_in_data_layer(cfg):
    with pytest.raises(PermissionError):
        data._read(cfg, "train", ["question_id", "answer"])


def test_ids_only_sampling_requires_authorization(cfg, tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"test_unlocked": False}))
    with pytest.raises(PermissionError):
        data.sample_test_ids_ids_only(cfg, 10, p)
    p.write_text(json.dumps({"test_unlocked": False, "test_id_sampling": {"authorized": True, "consumed": True}}))
    with pytest.raises(PermissionError):
        data.sample_test_ids_ids_only(cfg, 10, p)
