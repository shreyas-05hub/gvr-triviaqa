import json
from gvr.config import sha256_file
from gvr.data import read_ids, sample_ids, sample_train_ids
from gvr.paths import DATA, FREEZE_MANIFEST


def test_ids_frozen_and_disjoint(cfg):
    pilot, calib = read_ids("pilot_ids"), read_ids("calib_ids")
    assert len(pilot) == 60 and len(calib) == 500 and not set(pilot) & set(calib)
    man = json.load(open(FREEZE_MANIFEST))
    assert man["pilot_ids_sha256"] == sha256_file(DATA / "pilot_ids.json")
    assert man["calib_ids_sha256"] == sha256_file(DATA / "calib_ids.json")
    assert man["test_unlocked"] in (False, True)   # unlocked state validated in test_real_manifest_state_is_consistent


def test_sampling_reproducible(cfg):
    p, c = sample_train_ids(cfg)
    assert p == read_ids("pilot_ids") and c == read_ids("calib_ids")


def test_sample_ids_exclusion():
    ids = [str(i) for i in range(100)]
    a = sample_ids(ids, 10, 1)
    b = sample_ids(ids, 90, 2, exclude=a)
    assert not set(a) & set(b) and len(set(a) | set(b)) == 100
