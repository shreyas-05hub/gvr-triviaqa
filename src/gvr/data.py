"""Question/ID access. Never reads answer columns: gold is loaded only in gvr.scoring."""
import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .config import load_config, sha256_file
from .guard import check_file, check_split
from .paths import DATA, FREEZE_MANIFEST

QUESTION_COLUMNS = ["question_id", "question"]


def split_file(cfg, split):
    check_split(split)
    if split == "train":
        return cfg["dataset"]["train_file"]
    for f in cfg["dataset"]["locked_files"]:
        if f.rsplit("/", 1)[-1].startswith(split + "-"):
            return f
    raise ValueError(split)


def download_split_file(cfg, split):
    fname = split_file(cfg, split)
    check_file(fname)
    return hf_hub_download(cfg["dataset"]["repo_id"], fname, repo_type="dataset",
                           revision=cfg["dataset"]["revision"])


def _test_questions_allowed(manifest_path=FREEZE_MANIFEST):
    """Validation question text may be read only inside a Kaggle runner session (GVR_SESSION=kaggle-runner) or after
    the test completeness check passed. Keeps the local cached validation parquet unread until completeness."""
    import os
    if os.environ.get("GVR_SESSION") == "kaggle-runner":
        return True
    return json.load(open(manifest_path)).get("test_completeness", {}).get("passed") is True


def _read(cfg, split, columns):
    if "answer" in columns:
        raise PermissionError("answer columns may only be read inside gvr.scoring")
    if split != "train":
        check_split(split)
        if not _test_questions_allowed():
            raise PermissionError("validation questions readable only in a Kaggle runner session or after completeness passed")
    return pq.read_table(download_split_file(cfg, split), columns=columns).to_pylist()


def load_questions(cfg, split, ids=None):
    """{question_id: question} only. No gold."""
    rows = _read(cfg, split, QUESTION_COLUMNS)
    q = {r["question_id"]: r["question"] for r in rows}
    if ids is None:
        return q
    return {i: q[i] for i in ids}


def sample_ids(all_ids, n, seed, exclude=()):
    pool = sorted(set(all_ids) - set(exclude))
    rng = np.random.Generator(np.random.PCG64(seed))
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def sample_train_ids(cfg):
    s = cfg["sampling"]
    all_ids = list(load_questions(cfg, "train").keys())
    assert len(all_ids) == len(set(all_ids)), "duplicate question_id in train"
    pilot = sample_ids(all_ids, s["n_pilot"], s["seed_a_pilot"])
    calib = sample_ids(all_ids, s["n_calib"], s["seed_b_calib"], exclude=pilot)
    assert set(pilot).isdisjoint(calib)
    return pilot, calib


def sample_test_ids(cfg, n):
    """Implemented, NOT run in the pilot. Samples final test IDs from validation by seed C."""
    split = cfg["dataset"]["test_source_split"]
    check_split(split)  # raises while test_unlocked is false
    ids = list(load_questions(cfg, split).keys())
    return sample_ids(ids, n, cfg["sampling"]["seed_c_test"])


def write_ids(pilot, calib, manifest_path=FREEZE_MANIFEST):
    DATA.mkdir(exist_ok=True)
    paths = {"pilot_ids": DATA / "pilot_ids.json", "calib_ids": DATA / "calib_ids.json"}
    for key, ids in (("pilot_ids", pilot), ("calib_ids", calib)):
        with open(paths[key], "w") as f:
            json.dump(ids, f, indent=1)
    with open(manifest_path) as f:
        man = json.load(f)
    for key, p in paths.items():
        man[f"{key}_sha256"] = sha256_file(p)
    with open(manifest_path, "w") as f:
        json.dump(man, f, indent=2)
    return paths


def read_ids(name):
    with open(DATA / f"{name}.json") as f:
        return json.load(f)


def sample_test_ids_ids_only(cfg, n, manifest_path=FREEZE_MANIFEST):
    """ONE-SHOT, PI-authorized test-ID draw while test_unlocked stays false: reads ONLY the question_id column of the
    test source split, samples n with seed C, returns them IN DRAW ORDER (random, fixed). Marks the authorization
    consumed. Never reads questions or answers."""
    man = json.load(open(manifest_path))
    auth = man.get("test_id_sampling", {})
    if auth.get("authorized") is not True or auth.get("consumed"):
        raise PermissionError("test-ID sampling not authorized (or already consumed)")
    split = cfg["dataset"]["test_source_split"]
    fname = next(f for f in cfg["dataset"]["locked_files"] if f.rsplit("/", 1)[-1].startswith(split + "-"))
    path = hf_hub_download(cfg["dataset"]["repo_id"], fname, repo_type="dataset", revision=cfg["dataset"]["revision"])
    all_ids = pq.read_table(path, columns=["question_id"]).column("question_id").to_pylist()
    pool = sorted(set(all_ids))
    assert len(pool) == len(all_ids), "duplicate question_id in test source split"
    rng = np.random.Generator(np.random.PCG64(cfg["sampling"]["seed_c_test"]))
    drawn = rng.choice(pool, size=n, replace=False).tolist()
    auth.update(consumed=True, n_pool=len(pool), n_drawn=n, columns_read=["question_id"], file=fname)
    man["test_id_sampling"] = auth
    with open(manifest_path, "w") as f:
        json.dump(man, f, indent=2)
    return drawn
