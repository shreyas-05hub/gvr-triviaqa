import pytest
from gvr.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def pytest_sessionfinish(session, exitstatus):
    """With GVR_MODEL=main, record per-test outcomes for the go/no-go table (runs/unit_main/)."""
    import json, os, pathlib
    if os.environ.get("GVR_MODEL") != "main":
        return
    out = pathlib.Path(__file__).resolve().parents[1] / "runs" / "unit_main"
    out.mkdir(parents=True, exist_ok=True)
    res = _OUTCOMES
    json.dump(dict(outcomes=res, exitstatus=int(exitstatus)), open(out / "pytest_main.json", "w"), indent=2)


_OUTCOMES = {}


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _OUTCOMES[report.nodeid] = report.outcome


@pytest.fixture
def locked_manifest(tmp_path, monkeypatch):
    """Temporary copy of the real freeze manifest with test_unlocked: false, wired in as the guard's default path."""
    import json
    from gvr import guard
    from gvr.paths import FREEZE_MANIFEST
    m = json.load(open(FREEZE_MANIFEST))
    m["test_unlocked"] = False
    p = tmp_path / "freeze_manifest_locked.json"
    p.write_text(json.dumps(m))
    for fn in (guard.check_split, guard.check_file, guard.test_unlocked):
        monkeypatch.setattr(fn, "__defaults__", (p,))
    return p


@pytest.fixture
def completeness_unpassed_manifest(tmp_path, monkeypatch):
    """Temporary copy of the real freeze manifest with test_completeness.passed = false, wired into the gold and
    validation-question locks (they otherwise read the real manifest, where completeness has passed)."""
    import json
    import gvr.paths as P
    import gvr.data as D
    from gvr.paths import FREEZE_MANIFEST
    m = json.load(open(FREEZE_MANIFEST))
    m.setdefault("test_completeness", {})["passed"] = False
    p = tmp_path / "freeze_manifest_completeness_unpassed.json"
    p.write_text(json.dumps(m))
    monkeypatch.setattr(P, "FREEZE_MANIFEST", p)                       # read at call time by scoring.gold.load_gold
    monkeypatch.setattr(D._test_questions_allowed, "__defaults__", (p,))  # bound default in gvr.data
    return p
