"""No reports/ path and no *_LABELED* file can enter the Kaggle package."""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_excludes_reports_and_labels(tmp_path):
    canary = ROOT / "runs" / "_t_pkg_canary"
    canary.mkdir(parents=True, exist_ok=True)
    (canary / "x_LABELED.csv").write_text("secret")
    (canary / "ok.json").write_text("{}")
    out = tmp_path / "pkg.zip"
    try:
        r = subprocess.run(["bash", "scripts/package_repo.sh"], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "OUT": str(out), "INCLUDE_RUNS": "_t_pkg_canary"})
        assert r.returncode == 0, r.stdout + r.stderr
        names = zipfile.ZipFile(out).namelist()
        assert not [n for n in names if n.startswith("reports/") or "_LABELED" in n]
        assert "runs/_t_pkg_canary/ok.json" in names and "configs/config.yaml" in names
    finally:
        for f in canary.iterdir():
            f.unlink()
        canary.rmdir()


def test_format_compliance_never_touches_gold():
    import ast
    src = (ROOT / "scripts" / "format_compliance.py").read_text()
    tree = ast.parse(src)
    mods = {getattr(n, "module", None) or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("scoring" in m or "gold" in m for m in mods), mods
    for banned in ("exact_match", "ground_truths", "load_gold", "f1_score"):
        assert banned not in src, banned
