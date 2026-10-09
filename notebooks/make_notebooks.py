"""Generates the Kaggle notebooks from notebooks/src/*.py (keeps notebook code reviewable as plain files)."""
import json
from pathlib import Path

H = Path(__file__).parent


def nb(cells):
    out = []
    for kind, src in cells:
        c = dict(cell_type=kind, metadata={}, source=src.splitlines(keepends=True))
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        out.append(c)
    return dict(cells=out, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
                nbformat=4, nbformat_minor=5)


audit = nb([
    ("markdown", "# Stage-0 audit (Kaggle / Colab)\nMeasurements only: no installs, no model or index downloads. "
                 "Enable GPU (T4) and Internet in notebook settings, Run All, paste the full output back."),
    ("code", (H / "src/audit_system.py").read_text()),
    ("code", (H / "src/audit_pyserini_meta.py").read_text()),
])
json.dump(audit, open(H / "kaggle_stage0_audit.ipynb", "w"), indent=1)

runner = nb([
    ("markdown", "# GPU stage runner\nAttach the repo as a Kaggle dataset (zip from `scripts/package_repo.sh`), "
                 "and previous outputs (runs/, configs/) as datasets if continuing. Set STAGE then Run All.\n\n"
                 "Large downloads happen only in stages that the user has explicitly confirmed."),
    ("code", (H / "src/runner.py").read_text()),
])
json.dump(runner, open(H / "kaggle_runner.ipynb", "w"), indent=1)
print("ok")
