"""Completeness check (test_execution_rule): 3,000 items x 3 arms done across c1..c3. NO gold. Records result."""
import datetime
import json
from _common import ROOT
from gvr.config import load_config
from gvr.data import read_ids
from gvr.test_runner import done_units, effective_calls

cfg = load_config()
man = json.load(open(ROOT / "configs/freeze_manifest.json"))
ids = read_ids("test_ids")
res = {}
for c, (lo, hi) in man["test_ids"]["chunks"].items():
    d = done_units(ROOT / "runs" / f"test-{c}")
    want = {(i, a) for i in ids[lo:hi] for a in ("E0", "E1", "E3")}
    res[c] = dict(units_expected=len(want), units_done=len(d & want), extra_units=len(d - want),
                  n_effective_calls=len(effective_calls(ROOT / "runs" / f"test-{c}")))
passed = all(r["units_done"] == r["units_expected"] == 3000 and r["extra_units"] == 0 for r in res.values())
man["test_completeness"] = dict(passed=passed, per_chunk=res, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(man, open(ROOT / "configs/freeze_manifest.json", "w"), indent=2)
print(json.dumps(man["test_completeness"], indent=1))
