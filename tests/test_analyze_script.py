"""Runs scripts/analyze_report.py on SYNTHETIC calls for real pilot ids in a tmp dir (never in repo dirs).
Verifies the report pipeline end-to-end incl. gold loading inside scoring and tau reproducibility."""
import json
import subprocess
import sys
from pathlib import Path
from gvr.calllog import CallLogger
from gvr.data import read_ids

ROOT = Path(__file__).resolve().parents[1]


def test_analyze_report_script(tmp_path):
    ids = read_ids("pilot_ids")
    lg = CallLogger(tmp_path / "runs" / "syn", "syn", "train:pilot_ids", {}, fixture=False)
    def m(i, arm, step, idx):
        return dict(item_id=i, arm=arm, step=step, step_idx=idx, model_id="SYNTHETIC", model_revision="x",
                    precision="x", prompt_id="p", prompt_sha={})
    def g(raw, u, t):
        return lambda: dict(input_text="Question: q\nAnswer:", raw_output=raw, input_tokens=5, output_tokens=2,
                            token_logprobs=[-u], t_start=t, t_end=t + 0.5, u=u, gen_ids=[1])
    for n, i in enumerate(ids):
        lg.run(g("I don't know", 0.01 * n, n), m(i, "E0", "answer", 0))   # abstain -> no judge needed
        lg.run(g("\n", 0.1, n), m(i, "E1", "plan", 0))                    # sv_plan_invalid -> final = A1
        lg.run(lambda: dict(retrieval=dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=[], scores=[], latency_s=0.01),
                            passages=[], t_start=0, t_end=0.01), dict(item_id=i, arm="E3", step="retrieve", step_idx=0,
                                                                      input_text="q"))
        lg.run(lambda n=n: dict(g("not sure", 0.2, n)(), truncated=False), m(i, "E3", "answer", 1))
    out = tmp_path / "out"
    real_gate = ROOT / "configs" / "gate.json"
    before = real_gate.read_bytes() if real_gate.exists() else None
    r = subprocess.run([sys.executable, "scripts/analyze_report.py", "--run-id", "syn", "--runs-dir",
                        str(tmp_path / "runs"), "--out-dir", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    md = (out / "reports" / "pilot_report.md").read_text()
    assert "Engineering pilot; descriptive only; no scientific conclusions." in md
    gate = json.load(open(out / "configs" / "gate.json"))
    assert gate["provisional"] is True and gate["comparison"] == "strict_gt"
    res = json.load(open(tmp_path / "runs" / "syn" / "results.json"))
    assert res["checks"]["tau_reproducible"] is True and res["checks"]["unrecovered"] == 0
    assert res["results"]["summary"]["E0"]["abstain"] == 1.0
    assert all(v["judged"] == 0 for v in res["results"]["summary"].values())   # main-study default: judge off
    after = real_gate.read_bytes() if real_gate.exists() else None
    assert before == after   # synthetic run must never touch the repo's gate.json
