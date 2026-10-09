"""scripts/determinism.py on synthetic separate-process runs (incl. --a1-from E1 and E3 passage IDs). No gold."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from gvr.calllog import CallLogger
from gvr.data import read_ids

ROOT = Path(__file__).resolve().parents[1]
P = "_t_det_"


def _mk(run, arms, ids, a1_from=None, perturb=False):
    d = ROOT / "runs" / (P + run)
    lg = CallLogger(d, P + run, "train:pilot_ids", {}, fixture=False)
    def m(i, arm, step, idx):
        return dict(item_id=i, arm=arm, step=step, step_idx=idx, model_id="SYNTHETIC", model_revision="x",
                    precision="x", prompt_id="p", prompt_sha={})
    def g(raw, u, ids_=(1, 151645)):
        return lambda: dict(input_text="q", raw_output=raw, input_tokens=5, output_tokens=2,
                            token_logprobs=[-u], t_start=0.0, t_end=0.1, u=u, gen_ids=list(ids_))
    for n, i in enumerate(ids):
        if "E0" in arms:
            lg.run(g(f"ans{n}", 0.1 + n / 100 + (1e-4 if perturb else 0)), m(i, "E0", "answer", 0))
        if "E1" in arms:
            lg.run(g("Q a?\nQ b?\nQ c?", 0.2), m(i, "E1", "plan", 0))
            for k in (1, 2, 3):
                lg.run(g("x", 0.3), m(i, "E1", "verify", k))
            lg.run(g(f"rev{n}", 0.4, (2, 151645) if (perturb and n == 0) else (1, 151645)), m(i, "E1", "revise", 4))
        if "E3" in arms:
            ps = [dict(id=f"d{n}-{j}", score=1.0, title="t", text="x") for j in range(5)]
            lg.run(lambda ps=ps: dict(retrieval=dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=[p["id"] for p in ps],
                                                     scores=[1.0] * 5, latency_s=0.01), passages=ps, t_start=0, t_end=0.01),
                   dict(item_id=i, arm="E3", step="retrieve", step_idx=0, input_text="q"))
            lg.run(lambda n=n: dict(g(f"rag{n}", 0.5)(), truncated=False), m(i, "E3", "answer", 1))
    json.dump(dict(run_id=P + run, a1_from=(P + a1_from) if a1_from else None, peak_gpu_mem_allocated_bytes=1,
                   total_wall_s=1.0), open(d / "run_meta.json", "w"))


def test_determinism_script():
    ids = read_ids("pilot_ids")
    try:
        _mk("E0-1", ["E0"], ids[:20]); _mk("E0-2", ["E0"], ids[:20], perturb=True)
        _mk("E13-1", ["E1", "E3"], ids[:10], a1_from="E0-1"); _mk("E13-2", ["E1", "E3"], ids[:10], a1_from="E0-1", perturb=True)
        r = subprocess.run([sys.executable, "scripts/determinism.py", "--pair", P + "E0-1", P + "E0-2",
                            "--pair", P + "E13-1", P + "E13-2", "--out", P + "summary.json"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        s = json.load(open(ROOT / "runs" / (P + "summary.json")))
        e0 = s["pairs"][0]["arms"]["E0"]
        assert e0["primary_answer_mismatches"] == 0 and abs(e0["max_abs_du"] - 1e-4) < 1e-9
        assert e0["n_calls_abs_du_gt_tol"] == 0 and e0["n_items"] == 20
        e1 = s["pairs"][1]["arms"]["E1"]
        assert e1["n_items"] == 10 and len(e1["secondary_token_id_mismatch_calls"]) == 1
        assert s["pairs"][1]["a1_from"] == P + "E0-1"
        assert s["e3_passage_ids_identical"] and not s["any_primary_mismatch"] and s["no_gold_no_accuracy"]
    finally:
        for p in (ROOT / "runs").glob(P + "*"):
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def test_determinism_never_touches_gold():
    import ast
    tree = ast.parse((ROOT / "scripts/determinism.py").read_text())
    mods = {getattr(n, "module", None) or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("scoring" in m or "gold" in m for m in mods), mods
