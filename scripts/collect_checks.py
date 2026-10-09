"""Builds runs/checks.json for the report from GPU unit-test outcomes and the BM25 index record."""
import json
from _common import ROOT, write_json

chk = {}
p = ROOT / "runs" / "unit_main" / "pytest_main.json"
if p.exists():
    o = json.load(open(p))["outcomes"]
    def passed(name):
        hits = [v for k, v in o.items() if k.endswith(name)]
        return (all(v == "passed" for v in hits)) if hits else None
    chk["rescore_pass_superseded"] = passed("test_teacher_forced_rescore")   # informational only (case 1)
    al = [passed(n) for n in ("test_logits_length", "test_greedy_argmax", "test_eos_excluded_and_generation_overrides")]
    chk["alignment_pass"] = all(al) if None not in al else None
    chk["no_oom_2048"] = passed("test_runner_e0_e1_e3_logging")
    chk["unit_main_all_passed"] = all(v == "passed" for v in o.values()) if o else None
    chk["unit_main_outcomes"] = o
mp = ROOT / "runs" / "unit_main" / "measurements.json"
if mp.exists():
    m = json.load(open(mp))
    chk["measurements"] = m
    rs = m.get("rescore") or []
    chk["rescore_max_abs_delta"] = max((r["abs_delta"] for r in rs if r["abs_delta"] is not None), default=None)
    with open(ROOT / "runs" / "setup_timing.jsonl", "a") as f:
        f.write(json.dumps(dict(step="qwen_nf4_model_load", seconds=m.get("model_load_s"))) + "\n")
dp = ROOT / "runs" / "diag" / "rescore_diag.json"
if dp.exists():
    _items = json.load(open(dp))["items"]
    chk["same_path_repro_pass"] = len(_items) == 5 and all(
        r["abs_delta_u"]["gen1_vs_gen2"] < 1e-6 and r["tokens_equal"]["gen1_gen2"] for r in _items)
    chk["rescore_diag"] = {r["item_id"]: dict(r["abs_delta_u"], tokens_equal=r["tokens_equal"],
                                              **r["max_abs_delta_logprob"]) for r in json.load(open(dp))["items"]}
ir = ROOT / "configs" / "index_record.json"
if ir.exists():
    chk["bm25_ok"] = json.load(open(ir)).get("bm25_ok")
    chk["index_record"] = json.load(open(ir))
write_json(ROOT / "runs" / "checks.json", chk)
print(json.dumps(chk, indent=2))
