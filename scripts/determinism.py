"""Stage 11 determinism: compare two runs of the same items made in SEPARATE processes.
Primary mismatch = parsed answer differs. Secondary = token-id differences and |du| (exact values; flag > 1e-3).
E3: retrieved passage IDs must be identical. NO gold, NO accuracy, NO judge."""
import argparse
import json
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RUN_A", "RUN_B"))
ap.add_argument("--out", default="determinism_summary.json")
a = ap.parse_args()
cfg = load_config()
TOL = cfg["reproducibility"]["u_tolerance"]
WARM = cfg["reproducibility"]["warmup_calls_excluded"]


def load_run(rid):
    d = ROOT / "runs" / rid
    calls = read_jsonl(d / "calls.jsonl")
    meta = json.load(open(d / "run_meta.json")) if (d / "run_meta.json").exists() else {}
    src = meta.get("a1_from")
    e0_src = [c for c in read_jsonl(ROOT / "runs" / src / "calls.jsonl") if c["arm"] == "E0"] if src else []
    return calls, e0_src, meta


def key(c):
    return (c["item_id"], c["arm"], c["step"], c["step_idx"])


def latest_ok(calls):
    out = {}
    for c in calls:
        if c.get("error") is None:
            out[key(c)] = c
    return out


summary = dict(pairs=[], no_gold_no_accuracy=True)
for ra, rb in a.pair:
    ca, sa, ma = load_run(ra)
    cb, sb, mb = load_run(rb)
    assert ma.get("a1_from") == mb.get("a1_from"), "both runs must consume the same logged E0 source"
    ids = sorted({c["item_id"] for c in ca})
    qs = load_questions(cfg, "train", ids)
    ia, ib = derive_items(ca + sa, cfg, qs), derive_items(cb + sb, cfg, qs)
    ga, gb = latest_ok(ca), latest_ok(cb)
    arms = sorted({c["arm"] for c in ca})
    pair = dict(run_a=ra, run_b=rb, a1_from=ma.get("a1_from"), arms={})
    for arm in arms:
        keys = sorted(k for k in ia if k[1] == arm and k in ib)
        prim = [dict(item_id=k[0], a=ia[k]["answer"], b=ib[k]["answer"]) for k in keys
                if ia[k]["answer"] != ib[k]["answer"]]
        ck = sorted(k for k in ga if k[1] == arm and k in gb and ga[k]["step"] != "retrieve")
        tok = [list(k) for k in ck if ga[k]["gen_ids"] != gb[k]["gen_ids"]]
        du = {"|".join(map(str, k)): (None if ga[k]["u"] is None or gb[k]["u"] is None else abs(ga[k]["u"] - gb[k]["u"]))
              for k in ck}
        vals = [v for v in du.values() if v is not None]
        rec = dict(n_items=len(keys), primary_answer_mismatches=len(prim), primary_detail=prim,
                   n_calls_compared=len(ck), secondary_token_id_mismatch_calls=tok,
                   abs_du_per_call=du, max_abs_du=max(vals) if vals else None,
                   n_calls_abs_du_gt_tol=sum(v > TOL for v in vals), u_none_mismatch=sum(
                       (ga[k]["u"] is None) != (gb[k]["u"] is None) for k in ck))
        if arm == "E3":
            ra_ret = {k[0]: c["retrieval"]["passage_ids"] for k, c in ga.items() if k[1] == "E3" and k[2] == "retrieve"}
            rb_ret = {k[0]: c["retrieval"]["passage_ids"] for k, c in gb.items() if k[1] == "E3" and k[2] == "retrieve"}
            rec["passage_ids"] = {i: dict(a=ra_ret.get(i), b=rb_ret.get(i), identical=ra_ret.get(i) == rb_ret.get(i))
                                  for i in sorted(set(ra_ret) | set(rb_ret))}
            rec["passage_ids_all_identical"] = all(v["identical"] for v in rec["passage_ids"].values())
            rec["truncated"] = {i: (ia[(i, "E3")].get("truncated"), ib[(i, "E3")].get("truncated"))
                                for i in sorted({k[0] for k in keys})}
        if arm == "E1":
            rec["verification_questions_identical"] = all(ia[k].get("verif") == ib[k].get("verif") for k in keys)
        pair["arms"][arm] = rec
    for tag, calls, meta in (("a", ca, ma), ("b", cb, mb)):
        gen = sorted([c for c in calls if c.get("error") is None and c["step"] != "retrieve"], key=lambda c: c["t_start"])
        lat = {}
        for c in gen[WARM:]:
            lat.setdefault(f"{c['arm']}/{c['step']}", []).append(c["t_end"] - c["t_start"])
        ret = [c["retrieval"]["latency_s"] for c in calls if c["step"] == "retrieve" and c.get("error") is None]
        pair[f"process_{tag}"] = dict(
            run_id=meta.get("run_id"), peak_gpu_mem_allocated_bytes=meta.get("peak_gpu_mem_allocated_bytes"),
            peak_gpu_mem_reserved_bytes=meta.get("peak_gpu_mem_reserved_bytes"), gpu=meta.get("gpu"),
            cuda_visible_devices=meta.get("cuda_visible_devices"), n_visible_gpus=meta.get("n_visible_gpus"),
            kaggle_dataset=meta.get("kaggle_dataset"), total_wall_s=meta.get("total_wall_s"),
            n_errors=sum(c.get("error") is not None for c in calls),
            n_retries=sum(c.get("retry_of") is not None for c in calls),
            latency_s={k: dict(n=len(v), mean=sum(v) / len(v), min=min(v), max=max(v)) for k, v in lat.items()},
            retrieval_latency_s=dict(n=len(ret), mean=sum(ret) / len(ret), max=max(ret)) if ret else None,
            warmup_calls_excluded=WARM, batch_size=1)
    summary["pairs"].append(pair)
summary["any_primary_mismatch"] = any(r["primary_answer_mismatches"] for p in summary["pairs"] for r in p["arms"].values())
summary["e3_passage_ids_identical"] = all(p["arms"]["E3"]["passage_ids_all_identical"]
                                          for p in summary["pairs"] if "E3" in p["arms"])
write_json(ROOT / "runs" / a.out, summary)
print(json.dumps({p["run_a"] + " vs " + p["run_b"]: {arm: dict(primary=r["primary_answer_mismatches"],
      token_calls=len(r["secondary_token_id_mismatch_calls"]), max_abs_du=r["max_abs_du"])
      for arm, r in p["arms"].items()} for p in summary["pairs"]}, indent=2))
if summary["any_primary_mismatch"]:
    raise SystemExit("PRIMARY MISMATCH: stop and report (no setting changed)")
