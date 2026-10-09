"""Resumable, idempotent chunk runner for the main test run (pre-registered test_execution_rule).
Unit = (item_id, arm). Order per item: E0 -> E1 (consumes that item's LOGGED E0 answer) -> E3.
A unit is done iff units.jsonl has its marker. On resume, done units are skipped; calls of an incomplete unit
are listed in superseded.jsonl (append-only) and the unit is rerun from scratch."""
import json
import time
from pathlib import Path

from .calllog import read_jsonl
from .derive import derive_items

ARMS = ("E0", "E1", "E3")


def _read(path):
    return read_jsonl(path) if Path(path).exists() else []


def effective_calls(run_dir):
    """calls.jsonl minus superseded call_ids."""
    run_dir = Path(run_dir)
    sup = {r["call_id"] for r in _read(run_dir / "superseded.jsonl")}
    return [c for c in _read(run_dir / "calls.jsonl") if c["call_id"] not in sup]


def done_units(run_dir):
    return {(r["item_id"], r["arm"]) for r in _read(Path(run_dir) / "units.jsonl")}


def _append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def supersede_incomplete(run_dir):
    """Mark calls of units without a done-marker as superseded (append-only). Returns count."""
    run_dir = Path(run_dir)
    done = done_units(run_dir)
    n = 0
    for c in effective_calls(run_dir):
        if (c["item_id"], c["arm"]) not in done:
            _append(run_dir / "superseded.jsonl", dict(call_id=c["call_id"], item_id=c["item_id"], arm=c["arm"],
                                                      reason="incomplete unit at resume", t=time.time()))
            n += 1
    return n


def run_chunk(runner, run_dir, ids, questions, cfg, arms=ARMS):
    """runner: gvr.arms.Runner (or test double) whose logger writes run_dir/calls.jsonl."""
    run_dir = Path(run_dir)
    n_sup = supersede_incomplete(run_dir)
    done = done_units(run_dir)
    ran = skipped = 0
    for iid in ids:
        for arm in arms:
            if (iid, arm) in done:
                skipped += 1
                continue
            q = questions[iid]
            if arm == "E0":
                runner.e0(iid, q)
            elif arm == "E1":
                assert (iid, "E0") in done_units(run_dir), "E1 requires the logged E0 unit"
                calls = [c for c in effective_calls(run_dir) if c["item_id"] == iid and c["arm"] == "E0"]
                e0 = derive_items(calls, cfg, {iid: q})[(iid, "E0")]
                runner.e1(iid, q, e0["answer"], e0["invalid"])
            elif arm == "E3":
                runner.e3(iid, q)
            _append(run_dir / "units.jsonl", dict(item_id=iid, arm=arm, t=time.time()))
            ran += 1
    return dict(units_run=ran, units_skipped=skipped, calls_superseded_at_start=n_sup)


def progress(run_dir, ids, cfg, questions):
    """Between-chunk report: completion, errors, retries, invalid counts. NO gold."""
    calls = effective_calls(run_dir)
    done = done_units(run_dir)
    items = derive_items([c for c in calls if (c["item_id"], c["arm"]) in done], cfg, questions) if done else {}
    out = dict(n_items=len(ids), units_done={a: sum((i, a) in done for i in ids) for a in ARMS},
               errors=sum(c.get("error") is not None for c in calls), retries=sum(c.get("retry_of") is not None for c in calls),
               invalid={a: sum(bool(v["invalid"]) for (i, aa), v in items.items() if aa == a) for a in ARMS})
    out["complete"] = all(out["units_done"][a] == len(ids) for a in ARMS)
    return out
