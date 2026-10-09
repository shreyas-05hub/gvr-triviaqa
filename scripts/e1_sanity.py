"""E1 engineering sanity check (dev logs only; NO gold, NO EM, NO accuracy)."""
import json
import re
import numpy as np
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.parsing import parse_answer
from gvr.scoring.normalize import normalize_answer

cfg = load_config()
YES = re.compile(r"^\s*(yes|true|correct)\b", re.I)
NO = re.compile(r"^\s*(no|false|incorrect|not)\b", re.I)
out = {}
for run in ("pilot-01", "judgedev-01", "pilot-02-v2"):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
    ok = [c for c in calls if c.get("error") is None]
    n = diff = verbatim = plan_inv = 0
    wiring_bad = []
    vans = []
    for iid in ids:
        e0, e1 = items[(iid, "E0")], items[(iid, "E1")]
        n += 1
        diff += normalize_answer(e1["answer"]) != normalize_answer(e0["answer"])
        if e1.get("sv_plan_invalid"):
            plan_inv += 1
            continue
        rev = next(c for c in reversed(ok) if c["item_id"] == iid and c["arm"] == "E1" and c["step"] == "revise")
        verbatim += parse_answer(rev["raw_output"], cfg["parser"]["max_words"])[0] == e0["answer"]
        # wiring: revise input contains A1 (logged E0 answer) and every verification Q/A pair, in order
        t = rev["input_text"]
        need = [f"Proposed answer: {e0['answer']}"] + [s for i, v in enumerate(e1["verif"], 1) for s in (v["q"], v["a"])]
        if not all(s in t for s in need):
            wiring_bad.append(iid)
        # verify contexts must NOT contain A1 or the original question
        for v in (c for c in ok if c["item_id"] == iid and c["arm"] == "E1" and c["step"] == "verify"):
            if "Proposed answer" in v["input_text"]:
                wiring_bad.append(iid + ":verify_has_A1")
        vans += [v["a"] for v in e1["verif"]]
    out[run] = dict(n_items=n, pct_e1_final_differs_from_e0_normalized=diff / n, pct_revise_equals_a1_verbatim=verbatim / (n - plan_inv),
                    plan_invalid=plan_inv, wiring_problems=wiring_bad,
                    verification_answers=dict(n=len(vans), yes_like=sum(bool(YES.match(a)) for a in vans),
                                              no_like=sum(bool(NO.match(a)) for a in vans),
                                              other=sum(not YES.match(a) and not NO.match(a) for a in vans),
                                              note="contradiction of A1 is NOT logged; yes/no-like counts are a surface heuristic only"))
write_json(ROOT / "runs" / "e1_sanity.json", out)
for r, v in out.items():
    print(r, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
# 5 random full traces from pilot-02-v2
calls = [c for c in read_jsonl(ROOT / "runs/pilot-02-v2/calls.jsonl") if c.get("error") is None]
ids = sorted({c["item_id"] for c in calls})
pick = sorted(np.random.Generator(np.random.PCG64(0)).choice(ids, 5, replace=False).tolist())
qs = load_questions(cfg, "train", pick)
traces = []
for iid in pick:
    get = lambda arm, step: [c for c in calls if c["item_id"] == iid and c["arm"] == arm and c["step"] == step]
    tr = dict(item_id=iid, question=qs[iid], A1=parse_answer(get("E0", "answer")[-1]["raw_output"], 10)[0],
              plan_raw=get("E1", "plan")[-1]["raw_output"],
              verify=[(c["input_text"].split("Question: ")[-1].split("\n")[0], c["raw_output"]) for c in sorted(get("E1", "verify"), key=lambda c: c["step_idx"])],
              revise_raw=get("E1", "revise")[-1]["raw_output"])
    traces.append(tr)
write_json(ROOT / "runs" / "e1_sample_traces_pilot02v2.json", dict(seed="PCG64(0)", traces=traces))
for t in traces:
    print(f"\n### {t['item_id']}: {t['question']}\nA1 (E0): {t['A1']}\nPLAN:\n{t['plan_raw']}")
    for q, a in t["verify"]: print(f"  VQ: {q}\n  VA: {a}")
    print(f"REVISE: {t['revise_raw']}")
