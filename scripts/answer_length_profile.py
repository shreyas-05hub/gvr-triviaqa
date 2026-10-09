"""Answer-length profile per arm over ALL pilot-01 + judgedev-01 outputs (descriptive; no accuracy, no gold)."""
import json
import re
import numpy as np
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items

cfg = load_config()
EX = set(cfg["uncertainty"]["excluded_token_ids"])
VERBS = re.compile(r"\b(is|was|are|were|has|had|have|did|does|do|became|played|wrote|won|refers|means)\b", re.I)
FINAL_STEP = {"E0": "answer", "E1": "revise", "E3": "answer"}
per = {a: dict(words=[], toks=[], sentence=0, period=0, n=0, plan_invalid_fallback=0) for a in FINAL_STEP}
for run in ("pilot-01", "judgedev-01"):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
    last = {}
    for c in calls:
        if c.get("error") is None and c["step"] == FINAL_STEP.get(c["arm"]):
            last[(c["item_id"], c["arm"])] = c
    for (iid, arm), it in items.items():
        p = per[arm]
        p["n"] += 1
        w = len(it["answer"].split())
        p["words"].append(w)
        c = last.get((iid, arm))
        if c is None:                      # E1 with 0 plan questions -> final = logged A1 (E0 answer call)
            p["plan_invalid_fallback"] += 1
            c = next(x for x in calls if x["item_id"] == iid and x["arm"] == "E0" and x["step"] == "answer" and x.get("error") is None)
        p["toks"].append(sum(t not in EX for t in c["gen_ids"]))
        first = next((l for l in c["raw_output"].splitlines() if l.strip()), "").strip()
        p["period"] += first.endswith(".")
        p["sentence"] += w >= 5 and (first.endswith(".") or bool(VERBS.search(it["answer"])))
out = {}
for a, p in per.items():
    W, T = np.array(p["words"]), np.array(p["toks"])
    out[a] = dict(n_outputs=p["n"], words=dict(mean=float(W.mean()), median=float(np.median(W)), p90=float(np.percentile(W, 90)),
                                               max=int(W.max())),
                  model_tokens_excl_eos=dict(mean=float(T.mean()), median=float(np.median(T)), p90=float(np.percentile(T, 90))),
                  pct_gt5_words=float((W > 5).mean()), pct_full_sentence=p["sentence"] / p["n"],
                  pct_first_line_ends_period=p["period"] / p["n"], e1_plan_invalid_fallback=p["plan_invalid_fallback"])
out["_rules"] = dict(words="whitespace tokens of the parsed answer", model_tokens="generated token ids of the final call, EOS ids excluded",
                     full_sentence=">=5 words AND (raw first line ends with '.' OR parsed answer contains a finite-verb keyword "
                                   "is|was|are|were|has|had|have|did|does|do|became|played|wrote|won|refers|means)",
                     final_call="E0 answer; E1 revise (or logged E0 answer if plan invalid); E3 answer")
write_json(ROOT / "runs" / "em_fn_diag" / "answer_length_profile.json", out)
for a in ("E0", "E1", "E3"):
    r = out[a]
    print(f"{a}: n={r['n_outputs']} words mean={r['words']['mean']:.2f} median={r['words']['median']:.0f} p90={r['words']['p90']:.1f} max={r['words']['max']} | "
          f"model tok mean={r['model_tokens_excl_eos']['mean']:.2f} median={r['model_tokens_excl_eos']['median']:.0f} p90={r['model_tokens_excl_eos']['p90']:.1f} | "
          f">5 words {100*r['pct_gt5_words']:.1f}% | full sentence {100*r['pct_full_sentence']:.1f}% | ends '.' {100*r['pct_first_line_ends_period']:.1f}% | E1 fallback {r['e1_plan_invalid_fallback']}")
