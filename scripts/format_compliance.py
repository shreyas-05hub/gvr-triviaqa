"""format_compliance_rule: v1 (pilot-01) vs v2 (pilot-02-v2) per arm. NO gold, NO EM, NO accuracy."""
import hashlib
import json
import re
import numpy as np
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items

cfg = load_config()
rule = cfg["format_compliance_rule"]
assert hashlib.sha256(rule["text"].rstrip("\n").encode()).hexdigest() == rule["sha256"], "rule hash"
EX = set(cfg["uncertainty"]["excluded_token_ids"])
VERBS = re.compile(r"\b(is|was|are|were|has|had|have|did|does|do|became|played|wrote|won|refers|means)\b", re.I)
FINAL = {"E0": "answer", "E1": "revise", "E3": "answer"}


def profile(run):
    calls = read_jsonl(ROOT / "runs" / run / "calls.jsonl")
    ids = sorted({c["item_id"] for c in calls})
    items = derive_items(calls, cfg, load_questions(cfg, "train", ids))
    last = {}
    for c in calls:
        if c.get("error") is None and c["step"] == FINAL.get(c["arm"]):
            last[(c["item_id"], c["arm"])] = c
    out = {}
    for arm in FINAL:
        its = {k: v for k, v in items.items() if k[1] == arm}
        W, T, sent, inv = [], [], 0, 0
        for (iid, _), it in its.items():
            w = len(it["answer"].split()); W.append(w)
            c = last.get((iid, arm)) or next(x for x in calls if x["item_id"] == iid and x["arm"] == "E0" and x["step"] == "answer")
            T.append(sum(t not in EX for t in c["gen_ids"]))
            first = next((l for l in c["raw_output"].splitlines() if l.strip()), "").strip()
            sent += w >= 5 and (first.endswith(".") or bool(VERBS.search(it["answer"])))
            inv += bool(it["invalid"])
        W, T, n = np.array(W), np.array(T), len(its)
        r = dict(n=n, words=dict(mean=float(W.mean()), median=float(np.median(W)), p90=float(np.percentile(W, 90)), max=int(W.max())),
                 model_tokens_excl_eos=dict(mean=float(T.mean()), median=float(np.median(T)), p90=float(np.percentile(T, 90))),
                 pct_gt5_words=float((W > 5).mean()), pct_full_sentence=sent / n, invalid_rate=inv / n,
                 prompt_ids=sorted({c["prompt_id"] for c in calls if c["arm"] == arm and c["step"] != "retrieve"}),
                 parser_version_logged=sorted({c.get("parser_version") for c in calls if c["step"] != "retrieve"}))
        if arm == "E1":
            v = [q for it in its.values() for q in it.get("verif", [])]
            r.update(plan_invalid_rate=sum(bool(it.get("sv_plan_invalid")) for it in its.values()) / n,
                     n_verification_questions=len(v), paraphrase_flags=sum(q["paraphrase_flag"] for q in v))
        out[arm] = r
    return out


v1, v2 = profile(rule["baseline_run"]), profile(rule["v2_run"])
th = rule["thresholds"]
acc = {a: dict(gt5=v2[a]["pct_gt5_words"] <= th["max_pct_gt5_words"],
               full_sentence=v2[a]["pct_full_sentence"] <= th["max_pct_full_sentence"],
               invalid_not_worse=v2[a]["invalid_rate"] <= v1[a]["invalid_rate"]) for a in FINAL}
res = dict(rule_sha256=rule["sha256"], note="no gold, no EM, no accuracy computed", v1=v1, v2=v2, acceptance=acc,
           passed=all(all(x.values()) for x in acc.values()))
write_json(ROOT / "runs" / "format_compliance_v1_v2.json", res)
print(json.dumps(res, indent=1))
