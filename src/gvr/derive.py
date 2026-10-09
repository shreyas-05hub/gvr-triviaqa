"""items.jsonl is DERIVED from calls.jsonl only (recomputable)."""
from collections import defaultdict
from .parsing import parse_answer, parse_plan
from .scoring.normalize import token_jaccard


def _ok(calls):
    return [c for c in calls if c.get("error") is None]


def cost_of(calls):
    return dict(n_calls=sum(c["step"] != "retrieve" for c in calls),
                n_retrievals=sum(c["step"] == "retrieve" for c in calls),
                n_attempts=len(calls), n_retries=sum(c.get("retry_of") is not None for c in calls),
                input_tokens=sum(c.get("input_tokens") or 0 for c in calls),
                output_tokens=sum(c.get("output_tokens") or 0 for c in calls),
                wall_s=sum((c.get("t_end") or 0) - (c.get("t_start") or 0) for c in calls))


def _answer_from(call, maxw):
    if call is None:
        return dict(answer="", invalid=True, u=None, failed=True)
    ans, inv = parse_answer(call["raw_output"], maxw)
    u = call.get("u")
    return dict(answer=ans, invalid=inv or u is None, u=u, failed=False)


def derive_items(calls, cfg, questions):
    maxw = cfg["parser"]["max_words"]
    n_vq = cfg["cove"]["n_verification_questions"]
    thr = cfg["cove"]["paraphrase_jaccard_flag"]
    g = defaultdict(list)
    for c in calls:
        g[(c["item_id"], c["arm"])].append(c)
    items = {}
    for (iid, arm), cs in g.items():
        if arm not in ("E0", "E3"):
            continue
        ok = _ok(cs)
        ans = next((c for c in reversed(ok) if c["step"] == "answer"), None)
        it = dict(item_id=iid, arm=arm, **_answer_from(ans, maxw), cost=cost_of(cs))
        if arm == "E3":
            r = next((c for c in reversed(ok) if c["step"] == "retrieve"), None)
            it.update(passage_ids=r["retrieval"]["passage_ids"] if r else [],
                      passages_full=[p["text"] for p in r["passages"]] if r else [],
                      truncated=ans.get("truncated") if ans else None,
                      retrieval_failed=r is None)
            if r is None:
                it.update(answer="", invalid=True, failed=True)
        items[(iid, arm)] = it
    for (iid, arm), cs in g.items():
        if arm != "E1":
            continue
        ok = _ok(cs)
        e0 = items[(iid, "E0")]
        plan = next((c for c in reversed(ok) if c["step"] == "plan"), None)
        qs = parse_plan(plan["raw_output"], n_vq) if plan else []
        if not qs:
            it = dict(item_id=iid, arm="E1", answer=e0["answer"], invalid=e0["invalid"], u=None,
                      failed=plan is None, sv_plan_invalid=True, verif=[])
        else:
            verif = []
            for i, q in enumerate(qs, 1):
                v = next((c for c in reversed(ok) if c["step"] == "verify" and c["step_idx"] == i), None)
                j = token_jaccard(q, questions[iid])
                verif.append(dict(q=q, a=parse_answer(v["raw_output"], maxw)[0] if v else "",
                                  jaccard=j, paraphrase_flag=j >= thr))
            rev = next((c for c in reversed(ok) if c["step"] == "revise"), None)
            it = dict(item_id=iid, arm="E1", **_answer_from(rev, maxw), sv_plan_invalid=False, verif=verif)
        it["cost"] = cost_of(cs)
        items[(iid, "E1")] = it
    return items
