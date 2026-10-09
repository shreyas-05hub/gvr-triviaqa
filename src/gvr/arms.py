"""E0 / E1 (CoVe-style, factored) / E3 (BM25 RAG) runners.

CoVe-style (factored); same-model verification is not independent evidence.
E1 verification answers are never ground truth and never used in scoring.
"""
from . import prompts as P
from .parsing import parse_answer, parse_plan, PARSER_VERSION
from .scoring.normalize import token_jaccard


class Runner:
    def __init__(self, cfg, qa, logger, bm25=None, require_frozen=True):
        if require_frozen and not getattr(logger, "fixture", False):
            P.assert_frozen()   # real runs abort if any active generation prompt != frozen hash
        self.cfg, self.qa, self.log, self.bm25 = cfg, qa, logger, bm25
        self.mx = cfg["generation"]["max_new_tokens"]
        self.maxw = cfg["parser"]["max_words"]
        self.retries = cfg["reproducibility"]["max_retries_per_call"]

    def _gen(self, item_id, arm, step, step_idx, messages, max_new, prompt_ids, extra=None):
        def fn():
            r = self.qa.generate(messages, max_new)
            rec = {k: r[k] for k in ("input_text", "raw_output", "input_tokens", "output_tokens",
                                     "token_logprobs", "t_start", "t_end", "u", "gen_ids")}
            rec.update(extra or {})
            return rec
        meta = dict(item_id=item_id, arm=arm, step=step, step_idx=step_idx, **self.qa.meta(),
                    prompt_id="+".join(prompt_ids), prompt_sha={p: P.sha(p) for p in prompt_ids},
                    parser_version=PARSER_VERSION)
        return self.log.run(fn, meta, self.retries)

    def e0(self, item_id, question):
        rec, ok = self._gen(item_id, "E0", "answer", 0, P.qa_messages(question), self.mx["answer"],
                            ["system_qa", "user_qa"])
        return self._final(item_id, "E0", rec, ok, calls=1)

    def _final(self, item_id, arm, rec, ok, calls, **extra):
        if not ok:
            return dict(item_id=item_id, arm=arm, answer="", invalid=True, u=None, failed=True,
                        n_calls=calls, **extra)
        ans, invalid = parse_answer(rec["raw_output"], self.maxw)
        u = rec.get("u")
        if u is None:
            invalid = True
        return dict(item_id=item_id, arm=arm, answer=ans, invalid=invalid, u=u, failed=False,
                    n_calls=calls, **extra)

    def _fit_passages(self, question, passages):
        """Cap templated input at max tokens; truncate passage text from the last passage backwards."""
        cap = self.cfg["retrieval"]["rag_max_input_tokens"]
        ps = [dict(p) for p in passages]
        if self.qa.n_tokens(P.rag_messages(question, ps)) <= cap:
            return ps, False
        tok = self.qa.tok
        for j in range(len(ps) - 1, -1, -1):
            ids = tok(ps[j]["text"], add_special_tokens=False)["input_ids"]
            while ids:
                over = self.qa.n_tokens(P.rag_messages(question, ps)) - cap
                if over <= 0:
                    return ps, True
                ids = ids[:max(0, len(ids) - max(over, 1))]
                ps[j]["text"] = tok.decode(ids)
            if self.qa.n_tokens(P.rag_messages(question, ps)) <= cap:
                return ps, True
        raise ValueError("prompt exceeds cap even with all passage text removed")

    def e3(self, item_id, question):
        def rfn():
            passages, r = self.bm25.search(question)
            return dict(retrieval=r, passages=passages, t_start=r["t_start"], t_end=r["t_end"])
        rmeta = dict(item_id=item_id, arm="E3", step="retrieve", step_idx=0, input_text=question)
        rrec, ok = self.log.run(rfn, rmeta, self.retries)
        if not ok:
            return dict(item_id=item_id, arm="E3", answer="", invalid=True, u=None, failed=True,
                        n_calls=1, n_retrievals=1)
        passages, truncated = self._fit_passages(question, rrec["passages"])
        rec, ok = self._gen(item_id, "E3", "answer", 1, P.rag_messages(question, passages),
                            self.mx["answer"], ["system_rag", "user_rag", "passage"],
                            extra=dict(truncated=truncated))
        return self._final(item_id, "E3", rec, ok, calls=1, n_retrievals=1, truncated=truncated,
                           passage_ids=rrec["retrieval"]["passage_ids"],
                           passages_full=[p["text"] for p in rrec["passages"]])

    def e1(self, item_id, question, a1, a1_invalid):
        """a1 = the LOGGED E0 answer (never regenerated). If the plan parses to 0 questions, final = a1."""
        rec, ok = self._gen(item_id, "E1", "plan", 0, P.plan_messages(question, a1),
                            self.mx["cove_plan"], ["system_cove_plan", "user_cove_plan"])
        calls = 1
        qs = parse_plan(rec["raw_output"], self.cfg["cove"]["n_verification_questions"]) if ok else []
        if not qs:
            return dict(item_id=item_id, arm="E1", answer=a1, invalid=a1_invalid, u=None, failed=not ok,
                        sv_plan_invalid=True, n_calls=calls, verif=[])
        verif = []
        for i, vq in enumerate(qs, 1):
            vrec, vok = self._gen(item_id, "E1", "verify", i, P.qa_messages(vq), self.mx["cove_verify"],
                                  ["system_qa", "user_qa"])
            calls += 1
            va = parse_answer(vrec["raw_output"], self.maxw)[0] if vok else ""
            jac = token_jaccard(vq, question)
            verif.append(dict(q=vq, a=va, jaccard=jac,
                              paraphrase_flag=jac >= self.cfg["cove"]["paraphrase_jaccard_flag"]))
        rrec, rok = self._gen(item_id, "E1", "revise", len(qs) + 1,
                              P.revise_messages(question, a1, [(v["q"], v["a"]) for v in verif]),
                              self.mx["cove_revise"], ["system_qa", "user_cove_revise"])
        calls += 1
        out = self._final(item_id, "E1", rrec, rok, calls=calls, sv_plan_invalid=False, verif=verif)
        return out
