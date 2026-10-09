"""End-to-end on synthetic calls: derive -> judge requests -> score -> gate -> compose -> random gate -> report."""
import numpy as np
from gvr.calllog import CallLogger, read_jsonl
from gvr.derive import derive_items
from gvr.scoring.analyze import analyze, judge_requests, score_items, write_handlabel_csv
from gvr.report import render


def test_synthetic_pipeline(tmp_path, cfg):
    rng = np.random.default_rng(0)
    lg = CallLogger(tmp_path, "syn", "train", {}, fixture=True)
    gold, qs = {}, {}
    def m(i, arm, step, idx):
        return dict(item_id=i, arm=arm, step=step, step_idx=idx, model_id="fixture", model_revision="x",
                    precision="fixture", prompt_id="p", prompt_sha={})
    def g(raw, u):
        return lambda: dict(input_text="Question: q\nAnswer:", raw_output=raw, input_tokens=5, output_tokens=2,
                            token_logprobs=[-u], t_start=0.0, t_end=0.5, u=u, gen_ids=[1])
    for n in range(30):
        i = f"t{n}"
        qs[i] = f"question {n}?"
        gold[i] = {"aliases": [f"Ans{n}"], "normalized_aliases": [f"ans{n}"], "value": f"Ans{n}"}
        e0 = f"Ans{n}" if n % 3 else ("" if n == 0 else f"wrong{n}")
        lg.run(g(e0, float(rng.random())), m(i, "E0", "answer", 0))
        lg.run(g("Q a?\nQ b?\nQ c?", 0.1), m(i, "E1", "plan", 0))
        for k in (1, 2, 3):
            lg.run(g("x", 0.1), m(i, "E1", "verify", k))
        lg.run(g(f"Ans{n}" if n % 2 else "nope", 0.2), m(i, "E1", "revise", 4))
        p = [dict(id=f"d{n}", score=1.0, title="t", text=f"text with ans{n}" if n % 4 else "nothing")]
        lg.run(lambda p=p: dict(retrieval=dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=[x["id"] for x in p],
                                               scores=[1.0], latency_s=0.01), passages=p, t_start=0, t_end=0.01),
               dict(item_id=i, arm="E3", step="retrieve", step_idx=0, input_text=qs[i]))
        lg.run(lambda: dict(g("I don't know" if n == 5 else f"Ans{n}", 0.3)(), truncated=False), m(i, "E3", "answer", 1))
    items = derive_items(read_jsonl(tmp_path / "calls.jsonl"), cfg, qs)
    reqs = judge_requests(items, gold, qs, cfg["scoring"]["abstain_list"])
    assert all(not (r["arm"] == "E3" and r["item_id"] == "t5") for r in reqs)  # abstain not judged
    verdicts = {(r["item_id"], r["arm"]): False for r in reqs}
    write_handlabel_csv(tmp_path / "h.csv", reqs, verdicts)
    assert (tmp_path / "h.csv").read_text().count("\n") == len(reqs) + 1
    up = [(r["item_id"], r["arm"]) for r in reqs[:2]]
    scored = score_items(items, gold, verdicts, cfg["scoring"]["abstain_list"], up)
    res = analyze(scored, cfg, seeds=range(50))
    assert sum(res["summary"][a]["judge_unparsed"] for a in ("E0", "E1", "E3")) == 2
    assert all(res["summary"][a]["N"] == 30 for a in res["summary"])   # no denominator exclusion
    assert all(not scored[k]["final"] for k in up)
    assert res["summary"]["E0"]["invalid"] == 1 / 30
    assert res["gate_rates"]["n_valid"] == 29
    md = render(res, {}, {}, cfg["go_no_go"], {}, {})
    assert "Engineering pilot; descriptive only; no scientific conclusions." in md
