import json
from gvr.calllog import CallLogger, read_jsonl
from gvr.derive import derive_items

REQUIRED = ["run_id", "item_id", "split", "arm", "step", "step_idx", "model_id", "model_revision",
            "precision", "prompt_id", "prompt_sha", "input_text", "raw_output", "input_tokens",
            "output_tokens", "t_start", "t_end", "token_logprobs", "retrieval", "truncated", "error",
            "retry_of", "env_versions", "fixture"]


def meta(iid, arm, step, idx):
    return dict(item_id=iid, arm=arm, step=step, step_idx=idx, model_id="m", model_revision="r",
                precision="p", prompt_id="x", prompt_sha={})


def gen(raw, u=0.5):
    return lambda: dict(input_text="Question: q\nAnswer:", raw_output=raw, input_tokens=5, output_tokens=2,
                        token_logprobs=[-0.5, -0.1], t_start=0.0, t_end=1.0, u=u, gen_ids=[1, 151645])


def test_retry_once_and_fields(tmp_path):
    lg = CallLogger(tmp_path, "run1", "train", {"torch": "x"}, fixture=True)
    n = {"c": 0}
    def flaky():
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("boom")
        return gen("Paris")()
    rec, ok = lg.run(flaky, meta("q1", "E0", "answer", 0))
    assert ok
    def bad():
        raise RuntimeError("always")
    rec2, ok2 = lg.run(bad, meta("q2", "E0", "answer", 0))
    assert not ok2
    calls = read_jsonl(tmp_path / "calls.jsonl")
    assert len(calls) == 4
    assert calls[1]["retry_of"] == calls[0]["call_id"] and calls[0]["error"]
    assert calls[3]["retry_of"] == calls[2]["call_id"] and calls[3]["error"]
    for c in calls:
        assert c["fixture"] is True
        if c["error"] is None:
            for k in REQUIRED:
                assert k in c, k


def test_derive(tmp_path, cfg):
    lg = CallLogger(tmp_path, "r", "train", {})
    lg.run(gen("Answer: Paris."), meta("q1", "E0", "answer", 0))
    lg.run(gen(""), meta("q2", "E0", "answer", 0))
    lg.run(gen("1. Is Paris in France?\n2. Is Paris a capital?\n3. Capital of France?"), meta("q1", "E1", "plan", 0))
    for i in (1, 2, 3):
        lg.run(gen("Yes"), meta("q1", "E1", "verify", i))
    lg.run(gen("Paris"), meta("q1", "E1", "revise", 4))
    lg.run(gen("\n"), meta("q2", "E1", "plan", 0))   # 0 questions -> final = A1
    p = [dict(id="d1", score=9.0, title="Paris", text="Paris is the capital of France.")]
    lg.run(lambda: dict(retrieval=dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=["d1"], scores=[9.0],
                                       latency_s=0.1), passages=p, t_start=0, t_end=0.1),
           dict(item_id="q1", arm="E3", step="retrieve", step_idx=0, input_text="q"))
    lg.run(lambda: dict(gen("Paris")(), truncated=False), meta("q1", "E3", "answer", 1))
    items = derive_items(read_jsonl(tmp_path / "calls.jsonl"), cfg,
                         {"q1": "What is the capital of France?", "q2": "x"})
    assert items[("q1", "E0")]["answer"] == "Paris" and not items[("q1", "E0")]["invalid"]
    assert items[("q2", "E0")]["invalid"]
    e1 = items[("q1", "E1")]
    assert e1["answer"] == "Paris" and len(e1["verif"]) == 3 and e1["cost"]["n_calls"] == 5
    assert items[("q2", "E1")]["sv_plan_invalid"] and items[("q2", "E1")]["invalid"]
    e3 = items[("q1", "E3")]
    assert e3["cost"]["n_calls"] == 1 and e3["cost"]["n_retrievals"] == 1 and e3["passage_ids"] == ["d1"]
