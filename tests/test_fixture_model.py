"""Stage 3/5/6 mechanics. Default: Qwen2.5-0.5B CPU FIXTURE (protocol amendment 3) on synthetic questions;
fixture outputs are never reported and logs are flagged fixture=True.
GVR_MODEL=main (GPU only): same tests on Qwen2.5-7B NF4 using the first 5 pilot questions (pass/fail only)."""
import os
import pytest
from gvr.calllog import CallLogger, read_jsonl
from gvr import prompts as P

MAIN = os.environ.get("GVR_MODEL") == "main"
pytestmark = pytest.mark.fixture
QS = ["What is the capital of France?", "Who wrote the play Hamlet?", "In which year did World War II end?",
      "What is the chemical symbol for gold?", "Which planet is known as the Red Planet?"]


if MAIN:
    from gvr.config import load_config
    from gvr.data import load_questions, read_ids
    _ids = read_ids("pilot_ids")[:5]
    _q = load_questions(load_config(), "train", _ids)
    QS = [_q[i] for i in _ids]


MEAS = {}   # GVR_MODEL=main only: measured values for runs/unit_main/measurements.json


@pytest.fixture(scope="module", autouse=True)
def _dump_measurements():
    yield
    if MAIN:
        import json, pathlib
        out = pathlib.Path(__file__).resolve().parents[1] / "runs" / "unit_main"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(MEAS, open(out / "measurements.json", "w"), indent=2, default=str)


@pytest.fixture(scope="module")
def qa(cfg):
    import time
    from gvr.model import QAModel
    if MAIN:
        import torch
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = QAModel(cfg, fixture=not MAIN)
    if MAIN:
        import os
        qc = m.model.config.quantization_config
        qd = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
        MEAS["model_load_s"] = time.time() - t0
        MEAS["quantization_config"] = {k: qd.get(k) for k in ("load_in_4bit", "bnb_4bit_quant_type",
                                       "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype")}
        MEAS["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        MEAS["device"] = str(m.device)
        MEAS["gpu_name"] = torch.cuda.get_device_name(0)
        MEAS["n_visible_gpus"] = torch.cuda.device_count()
        MEAS["peak_mem_after_load_bytes"] = int(torch.cuda.max_memory_allocated())
    return m


@pytest.fixture(scope="module")
def outs(qa, cfg):
    return [qa.generate(P.qa_messages(q), cfg["generation"]["max_new_tokens"]["answer"]) for q in QS]


def test_logits_length(outs):
    MEAS["items"] = [dict(n_logits=o["n_logits"], n_gen=len(o["gen_ids"]), input_tokens=o["input_tokens"],
                          raw_output=o["raw_output"], u=o["u"], last_token=o["gen_ids"][-1]) for o in outs]
    for o in outs:
        assert o["n_logits"] == len(o["gen_ids"]) == o["output_tokens"] > 0


def test_greedy_argmax(outs):
    MEAS["greedy_argmax_match"] = [o["argmax_ids"] == o["gen_ids"] for o in outs]
    for o in outs:
        assert o["argmax_ids"] == o["gen_ids"]


def test_teacher_forced_rescore(qa, outs):
    res = []
    for o in outs:
        u2, lps2 = qa.rescore(o["input_ids"], o["gen_ids"])
        res.append(dict(u_generate=o["u"], u_rescore=u2, abs_delta=None if u2 is None or o["u"] is None
                        else abs(u2 - o["u"])))
    MEAS["rescore"] = res
    for r, o in zip(res, outs):
        assert o["u"] is not None and r["abs_delta"] < 1e-3, r


def test_eos_excluded_and_generation_overrides(qa, outs, cfg):
    ex = set(cfg["uncertainty"]["excluded_token_ids"])
    for o in outs:
        kept = [lp for lp, t in zip(o["token_logprobs"], o["gen_ids"]) if t not in ex]
        assert abs(o["u"] + sum(kept) / len(kept)) < 1e-9
        if o["gen_ids"][-1] in ex:
            assert len(kept) == len(o["gen_ids"]) - 1
    MEAS["eos_exclusion"] = [dict(last_token_is_eos=o["gen_ids"][-1] in ex,
                                  n_kept=sum(t not in ex for t in o["gen_ids"]), n_gen=len(o["gen_ids"])) for o in outs]
    from gvr.model import generation_config
    gc = generation_config(cfg["generation"], 32)
    MEAS["generation_config_applied"] = {k: getattr(gc, k) for k in ("do_sample", "temperature", "top_p", "top_k",
                                         "repetition_penalty", "max_new_tokens", "eos_token_id", "pad_token_id",
                                         "return_dict_in_generate", "output_logits")}
    assert gc.do_sample is False and gc.repetition_penalty == 1.0
    assert gc.temperature is None and gc.top_p is None and gc.top_k is None
    assert gc.eos_token_id == [151645, 151643] and gc.pad_token_id == 151643
    assert qa.tok.convert_ids_to_tokens([151645, 151643]) == ["<|im_end|>", "<|endoftext|>"]


def test_chat_template(qa):
    t = qa.template(P.qa_messages("Q?"))
    assert t.endswith("<|im_start|>assistant\n") and P.qa_messages("Q?")[1]["content"] in t
    assert P.qa_messages("Q?")[1]["content"].endswith("\nAnswer:")


def test_runner_e0_e1_e3_logging(qa, cfg, tmp_path):
    from gvr.arms import Runner
    from gvr.derive import derive_items

    class StubBM25:  # fixture only: long passages to exercise the 2048-token cap
        def search(self, q):
            ps = [dict(id=f"s{i}", score=10.0 - i, title=f"T{i}", text=("lorem ipsum " * 300).strip())
                  for i in range(5)]
            return ps, dict(called=True, k=5, k1=0.9, b=0.4, passage_ids=[p["id"] for p in ps],
                            scores=[p["score"] for p in ps], latency_s=0.0, t_start=0.0, t_end=0.0)

    lg = CallLogger(tmp_path, "test-run", "test-harness", {}, fixture=not MAIN)
    r = Runner(cfg, qa, lg, bm25=StubBM25())
    e0 = r.e0("f0", QS[0])
    r.e1("f0", QS[0], e0["answer"], e0["invalid"])
    r.e3("f0", QS[0])
    calls = read_jsonl(tmp_path / "calls.jsonl")
    assert all(c["fixture"] is (not MAIN) for c in calls)
    gen = [c for c in calls if c["step"] != "retrieve"]
    assert all(c["model_id"] == (cfg["model"]["id"] if MAIN else cfg["model"]["fixture"]["id"]) for c in gen)
    verify = [c for c in calls if c["step"] == "verify"]
    for c in verify:  # fresh context: neither original question nor A1
        assert QS[0] not in c["input_text"] and "Proposed answer" not in c["input_text"]
    e3a = next(c for c in calls if c["arm"] == "E3" and c["step"] == "answer")
    if MAIN:
        import torch
        MEAS["e3_cap_test"] = dict(input_tokens=e3a["input_tokens"], truncated=e3a["truncated"], error=e3a["error"],
                                   peak_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                   peak_mem_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                                   n_errors=sum(c["error"] is not None for c in calls))
    assert e3a["input_tokens"] <= 2048 and e3a["truncated"] is True
    assert "Proposed answer" not in e3a["input_text"]
    items = derive_items(calls, cfg, {"f0": QS[0]})
    assert set(items) == {("f0", "E0"), ("f0", "E1"), ("f0", "E3")}
