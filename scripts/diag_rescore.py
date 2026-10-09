"""DIAGNOSTIC ONLY (user decision A, option 1). Changes no arm, no u definition, no tolerance, no setting.
For the first 5 pilot questions, compares u computed by three paths on Qwen-7B NF4:
  gen     : current definition (generate logits, KV cache)       -- run twice (same-path reproducibility)
  nocache : generate with use_cache=False (full recompute per step)
  tf      : one teacher-forced forward pass over prompt+gen tokens -- run twice
Writes runs/diag/rescore_diag.json."""
import time
import torch
from _common import ROOT, write_json
from gvr import prompts as P
from gvr.config import load_config
from gvr.data import load_questions, read_ids
from gvr.model import QAModel, generation_config
from gvr.uncertainty import token_logprobs, mean_nll

cfg = load_config()
EX = cfg["uncertainty"]["excluded_token_ids"]
ids = read_ids("pilot_ids")[:5]
qs = load_questions(cfg, "train", ids)
qa = QAModel(cfg, fixture=False)


@torch.no_grad()
def gen_nocache(messages, max_new):
    gc = generation_config(cfg["generation"], max_new)
    gc.use_cache = False
    enc = qa.tok(qa.template(messages), add_special_tokens=False, return_tensors="pt").to(qa.device)
    out = qa.model.generate(**enc, generation_config=gc)
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
    lps = token_logprobs(out.logits, gen_ids)
    return gen_ids, lps, mean_nll(lps, gen_ids, EX)


def maxdiff(a, b, gen_ids, kept_only):
    d = [abs(x - y) for x, y, t in zip(a, b, gen_ids) if not (kept_only and t in EX)]
    return max(d) if d else None


rows = []
for i in ids:
    msgs = P.qa_messages(qs[i])
    mx = cfg["generation"]["max_new_tokens"]["answer"]
    g1 = qa.generate(msgs, mx)
    g2 = qa.generate(msgs, mx)
    nc_ids, nc_lps, nc_u = gen_nocache(msgs, mx)
    tf1_u, tf1_lps = qa.rescore(g1["input_ids"], g1["gen_ids"])
    tf2_u, tf2_lps = qa.rescore(g1["input_ids"], g1["gen_ids"])
    gl = g1["token_logprobs"]
    rows.append(dict(
        item_id=i, n_gen=len(g1["gen_ids"]), answer=g1["raw_output"],
        tokens_equal=dict(gen1_gen2=g1["gen_ids"] == g2["gen_ids"], gen1_nocache=g1["gen_ids"] == nc_ids),
        u=dict(gen1=g1["u"], gen2=g2["u"], nocache=nc_u, tf1=tf1_u, tf2=tf2_u),
        abs_delta_u=dict(gen_vs_tf=abs(g1["u"] - tf1_u), gen_vs_nocache=abs(g1["u"] - nc_u),
                         nocache_vs_tf=abs(nc_u - tf1_u), gen1_vs_gen2=abs(g1["u"] - g2["u"]),
                         tf1_vs_tf2=abs(tf1_u - tf2_u)),
        max_abs_delta_logprob=dict(
            gen_vs_tf_all=maxdiff(gl, tf1_lps, g1["gen_ids"], False),
            gen_vs_tf_kept=maxdiff(gl, tf1_lps, g1["gen_ids"], True),
            nocache_vs_tf_all=maxdiff(nc_lps, tf1_lps, g1["gen_ids"], False) if nc_ids == g1["gen_ids"] else None,
            gen_vs_nocache_all=maxdiff(gl, nc_lps, g1["gen_ids"], False) if nc_ids == g1["gen_ids"] else None),
        per_position=dict(gen=gl, nocache=nc_lps, tf=tf1_lps, gen_ids=g1["gen_ids"]),
    ))
    print(i, {k: round(v, 6) for k, v in rows[-1]["abs_delta_u"].items()})
write_json(ROOT / "runs" / "diag" / "rescore_diag.json",
           dict(diagnostic_only=True, model=qa.meta(), cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=torch.cuda.get_device_name(0), excluded_ids=EX, items=rows,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
