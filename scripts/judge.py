"""Stage 7 judge: separate process, after Qwen is unloaded. Same NF4 config. Never Qwen."""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from _common import ROOT
from gvr.calllog import read_jsonl
from gvr.config import load_config, sha256_file
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.model import bnb_config
from gvr.scoring.analyze import judge_requests
from gvr.scoring.gold import load_gold
from gvr.scoring.judge import Judge, load_judge_template, check_judge_model
from gvr import prompts as P

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
a = ap.parse_args()
cfg = load_config()
jc = cfg["scoring"]["judge"]
check_judge_model(jc["model_id"])
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "set CUDA_VISIBLE_DEVICES=0 (one GPU only)"
assert jc["revision"], "judge revision must be pinned"
run_dir = ROOT / "runs" / a.run_id
calls = read_jsonl(run_dir / "calls.jsonl")
ids = sorted({c["item_id"] for c in calls})
qs = load_questions(cfg, "train", ids)
gold = load_gold(cfg, "train", ids)
reqs = judge_requests(derive_items(calls, cfg, qs), gold, qs, cfg["scoring"]["abstain_list"])

import time
torch.cuda.reset_peak_memory_stats()
from huggingface_hub import snapshot_download
local = snapshot_download(jc["model_id"], revision=jc["revision"], allow_patterns=jc["allow_patterns"])
assert not any(f.startswith("consolidated") for f in __import__("os").listdir(local)), "consolidated weights downloaded"
t_load = time.time()
tok = AutoTokenizer.from_pretrained(local)
model = AutoModelForCausalLM.from_pretrained(local,
                                             quantization_config=bnb_config(cfg["model"]["precision"]),
                                             device_map="cuda:0").eval()
gc = GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                      max_new_tokens=jc["max_new_tokens"], pad_token_id=tok.eos_token_id)


load_s = time.time() - t_load
CTX = model.config.max_position_embeddings
LAST = {}


@torch.no_grad()
def backend(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]
    LAST["input_tokens"] = n_in
    LAST["max_in"] = max(LAST.get("max_in", 0), n_in)
    if n_in + jc["max_new_tokens"] > CTX:
        raise SystemExit(f"STOP: judge input {n_in} tokens exceeds context {CTX}; not truncating. Report to user.")
    out = model.generate(**enc, generation_config=gc)
    return tok.decode(out[0, n_in:], skip_special_tokens=True)


judge = Judge(jc["model_id"], backend, load_judge_template())
lat = []
assert not (run_dir / "judge.jsonl").exists(), "judge.jsonl already exists for this run"
with open(run_dir / "judge.jsonl", "a") as f:
    for r in reqs:
        t0 = time.time()
        raw, v = judge(r["question"], r["gold_aliases"], r["prediction"])  # full alias list, greedy, no retry
        lat.append(time.time() - t0)
        f.write(json.dumps(dict(item_id=r["item_id"], arm=r["arm"], raw=raw, verdict=v, judge_unparsed=v is None,
                                n_aliases=len(r["gold_aliases"]), judge_input_tokens=LAST["input_tokens"],
                                judge_context=CTX, judge_model=jc["model_id"], judge_revision=jc["revision"],
                                judge_prompt_sha=P.sha("judge"))) + "\n")
qc = model.config.quantization_config
qd = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
json.dump(dict(judge_model=jc["model_id"], judge_revision=jc["revision"], n_requests=len(reqs),
               by_arm={a: sum(r["arm"] == a for r in reqs) for a in ("E0", "E1", "E3")},
               context=CTX, max_input_tokens=LAST.get("max_in"), model_load_s=load_s,
               quantization_config={k: qd.get(k) for k in ("load_in_4bit", "bnb_4bit_quant_type",
                                    "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype")},
               files=sorted(os.listdir(local)), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               gpu=torch.cuda.get_device_name(0), n_visible_gpus=torch.cuda.device_count(),
               peak_gpu_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()),
               latency_s=dict(n=len(lat), mean=sum(lat) / len(lat) if lat else None, max=max(lat) if lat else None),
               latency_s_excl_warmup=(sum(lat[3:]) / len(lat[3:])) if len(lat) > 3 else None),
          open(run_dir / "judge_meta.json", "w"), indent=2)
print("judged", len(reqs))
