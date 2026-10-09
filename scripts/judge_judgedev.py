"""judge_rule_v2 candidate A on the 170 EM=0 judge-dev targets ONLY (from runs/judgedev-01/blind_key.json).
Separate process; no Qwen. Never reads human labels (they are not in the package). Writes judge_v2.jsonl."""
import hashlib
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download
from _common import ROOT, write_json
from gvr.calllog import read_jsonl
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.model import bnb_config
from gvr.scoring.gold import load_gold
from gvr.scoring.judge import build_judge_prompt, check_judge_model, load_judge_template, parse_verdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "set CUDA_VISIBLE_DEVICES=0 (one GPU only)"
cfg = load_config()
jr = cfg["judge_rule_v2"]
assert hashlib.sha256(jr["text"].rstrip("\n").encode()).hexdigest() == jr["sha256"]
A = jr["candidate_A"]
check_judge_model(A["model_id"])
man = json.load(open(ROOT / "configs" / "freeze_manifest.json"))
pv = man["judge_v2_prompt"]
assert hashlib.sha256((ROOT / "prompts" / "judge_v2.txt").read_bytes()).hexdigest() == pv["sha256"]
assert "judgedev_labels" in man and man["judgedev_labels"]["recorded_before_any_judge_run_on_judgedev"]
template = load_judge_template("judge_v2")
run = ROOT / "runs" / "judgedev-01"
key = json.load(open(run / "blind_key.json"))
assert key["blind_csv_sha256"] == man["judgedev_blind"]["blind_csv_sha256"]
targets = [r for r in key["rows"] if not r["em"]]
assert len(targets) == 170, len(targets)
calls = read_jsonl(run / "calls.jsonl")
ids = sorted({c["item_id"] for c in calls})
qs = load_questions(cfg, "train", ids)
gold = load_gold(cfg, "train", ids)
items = derive_items(calls, cfg, qs)

torch.cuda.reset_peak_memory_stats()
local = snapshot_download(A["model_id"], revision=A["revision"], allow_patterns=cfg["scoring"]["judge"]["allow_patterns"])
assert not any(f.startswith("consolidated") for f in os.listdir(local))
tok = AutoTokenizer.from_pretrained(local)
model = AutoModelForCausalLM.from_pretrained(local, quantization_config=bnb_config(cfg["model"]["precision"]),
                                             device_map="cuda:0").eval()
gc = GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                      max_new_tokens=cfg["scoring"]["judge"]["max_new_tokens"], pad_token_id=tok.eos_token_id)
CTX = model.config.max_position_embeddings
out = run / "judge_v2.jsonl"
assert not out.exists(), "judge_v2.jsonl already exists"
lat, max_in = [], 0
with open(out, "w") as f, torch.no_grad():
    for r in targets:
        iid = r["item_id"]
        prompt = build_judge_prompt(template, qs[iid], sorted(set(gold[iid]["aliases"])), items[(iid, r["arm"])]["answer"])
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        n_in = enc["input_ids"].shape[1]
        max_in = max(max_in, n_in)
        if n_in + gc.max_new_tokens > CTX:
            raise SystemExit(f"STOP: judge input {n_in} exceeds context {CTX}; not truncating")
        t0 = time.time()
        o = model.generate(**enc, generation_config=gc)
        lat.append(time.time() - t0)
        raw = tok.decode(o[0, n_in:], skip_special_tokens=True)
        v = parse_verdict(raw)
        f.write(json.dumps(dict(row_id=r["row_id"], item_id=iid, arm=r["arm"], raw=raw, verdict=v, judge_unparsed=v is None,
                                judge_input_tokens=n_in, n_aliases=len(set(gold[iid]["aliases"])), prompt="judge_v2",
                                prompt_sha=pv["sha256"], judge_model=A["model_id"], judge_revision=A["revision"])) + "\n")
write_json(run / "judge_v2_meta.json", dict(n=len(targets), max_input_tokens=max_in, context=CTX,
           gpu=torch.cuda.get_device_name(0), n_visible_gpus=torch.cuda.device_count(),
           peak_gpu_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()), latency_mean_s=sum(lat) / len(lat)))
print("judged", len(targets))
