"""judge_rule_v2 step: PARSE CHECK ONLY of judge_v2 (frozen) on the 105 pilot judge requests.
Separate process after Qwen is unloaded. Logs raw outputs + parse_ok only. Never reads human labels, never
computes agreement, never runs on judge-dev outputs."""
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
from gvr.scoring.analyze import judge_requests
from gvr.scoring.gold import load_gold
from gvr.scoring.judge import build_judge_prompt, check_judge_model, load_judge_template, parse_verdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "set CUDA_VISIBLE_DEVICES=0 (one GPU only)"
cfg = load_config()
A = cfg["judge_rule_v2"]["candidate_A"]
check_judge_model(A["model_id"])
man = json.load(open(ROOT / "configs" / "freeze_manifest.json"))["judge_v2_prompt"]
tpath = ROOT / "prompts" / "judge_v2.txt"
assert hashlib.sha256(tpath.read_bytes()).hexdigest() == man["sha256"] and man["status"] == "frozen-for-judge-dev-validation"
template = load_judge_template("judge_v2")

calls = read_jsonl(ROOT / "runs" / "pilot-01" / "calls.jsonl")   # pilot rows ONLY
ids = sorted({c["item_id"] for c in calls})
qs = load_questions(cfg, "train", ids)
reqs = judge_requests(derive_items(calls, cfg, qs), load_gold(cfg, "train", ids), qs, cfg["scoring"]["abstain_list"])
assert len(reqs) == 105, len(reqs)

torch.cuda.reset_peak_memory_stats()
local = snapshot_download(A["model_id"], revision=A["revision"], allow_patterns=cfg["scoring"]["judge"]["allow_patterns"])
assert not any(f.startswith("consolidated") for f in os.listdir(local))
tok = AutoTokenizer.from_pretrained(local)
model = AutoModelForCausalLM.from_pretrained(local, quantization_config=bnb_config(cfg["model"]["precision"]),
                                             device_map="cuda:0").eval()
gc = GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                      max_new_tokens=cfg["scoring"]["judge"]["max_new_tokens"], pad_token_id=tok.eos_token_id)
CTX = model.config.max_position_embeddings
out_dir = ROOT / "runs" / "judgedev-01"
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "parse_check.jsonl"
assert not path.exists()
n_ok, lat, max_in = 0, [], 0
with open(path, "w") as f, torch.no_grad():
    for r in reqs:
        prompt = build_judge_prompt(template, r["question"], r["gold_aliases"], r["prediction"])
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
        ok = parse_verdict(raw) is not None
        n_ok += ok
        f.write(json.dumps(dict(item_id=r["item_id"], arm=r["arm"], raw=raw, parse_ok=ok, judge_input_tokens=n_in,
                                n_aliases=len(r["gold_aliases"]), prompt="judge_v2", prompt_sha=man["sha256"],
                                judge_model=A["model_id"], judge_revision=A["revision"])) + "\n")
write_json(out_dir / "parse_check_summary.json", dict(
    purpose="parse check only (judge_rule_v2); no agreement computed", n=len(reqs), n_parsed=n_ok,
    parse_rate=n_ok / len(reqs), max_input_tokens=max_in, context=CTX, gpu=torch.cuda.get_device_name(0),
    n_visible_gpus=torch.cuda.device_count(), peak_gpu_mem_allocated_bytes=int(torch.cuda.max_memory_allocated()),
    latency_mean_s=sum(lat) / len(lat)))
print(f"parse rate {n_ok}/{len(reqs)}")
