"""Qwen generation wrapper (GPU: 7B NF4; CPU: 0.5B fixture for unit tests only)."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .uncertainty import token_logprobs, mean_nll


def bnb_config(prec):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=prec["load_in_4bit"],
                              bnb_4bit_quant_type=prec["bnb_4bit_quant_type"],
                              bnb_4bit_use_double_quant=prec["bnb_4bit_use_double_quant"],
                              bnb_4bit_compute_dtype=getattr(torch, prec["bnb_4bit_compute_dtype"]))


def generation_config(gcfg, max_new_tokens):
    """Fresh config: every field explicit, so repo generation_config.json defaults never apply."""
    return GenerationConfig(do_sample=gcfg["do_sample"], temperature=gcfg["temperature"],
                            top_p=gcfg["top_p"], top_k=gcfg["top_k"],
                            repetition_penalty=gcfg["repetition_penalty"],
                            max_new_tokens=max_new_tokens, eos_token_id=gcfg["eos_token_id"],
                            pad_token_id=gcfg["pad_token_id"],
                            return_dict_in_generate=gcfg["return_dict_in_generate"],
                            output_logits=gcfg["output_logits"])


class QAModel:
    def __init__(self, cfg, fixture=False):
        self.cfg, self.fixture = cfg, fixture
        m = cfg["model"]
        if fixture:
            self.model_id, self.revision = m["fixture"]["id"], m["fixture"].get("revision")
            self.precision = "fixture-cpu-float32"
            kw = dict(dtype=torch.float32)
        else:
            if not m["revision"]:
                raise ValueError("model.revision must be pinned to a commit SHA")
            self.model_id, self.revision = m["id"], m["revision"]
            self.precision = m["precision"]["name"]
            import os
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "set CUDA_VISIBLE_DEVICES=0 (one GPU only)"
            kw = dict(quantization_config=bnb_config(m["precision"]), device_map="cuda:0")
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, **kw)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def template(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def n_tokens(self, messages):
        return len(self.tok(self.template(messages), add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate(self, messages, max_new_tokens):
        g = self.cfg["generation"]
        text = self.template(messages)
        enc = self.tok(text, add_special_tokens=False, return_tensors="pt").to(self.device)
        t0 = time.time()
        out = self.model.generate(**enc, generation_config=generation_config(g, max_new_tokens))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        n_in = enc["input_ids"].shape[1]
        gen_ids = out.sequences[0, n_in:].tolist()
        lps = token_logprobs(out.logits, gen_ids)
        u = mean_nll(lps, gen_ids, self.cfg["uncertainty"]["excluded_token_ids"])
        return dict(input_text=text, input_ids=enc["input_ids"][0].tolist(), gen_ids=gen_ids,
                    raw_output=self.tok.decode(gen_ids, skip_special_tokens=True),
                    input_tokens=n_in, output_tokens=len(gen_ids), token_logprobs=lps, u=u,
                    n_logits=len(out.logits), argmax_ids=[int(l[0].argmax()) for l in out.logits],
                    t_start=t0, t_end=t1)

    @torch.no_grad()
    def rescore(self, input_ids, gen_ids):
        """Teacher-forced mean NLL of gen_ids given input_ids (one forward pass)."""
        ids = torch.tensor([input_ids + gen_ids], device=self.device)
        logits = self.model(input_ids=ids).logits[0]
        n = len(input_ids)
        lps = [torch.log_softmax(logits[n - 1 + i].float(), -1)[t].item() for i, t in enumerate(gen_ids)]
        return mean_nll(lps, gen_ids, self.cfg["uncertainty"]["excluded_token_ids"]), lps

    def meta(self):
        return dict(model_id=self.model_id, model_revision=self.revision, precision=self.precision)
