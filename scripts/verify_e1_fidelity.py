"""READ-ONLY verification 1: E1 verify-call fidelity on test-c1..c3 (no artifacts modified)."""
import json
import re
import numpy as np
from _common import ROOT, write_json
from gvr import prompts as P
from gvr.config import load_config
from gvr.data import load_questions
from gvr.derive import derive_items
from gvr.parsing import parse_plan
from gvr.test_runner import effective_calls, done_units

cfg = load_config()
RX = re.compile(r"^<\|im_start\|>system\n(?P<sys>.*?)<\|im_end\|>\n<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>\n<\|im_start\|>assistant\n$", re.S)
calls = []
for r in ("test-c1", "test-c2", "test-c3"):
    d = ROOT / "runs" / r
    done = done_units(d)
    calls += [c for c in effective_calls(d) if (c["item_id"], c["arm"]) in done]
ids = sorted({c["item_id"] for c in calls})
qs = load_questions(cfg, "validation", ids)
items = derive_items(calls, cfg, qs)
plan = {c["item_id"]: c for c in calls if c["arm"] == "E1" and c["step"] == "plan" and c.get("error") is None}
ver = [c for c in calls if c["arm"] == "E1" and c["step"] == "verify" and c.get("error") is None]
SYS, UQ = P.text("system_qa"), P.text("user_qa")
cnt = dict(n_verify_calls=len(ver), parsed_chat_format=0, system_equals_system_qa_v1=0, user_equals_user_qa_v2_of_plan_vq=0,
           exact_reconstruction=0, A1_anywhere=0, A1_outside_vq=0, Q_verbatim_anywhere=0, Q_verbatim_outside_vq=0,
           extra_content_beyond_template_and_vq=0, prompt_sha_matches_frozen=0,
           A1_outside_vq_only_as_substring_of_fixed_template=0, A1_outside_vq_examples=[])
TMPL = SYS + "\n" + UQ.replace("{question}", "")
examples = []
for c in ver:
    iid, k = c["item_id"], c["step_idx"]
    vq = parse_plan(plan[iid]["raw_output"], cfg["cove"]["n_verification_questions"])[k - 1]
    a1 = items[(iid, "E0")]["answer"]; q = qs[iid]
    m = RX.match(c["input_text"])
    if not m:
        continue
    cnt["parsed_chat_format"] += 1
    sys_ok, user = m["sys"] == SYS, m["user"]
    cnt["system_equals_system_qa_v1"] += sys_ok
    exp_user = UQ.format(question=vq)
    cnt["user_equals_user_qa_v2_of_plan_vq"] += user == exp_user
    cnt["exact_reconstruction"] += sys_ok and user == exp_user
    outside = m["sys"] + "\n" + user.replace(vq, "")
    cnt["A1_anywhere"] += bool(a1) and a1 in c["input_text"]
    if bool(a1) and a1 in outside:
        cnt["A1_outside_vq"] += 1
        cnt["A1_outside_vq_only_as_substring_of_fixed_template"] += a1 in TMPL and user == exp_user
        cnt["A1_outside_vq_examples"].append(dict(item_id=iid, step_idx=k, A1=a1))
    cnt["Q_verbatim_anywhere"] += q in c["input_text"]
    cnt["Q_verbatim_outside_vq"] += q in outside
    cnt["extra_content_beyond_template_and_vq"] += user != exp_user
    cnt["prompt_sha_matches_frozen"] += c["prompt_sha"] == {"system_qa": P.sha("system_qa"), "user_qa": P.sha("user_qa")}
rng = np.random.Generator(np.random.PCG64(0))
for j in sorted(rng.choice(len(ver), 3, replace=False).tolist()):
    c = ver[j]
    examples.append(dict(item_id=c["item_id"], step_idx=c["step_idx"], original_question=qs[c["item_id"]],
                         A1=items[(c["item_id"], "E0")]["answer"], verify_input_text_verbatim=c["input_text"]))
protocol_quote = ("Stage 6 - E1 CoVe-style (factored), as given to Claude at session start (no 'Pilot Protocol v2' file exists in the repo): "
                  "\"Verify calls: each verification question answered in a FRESH context containing only that question (same system prompt; "
                  "max_new_tokens=32). A1 and the original question are not in this context.\"")
config_quote = dict(cove=cfg["cove"], max_new_tokens_cove_verify=cfg["generation"]["max_new_tokens"]["cove_verify"],
                    prompts_readme="The CoVe verify step reuses system_qa + user_qa (fresh context).")
impl = ("src/gvr/arms.py Runner.e1: verify call = P.qa_messages(vq) -> system_qa_v1 + user_qa_v2.format(question=vq); "
        "user_qa_v2 = 'Question: {question}\\nReply with only the shortest standard name, number, date or phrase - no extra words, "
        "qualifiers or sentence.\\nAnswer:'. Each verify call is a separate model.generate (no shared KV cache/history).")
finding = ("A1 and the original question are never injected by the template (A1_outside_vq = 0, Q_verbatim_outside_vq = 0 if counts below "
           "say so). Beyond the verification question and the system prompt, each verify input contains the user_qa_v2 scaffolding "
           "('Question:' prefix, the v2 format line, 'Answer:'). A1/Q strings can appear INSIDE the verification question itself because "
           "the plan step generates verification questions from Q and A1 (factored CoVe); short A1 strings may also match by chance.")
res = dict(counts=cnt, examples=examples, protocol_text=protocol_quote, config=config_quote, implementation=impl, finding=finding)
write_json(ROOT / "reports/verification/e1_fidelity.json", res)
print(json.dumps(cnt, indent=1))
