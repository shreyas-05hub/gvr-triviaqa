# Prompts
Each `*_vN.txt` file is a versioned template. Hashes are recorded in `prompts/registry.json`
(`python scripts/hash_prompts.py`). Wording may change during the pilot ONLY to fix format/parsing
failures; each change gets a new version file plus a `changelog.jsonl` entry with its reason.
Any change after calibration voids the gate threshold.

Specified verbatim by the protocol: system_qa, user_qa, system_rag, user_rag, passage.
Draft v1 (wording not specified by the protocol; pending user review): system_cove_plan, user_cove_plan,
user_cove_revise, judge. The CoVe verify step reuses system_qa + user_qa (fresh context).
