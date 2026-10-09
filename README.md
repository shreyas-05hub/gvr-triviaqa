# Gating Self-Verification or Retrieval? — engineering pilot code

**Engineering pilot; descriptive only; no scientific conclusions.** Pilot numbers are never evidence that any
method improves accuracy.

Arms: E0 baseline · E1 CoVe-style (factored) self-verification · E3 BM25 RAG · E2 = gated E1 · E4 = gated E3 ·
random-gate control. **CoVe-style (factored); same-model verification is not independent evidence.**
E1 verification answers are never ground truth and never used in scoring.

## Layout
- `configs/config.yaml` – every protocol parameter (no hidden defaults); `freeze_manifest.json` (`test_unlocked: false`).
- `src/gvr/` – generation side (`data`, `prompts`, `model`, `arms`, `retrieval`, `calllog`, `derive`, `gate`,
  `compose`, `random_gate`, `manifests`, `report`). `src/gvr/scoring/` – the ONLY code that loads gold.
- `prompts/` – versioned templates + `registry.json` (SHA-256). See `prompts/README.md`.
- `scripts/` – stage scripts. `notebooks/` – Kaggle audit + runner (generated from `notebooks/src/`).
- `env/` – `requirements-core.in`/`requirements-gpu.in` and compiled exact locks (Python 3.11).

## Laptop (CPU)
```
uv venv --python 3.11 .venv
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r env/requirements-core.lock.txt   # torch already satisfied by the CPU build
.venv/bin/python -m pytest -q -m "not fixture"   # fast suite
.venv/bin/python -m pytest -q -m fixture         # Qwen2.5-0.5B CPU fixture (test-only, ~1 GB)
```
## GPU (Kaggle T4)
`scripts/package_repo.sh` → upload `gvr_repo.zip` as a dataset → `notebooks/kaggle_runner.ipynb`
(stages: env, test_main, bm25_meta, bm25_load, pilot, determinism, judge, report). Persist `/kaggle/working/out`
as an output dataset between sessions.

## What is and isn't released

**Released:** code, configuration, the eight frozen prompt templates, a redacted copy of the freeze manifest
(`configs/freeze_manifest.public.json`; hashes refer to the private originals), analysis scripts, identifiers
(pilot, calibration, judge-development and test question IDs; model, dataset and index identifiers), aggregate results,
and this README.

**Not released:** TriviaQA question/answer/alias text, the BM25 index, call logs and any output containing questions,
answers, aliases or retrieved passage text, and the audit/label CSVs.

**To reproduce:**
- Dataset: `mandarjoshi/trivia_qa`, config `unfiltered.nocontext`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`
  (Hugging Face), loaded by `gvr.data` at that pinned revision.
- BM25 index: Pyserini 1.6.0 prebuilt `wikipedia-dpr-100w`
  (`lucene-index.wikipedia-dpr-100w.20210120.d1b9e6`, md5 `7b58c08da992b2ea7e96667f0b176651`), downloaded by Pyserini and
  verified by md5.
- Model: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`, NF4 (see `configs/config.yaml`).

## Differences from the executed code

The code that produced the reported results is the private original; this public copy differs only in:

- `src/gvr/paths.py`: falls back to `configs/freeze_manifest.public.json` when `configs/freeze_manifest.json` is absent
  (release-only). Executed original sha256: `3b1e09710aabe688ea584967b94f45b74b11f8e97089e95042d0c6c46e24a42f`.
- `configs/freeze_manifest.public.json` replaces the private `configs/freeze_manifest.json` (sha256 `927e81508f5a273c8dc70846523f82c4a1851e81e6169d99e67317d70cdc647a`): local
  absolute paths and quoted development examples redacted; all other fields and hashes unchanged.
- `tests/test_parsing.py`: three parser cases taken from development model outputs replaced by synthetic strings of the
  same form. Executed original sha256: `49f2efad2dafe7061c71bae25171b752ca969f9c0ad5487bd7e29006e5f2df24`.
- `scripts/em_official_check.py`: one literal TriviaQA alias replaced by a synthetic annotated alias of the same form.
  Executed original sha256: `90f9182c6890d80f1fff95514e6d3d8aa3f1c54c42161afdc1351fd9fbc0db5d`.
