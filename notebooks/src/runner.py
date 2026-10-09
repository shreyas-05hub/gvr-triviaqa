# ---- configure ----
STAGE = "env"   # env | bw_probe | unit_main | diag | diag_bm25 | determinism | pilot | determinism | judge | report
REPO_ZIP_GLOB = "/kaggle/input/*/gvr_repo.zip"   # uploaded repo dataset
PRIOR_OUTPUTS = "/kaggle/input/gvr-outputs"      # optional: previous session's out/ dataset (runs, configs, reports)
CONFIRM_DOWNLOAD = False                         # user sets True only after approving sizes/estimates
RUN_ID = "pilot-01"
CHUNK = None                                     # test_chunk only: "c1" | "c2" | "c3"
EXPECTED_DATASET_VERSION = None                  # injected by scripts/kaggle_push.sh
# --------------------
import glob, json, os, subprocess, sys, time
W = "/kaggle/working"; REPO = "/root/gvr_repo"   # code + venv ephemeral; only out/ is persisted
LOG = f"{W}/runner_log.txt"                       # self-written log, downloaded with kernel output


class _Tee:
    def __init__(self, *s):
        self.s = s
    def write(self, x):
        for f in self.s:
            f.write(x); f.flush()
    def flush(self):
        for f in self.s:
            f.flush()


_logf = open(LOG, "a")
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)
print("runner start", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "STAGE =", STAGE)
print("/kaggle/input listing (depth<=4):")
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count(os.sep) - "/kaggle/input".count(os.sep) >= 4:
        dirs[:] = []
    print("  ", root, f"[{len(files)} files]")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # ONE GPU only
os.environ["GVR_SESSION"] = "kaggle-runner"       # only Kaggle runner sessions may read validation question text
os.environ["PYSERINI_CACHE"] = "/root/.cache/pyserini"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
SETUP = []
def sh(c, setup_step=None):
    print("$", c); t0 = time.time()
    r = subprocess.run(["bash", "-o", "pipefail", "-c", f"( {c} ) 2>&1 | tee -a {LOG}"])
    if setup_step:
        SETUP.append(dict(step=setup_step, seconds=time.time() - t0, t_end=time.time()))
    assert r.returncode == 0, c
if not os.path.exists(REPO):
    zips = glob.glob(REPO_ZIP_GLOB) + glob.glob("/kaggle/input/**/gvr_repo.zip", recursive=True)
    if zips:
        sh(f"mkdir -p {REPO} && cd {REPO} && unzip -q {zips[0]}", "unzip_repo")
    else:  # Kaggle may auto-extract uploaded zips
        cfgs = glob.glob("/kaggle/input/**/configs/config.yaml", recursive=True)
        assert cfgs, "repo not found under /kaggle/input"
        sh(f"cp -r {os.path.dirname(os.path.dirname(cfgs[0]))} {REPO}", "copy_repo")
_mounted = json.load(open(f"{REPO}/configs/kaggle_dataset.json"))["version"]
print("mounted gvr-repo dataset version:", _mounted, "| expected:", EXPECTED_DATASET_VERSION)
if EXPECTED_DATASET_VERSION is not None and _mounted != EXPECTED_DATASET_VERSION:
    raise SystemExit(f"ABORT: mounted dataset version {_mounted} != expected {EXPECTED_DATASET_VERSION}; "
                     "no install/download performed")
if os.path.exists(PRIOR_OUTPUTS):
    sh(f"cp -rn {PRIOR_OUTPUTS}/runs {PRIOR_OUTPUTS}/configs {PRIOR_OUTPUTS}/reports {REPO}/ 2>/dev/null || true")
import shutil
def find_uv():
    return shutil.which("uv") or next((p for p in ("/usr/local/bin/uv", os.path.expanduser("~/.local/bin/uv"))
                                       if os.path.exists(p)), None)
UV = find_uv()
if UV is None:
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh", "install_uv")
    UV = find_uv()
assert UV, "uv not found after install"
print("uv:", UV)
PY = f"{REPO}/.venv/bin/python"
if not os.path.exists(PY):   # uv-managed Python 3.11; Kaggle's preinstalled torch/transformers/datasets unused
    sh(f"cd {REPO} && {UV} python install 3.11 && {UV} venv --python 3.11 .venv && "
       f"{UV} pip install --python .venv -r env/requirements-gpu.lock.txt", "install_venv_from_lock")
jv = subprocess.run("java -version 2>&1", shell=True, capture_output=True, text=True).stdout
if 'version "21' not in jv:
    sh("apt-get -qq update && apt-get -qq install -y openjdk-21-jdk-headless", "install_openjdk21")
os.environ["JAVA_HOME"] = subprocess.run("dirname $(dirname $(readlink -f $(which java)))", shell=True,
                                         capture_output=True, text=True).stdout.strip()
print("JAVA_HOME", os.environ["JAVA_HOME"], subprocess.run("java -version 2>&1", shell=True, capture_output=True, text=True).stdout)
os.chdir(REPO); os.makedirs("runs", exist_ok=True)
with open("runs/setup_timing.jsonl", "a") as f:
    for s in SETUP:
        f.write(json.dumps(dict(session_stage=STAGE, **s)) + "\n")
cmds = {
    "env": f"( echo '# Stage-0 audit RE-RUN captured by runner (env stage), not the original audit session'; date -u; "
           f"{PY} notebooks/src/audit_system.py; {PY} notebooks/src/audit_pyserini_meta.py ) > configs/kaggle_audit_stage0.txt 2>&1; "
           f"{PY} scripts/write_manifests.py --role kaggle-gpu && {PY} -m pytest -q -rA 2>&1 | tee runs/pytest_env.txt && "
           f"{PY} scripts/prefetch.py --what qwen && {PY} scripts/stage4_bm25.py",   # sizes only, no downloads
    "bw_probe": f"{PY} scripts/bw_probe.py",                                         # 2 x 128 MB ranged reads
    "unit_main": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                  f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                  f"GVR_MODEL=main {PY} -m pytest -q tests/test_fixture_model.py; "     # outcomes saved even on failure
                  f"{PY} scripts/stage4_bm25.py --load --confirm-download; "
                  f"{PY} scripts/collect_checks.py"),
    "diag": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
             f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_diag.txt; "                    # full CPU suite (89)
             f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
             f"{PY} scripts/diag_rescore.py; "                                              # diagnostic only; no BM25
             f"{PY} scripts/collect_checks.py"),
    "diag_bm25": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                  f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_diag_bm25.txt; "              # all CPU tests
                  f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                  f"{PY} scripts/diag_rescore.py; "                                         # diagnostic only
                  f"OPENAI_API_KEY=unused-placeholder {PY} scripts/stage4_bm25.py --load --confirm-download; "
                  f"{PY} scripts/collect_checks.py"),
    "pilot": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
              f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_pilot.txt && "
              f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
              f"{PY} scripts/run_arms.py --run-id {RUN_ID}"),        # BM25 guarded load + smoke regression first
    "determinism": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                    f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_determinism.txt && "        # full suite; failure stops
                    f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                    f"{PY} scripts/stage4_bm25.py --load --confirm-download && "               # P0: guarded load + smoke regression
                    f"{PY} scripts/run_arms.py --arms E0 --limit 20 --run-id det-E0-1 && "      # P1
                    f"{PY} scripts/run_arms.py --arms E0 --limit 20 --run-id det-E0-2 && "      # P2 (separate process)
                    f"{PY} scripts/run_arms.py --arms E1,E3 --limit 10 --a1-from det-E0-1 --run-id det-E13-1 && "  # P3
                    f"{PY} scripts/run_arms.py --arms E1,E3 --limit 10 --a1-from det-E0-1 --run-id det-E13-2 && "  # P4
                    f"{PY} scripts/determinism.py --pair det-E0-1 det-E0-2 --pair det-E13-1 det-E13-2"),
    "judgedev": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                 f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_judgedev.txt && "
                 f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                 f"{PY} scripts/run_arms.py --ids judgedev_ids --run-id judgedev-01 && "   # Qwen process exits -> GPU freed
                 f"test -f runs/pilot-01/calls.jsonl && "
                 f"{PY} scripts/prefetch.py --what judge --confirm-download && "
                 f"{PY} scripts/judge_parse_check.py"),   # parse check on 105 PILOT rows only; no judge on judge-dev
    "judge_judgedev": (f"test -f runs/judgedev-01/blind_key.json && test -f runs/judgedev-01/calls.jsonl && "
                       f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                       f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_judge_judgedev.txt && "
                       f"{PY} scripts/prefetch.py --what judge --confirm-download && {PY} scripts/judge_judgedev.py"),
    "format_check": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                     f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_format_check.txt && "
                     f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                     f"{PY} scripts/run_arms.py --ids pilot_ids --run-id pilot-02-v2"),   # v2 prompts; BM25 checks first; no judge
    "calibration": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                    f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_calibration.txt && "
                    f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                    f"{PY} scripts/run_arms.py --ids calib_ids --arms E0 --run-id calib-01"),   # E0 ONLY; no BM25/E1/E3/judge
    "test_chunk": (f"{PY} scripts/write_manifests.py --role kaggle-gpu && "
                   f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_test_{CHUNK}.txt && "
                   f"{PY} scripts/prefetch.py --what qwen --confirm-download && "
                   f"{PY} scripts/run_test_chunk.py --chunk {CHUNK}"),   # resumable; no gold; no judge
    "judge": (f"test -f runs/{RUN_ID}/calls.jsonl && {PY} scripts/write_manifests.py --role kaggle-gpu && "
              f"{PY} -m pytest -q -rA 2>&1 | tee runs/pytest_judge.txt && "
              f"{PY} scripts/prefetch.py --what judge --confirm-download && {PY} scripts/judge.py --run-id {RUN_ID}"),
    "report": f"{PY} scripts/collect_checks.py && {PY} scripts/analyze_report.py --run-id {RUN_ID} "
              f"--checks runs/checks.json --determinism runs/determinism_summary.json",
}
if STAGE in ("unit_main", "diag", "diag_bm25", "judgedev", "judge_judgedev", "format_check", "calibration", "pilot", "determinism", "judge") and not CONFIRM_DOWNLOAD:
    raise SystemExit(f"stage {STAGE} downloads large files / uses GPU time: set CONFIRM_DOWNLOAD=True after approval")
if STAGE in ("calibration", "test"):
    raise SystemExit("calibration/test are blocked (calib-01 is the only calibration run; rerun needs PI approval)")
if STAGE == "test_chunk":
    _appr = json.load(open(f"{REPO}/configs/freeze_manifest.json")).get("test_chunk_approvals", [])
    if CHUNK not in {x["chunk"] for x in _appr}:
        raise SystemExit(f"test_chunk {CHUNK!r} not approved in freeze_manifest.test_chunk_approvals")
try:
    sh(cmds[STAGE])
finally:  # persist ONLY logs, manifests, gate.json, reports
    sh(f"mkdir -p {W}/out && cp -r runs configs reports {W}/out/ 2>/dev/null || true")
    print("Save /kaggle/working/out as a Kaggle output dataset (gvr-outputs) before ending the session.")
