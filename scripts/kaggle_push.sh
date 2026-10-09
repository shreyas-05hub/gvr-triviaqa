#!/usr/bin/env bash
# Usage: scripts/kaggle_push.sh <stage> <timeout_s> [confirm]
# Builds runner_stage.py with STAGE/CONFIRM set, logs quota before, pushes, waits, pulls output, logs quota after.
set -euo pipefail
cd "$(dirname "$0")/.."
K=~/.venvs/kaggle-cli/bin/kaggle
STAGE=$1; TMO=$2; CONF=${3:-}
if [ "$STAGE" = test_chunk ]; then
  [ -n "${CHUNK:-}" ] || { echo "refused: test_chunk needs CHUNK=c1|c2|c3"; exit 1; }
  python3 -c "import json,sys;a=json.load(open('configs/freeze_manifest.json')).get('test_chunk_approvals',[]);sys.exit(0 if any(x['chunk']=='$CHUNK' for x in a) else 1)" \
    || { echo "refused: chunk $CHUNK not approved in freeze_manifest.test_chunk_approvals"; exit 1; }
fi
case "$STAGE" in calibration|test) echo "refused: $STAGE needs explicit user approval (calib-01 is the only calibration run; rerun requires PI approval, logged in configs/calibration_runs.json)"; exit 1;; esac
CONFIRM=False; [ "$CONF" = confirm ] && CONFIRM=True
EXP=$(python3 -c "import json;print(json.load(open('configs/kaggle_dataset.json'))['version'])")
CUR=$(~/.venvs/kaggle-cli/bin/python -c "from kaggle.api.kaggle_api_extended import KaggleApi as A;import json;a=A();a.authenticate();print(json.loads(a.dataset_status('shreyas572/gvr-repo',format='json(current_version_number)'))['current_version_number'])")
[ "$EXP" = "$CUR" ] || { echo "ABORT: local expects dataset v$EXP but Kaggle current is v$CUR"; exit 1; }
sed -e "s/^STAGE = \"env\"/STAGE = \"$STAGE\"/" -e "s/^CONFIRM_DOWNLOAD = False/CONFIRM_DOWNLOAD = $CONFIRM/" \
    -e "s/^EXPECTED_DATASET_VERSION = None/EXPECTED_DATASET_VERSION = $EXP/" \
    -e "s/^CHUNK = None /CHUNK = \"${CHUNK:-}\" /" \
    notebooks/src/runner.py > kaggle/kernel/runner_stage.py
grep -q "^EXPECTED_DATASET_VERSION = $EXP" kaggle/kernel/runner_stage.py
grep -q "^STAGE = \"$STAGE\"" kaggle/kernel/runner_stage.py
mkdir -p runs
q() { $K quota 2>&1 | tr '\n' ' ' | sed 's/"/\\"/g'; }
echo "{\"step\":\"kaggle_quota_before\",\"stage\":\"$STAGE\",\"t\":$(date +%s),\"quota\":\"$(q)\"}" >> runs/setup_timing.jsonl
$K kernels push -p kaggle/kernel -t "$TMO"
REF=$(python3 -c "import json;print(json.load(open('kaggle/kernel/kernel-metadata.json'))['id'])")
while true; do
  S=$($K kernels status "$REF" 2>&1); echo "$(date +%T) $S"
  echo "$S" | grep -qiE "complete|error|cancel" && break; sleep 120
done
OUT=kaggle/output/$STAGE-$(date +%Y%m%dT%H%M%S); mkdir -p "$OUT"
$K kernels output "$REF" -p "$OUT"
echo "{\"step\":\"kaggle_quota_after\",\"stage\":\"$STAGE\",\"t\":$(date +%s),\"quota\":\"$(q)\",\"output_dir\":\"$OUT\"}" >> runs/setup_timing.jsonl
echo "output in $OUT"
