#!/usr/bin/env bash
# Versions ONLY shreyas572/gvr-repo (never gvr-repo1); records the version number in configs/kaggle_dataset.json.
set -euo pipefail
cd "$(dirname "$0")/.."
K=~/.venvs/kaggle-cli/bin/kaggle; P=~/.venvs/kaggle-cli/bin/python
DS=shreyas572/gvr-repo
st() { $P -c "from kaggle.api.kaggle_api_extended import KaggleApi as A;a=A();a.authenticate();import json;print(json.loads(a.dataset_status('$DS',format='json(status,current_version_number)'))['$1'])"; }
CUR=$(st current_version_number); NEXT=$((CUR + 1))
printf '{\n  "dataset": "%s",\n  "version": %d,\n  "note": "written before upload; verified == current_version_number after upload"\n}\n' "$DS" "$NEXT" > configs/kaggle_dataset.json
INCLUDE_RUNS="${INCLUDE_RUNS:-}" scripts/package_repo.sh
rm -f kaggle/dataset/*.zip && cp gvr_repo.zip kaggle/dataset/
$K datasets version -p kaggle/dataset -m "${1:-sync}"
for i in $(seq 1 30); do [ "$(st status)" = ready ] && break; sleep 10; done
GOT=$(st current_version_number)
[ "$GOT" = "$NEXT" ] || { echo "VERSION MISMATCH expected $NEXT got $GOT"; exit 1; }
echo "dataset $DS now at version $GOT (status $(st status))"
