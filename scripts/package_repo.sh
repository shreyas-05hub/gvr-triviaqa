#!/usr/bin/env bash
# Zips the repo for upload as a Kaggle dataset (excludes venv, caches, runs, kaggle/, reports/, any *_LABELED* file).
# INCLUDE_RUNS="pilot-01 ..." additionally bundles those runs/<id>/ dirs (e.g. for the judge session).
# OUT=path overrides the output zip (used by tests).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-gvr_repo.zip}
rm -f "$OUT"
zip -qr "$OUT" . -x '.venv/*' '*__pycache__*' '.pytest_cache/*' 'runs/*' 'gvr_repo.zip' 'kaggle/*' 'reports/*' '*_LABELED*'
for r in ${INCLUDE_RUNS:-}; do
  [ -d "runs/$r" ] || { echo "missing runs/$r"; exit 1; }
  zip -qr "$OUT" "runs/$r" -x '*_LABELED*'
done
if unzip -Z1 "$OUT" | grep -E '^reports/|_LABELED' ; then echo "ABORT: reports/ or *_LABELED* in package"; exit 1; fi
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) runs included: ${INCLUDE_RUNS:-none}"
