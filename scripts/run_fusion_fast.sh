#!/usr/bin/env bash
set -euo pipefail
PY=/root/autodl-tmp/nailfold/venv/bin/python
SCRIPT=/root/autodl-tmp/nailfold/code_latest/scripts/train_fusion_cv.py
LABELS=/root/autodl-tmp/nailfold/code_latest/artifacts/labels/rapidocr_consensus_clean_v2.csv
FOLDS=/root/autodl-tmp/nailfold/code_latest/artifacts/manifest/stratified_folds_v2.csv
ROOT=/root/autodl-tmp/nailfold/artifacts

"$PY" "$SCRIPT" \
  --features "$ROOT/features/geometry_v1" \
  --labels "$LABELS" --folds "$FOLDS" \
  --output "$ROOT/fusion_cv_fast/geometry"
"$PY" "$SCRIPT" \
  --features "$ROOT/features/siglip2_base_naflex" \
  --labels "$LABELS" --folds "$FOLDS" \
  --output "$ROOT/fusion_cv_fast/siglip"
"$PY" "$SCRIPT" \
  --features "$ROOT/features/siglip2_base_naflex" \
  --features "$ROOT/features/geometry_v1" \
  --labels "$LABELS" --folds "$FOLDS" \
  --output "$ROOT/fusion_cv_fast/combined"
