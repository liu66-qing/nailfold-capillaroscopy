#!/usr/bin/env bash
set -euo pipefail
PY=/root/autodl-tmp/nailfold/venv/bin/python
SCRIPT=/root/autodl-tmp/nailfold/code_latest/scripts/train_video_temporal.py
FEATURES=/root/autodl-tmp/nailfold/artifacts/video_features/siglip2_flow_seq
LABELS=/root/autodl-tmp/nailfold/code_latest/artifacts/labels/rapidocr_consensus_clean_v2.csv
FOLDS=/root/autodl-tmp/nailfold/code_latest/artifacts/manifest/stratified_folds_v2.csv
for fold in 1 2 3 4; do
  "$PY" "$SCRIPT" --features "$FEATURES" --labels "$LABELS" --folds "$FOLDS" \
    --output "/root/autodl-tmp/nailfold/artifacts/video_temporal_v2/fold${fold}" \
    --test-fold "$fold" --epochs 80
done
