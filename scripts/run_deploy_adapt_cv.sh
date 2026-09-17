#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/nailfold
PY="$ROOT/venv/bin/python"
SCRIPT="$ROOT/code_latest/scripts/train_siglip_case_finetune.py"
export PYTHONPATH="$ROOT/code_latest/src"
for fold in 1 2 3 4; do
  "$PY" "$SCRIPT" \
    --model "$ROOT/models/siglip2-base-patch16-naflex-ms" \
    --image-root "$ROOT/data768" \
    --files "$ROOT/code_latest/artifacts/manifest/files.csv" \
    --labels "$ROOT/code_latest/artifacts/labels/rapidocr_consensus_clean_v2.csv" \
    --folds "$ROOT/code_latest/artifacts/manifest/stratified_folds_v2.csv" \
    --output "$ROOT/artifacts/siglip_deploy_adapt/fold${fold}" \
    --test-fold "$fold" --epochs 1 --batch-size 4 --images-per-case 4 --eval-images 8 \
    --deployment-preprocess \
    --resume-checkpoint "$ROOT/artifacts/siglip_case_ft/fold${fold}/best.pt" \
    --encoder-lr 2e-6 --head-lr 5e-5
done
