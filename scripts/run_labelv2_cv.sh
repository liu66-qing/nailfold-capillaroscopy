#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/nailfold/code_latest
for fold in 1 2 3 4; do
  output="/root/autodl-tmp/nailfold/artifacts/labelv2_dev/fold${fold}"
  mkdir -p "${output}"
  nohup env PYTHONPATH=/root/autodl-tmp/nailfold/code_latest/src \
    /root/autodl-tmp/nailfold/venv/bin/python \
    scripts/train_siglip_case_finetune.py \
    --model /root/autodl-tmp/nailfold/models/siglip2-base-patch16-naflex-ms \
    --image-root /root/autodl-tmp/nailfold/data768 \
    --files artifacts/manifest/files.csv \
    --labels artifacts/labels/multisource_confidence_v2.csv \
    --folds artifacts/manifest/stratified_folds_v3.csv \
    --evaluation-roles artifacts/manifest/locked_evaluation_v1.csv \
    --output "${output}" \
    --test-fold "${fold}" \
    --epochs 30 \
    --batch-size 4 \
    --images-per-case 4 \
    --eval-images 8 \
    --deployment-preprocess \
    > "${output}/train.log" 2>&1 </dev/null &
done
wait
