#!/bin/bash
# Retrain all 4 LoRA routing models with reviewed (corrected) labels
# v3_detector (blood_color) is a different architecture, skip for now

set -e
PYTHON=/root/miniconda3/envs/nfc/bin/python
INDEX=/root/nailfold/artifacts/features/image_index.csv
LABELS=/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv
IMGROOT=/root/nailfold/data
WEIGHTS=/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth
OUTBASE=/root/nailfold/artifacts/experiments/retrain_reviewed
SCRIPTS=/root/nailfold/scripts

mkdir -p $OUTBASE

echo "=========================================="
echo "Retrain with reviewed labels"
echo "Labels: $LABELS"
echo "=========================================="

# 1. rank8_lr1e4 (clarity best) — rank=8, lr=1e-4, 30 epochs
echo ""
echo "[1/4] rank8_lr1e4 (clarity) — rank=8, lr=1e-4, 30ep"
echo "Start: $(date)"
$PYTHON $SCRIPTS/finetune_dinov2_lora_rank8_lr1e4_cv.py \
    --index $INDEX --roles-labels $LABELS --image-root $IMGROOT \
    --weights $WEIGHTS --output-dir $OUTBASE/rank8_lr1e4 \
    --epochs 30 --batch-size 16 \
    2>&1 | tee $OUTBASE/rank8_lr1e4.log
echo "End: $(date)"

# 2. rank4_lr1e4 (exudation best) — rank=4, lr=1e-4, 30 epochs
echo ""
echo "[2/4] rank4_lr1e4 (exudation) — rank=4, lr=1e-4, 30ep"
echo "Start: $(date)"
$PYTHON $SCRIPTS/finetune_dinov2_lora_rank4_lr1e4_cv.py \
    --index $INDEX --roles-labels $LABELS --image-root $IMGROOT \
    --weights $WEIGHTS --output-dir $OUTBASE/rank4_lr1e4 \
    --epochs 30 --batch-size 16 \
    2>&1 | tee $OUTBASE/rank4_lr1e4.log
echo "End: $(date)"

# 3. rank8 (SVP best) — rank=8, lr=2e-4, 30 epochs
echo ""
echo "[3/4] rank8 (SVP) — rank=8, lr=2e-4, 30ep"
echo "Start: $(date)"
$PYTHON $SCRIPTS/finetune_dinov2_lora_rank8_cv.py \
    --index $INDEX --roles-labels $LABELS --image-root $IMGROOT \
    --weights $WEIGHTS --output-dir $OUTBASE/rank8 \
    --epochs 30 --batch-size 16 \
    2>&1 | tee $OUTBASE/rank8.log
echo "End: $(date)"

# 4. rank16 (papilla best) — rank=16, lr=2e-4, 30 epochs
echo ""
echo "[4/4] rank16 (papilla) — rank=16, lr=2e-4, 30ep"
echo "Start: $(date)"
$PYTHON $SCRIPTS/finetune_dinov2_lora_rank16_cv.py \
    --index $INDEX --roles-labels $LABELS --image-root $IMGROOT \
    --weights $WEIGHTS --output-dir $OUTBASE/rank16 \
    --epochs 30 --batch-size 16 \
    2>&1 | tee $OUTBASE/rank16.log
echo "End: $(date)"

echo ""
echo "=========================================="
echo "All 4 models done. Evaluating..."
echo "=========================================="
