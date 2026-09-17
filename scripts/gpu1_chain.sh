#!/bin/bash
# GPU 1: EXP-6 config 2,3 → EXP-10 (direct regression) → EXP-9 (grouped adapters)
set -e
PY=/root/miniconda3/bin/python
LOG=/root/nailfold/artifacts/experiments

echo "[GPU1] Starting EXP-6 config 2 (bn+scoreW)..." | tee -a $LOG/gpu1.log
CUDA_VISIBLE_DEVICES=1 $PY /tmp/exp6_skeleton.py --gpu 0 --config 2 --epochs 20 2>&1 | tee -a $LOG/gpu1.log

echo "[GPU1] Starting EXP-6 config 3 (bn+scoreW+mse)..." | tee -a $LOG/gpu1.log
CUDA_VISIBLE_DEVICES=1 $PY /tmp/exp6_skeleton.py --gpu 0 --config 3 --epochs 20 2>&1 | tee -a $LOG/gpu1.log

echo "[GPU1] EXP-6 c2,c3 done. Starting EXP-10 (direct regression)..." | tee -a $LOG/gpu1.log
CUDA_VISIBLE_DEVICES=1 $PY /tmp/exp10_direct.py --gpu 0 --epochs 25 2>&1 | tee -a $LOG/gpu1.log

echo "[GPU1] All done." | tee -a $LOG/gpu1.log
