#!/bin/bash
# GPU 0: EXP-6 config 0,1 → EXP-7 (backbone unfreeze)
set -e
PY=/root/miniconda3/bin/python
LOG=/root/nailfold/artifacts/experiments

echo "[GPU0] Starting EXP-6 config 0 (baseline reproduction)..." | tee -a $LOG/gpu0.log
CUDA_VISIBLE_DEVICES=0 $PY /tmp/exp6_skeleton.py --gpu 0 --config 0 --epochs 20 2>&1 | tee -a $LOG/gpu0.log

echo "[GPU0] Starting EXP-6 config 1 (bottleneck)..." | tee -a $LOG/gpu0.log
CUDA_VISIBLE_DEVICES=0 $PY /tmp/exp6_skeleton.py --gpu 0 --config 1 --epochs 20 2>&1 | tee -a $LOG/gpu0.log

echo "[GPU0] EXP-6 c0,c1 done. Starting EXP-7 (backbone unfreeze)..." | tee -a $LOG/gpu0.log
CUDA_VISIBLE_DEVICES=0 $PY /tmp/exp7_unfreeze.py --gpu 0 --epochs 20 2>&1 | tee -a $LOG/gpu0.log

echo "[GPU0] All done." | tee -a $LOG/gpu0.log
