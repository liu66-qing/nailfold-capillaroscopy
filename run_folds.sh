#!/bin/bash
cd "E:/甲劈微循环"
for f in 1 2 3 4; do
  PYTHONIOENCODING=utf-8 python scripts/train_det_lockedsafe.py --fold $f --epochs 25 --imgsz 768 --batch 8 --tag s768 > artifacts/experiments/fold${f}_s768.log 2>&1
done
echo ALLDONE > artifacts/experiments/.folds_done
