#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/nailfold
EXP=$ROOT/artifacts/experiments/qwen_stage3_qlora
PY=$ROOT/venv/bin/python
MODEL=$ROOT/models/Qwen3-VL-8B-Instruct-MS
MANIFEST=$ROOT/code_latest/artifacts/manifest/locked_evaluation_v1.csv
IMAGES=$ROOT/data768

run_eval() {
  local fold=$1
  if [[ -s "$EXP/fold${fold}_epoch1_eval/metrics.csv" ]]; then
    return
  fi
  "$PY" "$EXP/eval_qlora.py" \
    --model "$MODEL" \
    --adapter "$EXP/fold${fold}_epoch1/adapter" \
    --manifest "$MANIFEST" \
    --image-root "$IMAGES" \
    --output "$EXP/fold${fold}_epoch1_eval" \
    --fold "$fold" \
    > "$EXP/fold${fold}_epoch1_eval.log" 2>&1
}

# Fold 1 may have been launched before this orchestrator.
while pgrep -f 'train_qlora.py.*--fold 1' >/dev/null; do
  sleep 15
done
run_eval 1

for fold in 2 3 4; do
  if [[ ! -s "$EXP/fold${fold}_epoch1/adapter/adapter_model.safetensors" ]]; then
    "$PY" "$EXP/train_qlora.py" \
      --model "$MODEL" \
      --manifest "$MANIFEST" \
      --image-root "$IMAGES" \
      --output "$EXP/fold${fold}_epoch1" \
      --fold "$fold" \
      --max-train-cases 0 \
      --epochs 1 \
      --grad-accum 4 \
      > "$EXP/fold${fold}_epoch1.log" 2>&1
  fi
  run_eval "$fold"
done

date -Iseconds > "$EXP/REMAINING_FOLDS_COMPLETE"
