#!/usr/bin/env bash
set -euo pipefail
PY=/root/autodl-tmp/nailfold/venv/bin/python
SCRIPT=/root/autodl-tmp/nailfold/code_latest/scripts/extract_video_features.py
ROOT=/root/autodl-tmp/nailfold
for fold in 0 1 2 3 4; do
  "$PY" "$SCRIPT" --video-root "$ROOT/videos224" \
    --output "$ROOT/artifacts/video_features/siglip2_static_ft_fold${fold}" \
    --model "$ROOT/models/siglip2-base-patch16-naflex-ms" \
    --checkpoint "$ROOT/artifacts/siglip_case_ft/fold${fold}/best.pt" \
    --sample-frames 32 --max-num-patches 256
done
"$PY" "$SCRIPT" --video-root "$ROOT/videos224" \
  --output "$ROOT/artifacts/video_features/siglip2_static_ft_full15" \
  --model "$ROOT/models/siglip2-base-patch16-naflex-ms" \
  --checkpoint "$ROOT/artifacts/siglip_case_ft/full15/full.pt" \
  --sample-frames 32 --max-num-patches 256
