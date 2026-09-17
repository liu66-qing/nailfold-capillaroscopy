#!/usr/bin/env bash
set -u
BASE=/root/nailfold/artifacts/experiments/unified_protocol_20260908
OUT=/root/nailfold/artifacts/experiments/chain_20260909
mkdir -p "$OUT"
exec > >(tee -a "$OUT/chain.log") 2>&1
echo "CHAIN_START $(date)"
while true; do
  alive=$(pgrep -f '/tmp/unified_protocol_20260908.py --gpu' || true)
  complete=$(find "$BASE" -maxdepth 1 -name '*_results.json' | wc -l)
  if [ -z "$alive" ] && [ "$complete" -ge 5 ]; then break; fi
  sleep 60
done
echo "UNIFIED_DONE $(date)"
touch "$OUT/UNIFIED_DONE"

# These stages are deliberately separate entry points. They must only consume
# the development artifacts produced above and write their own completion mark.
if [ -f /tmp/stage3_frame_attention_20260909.py ]; then
  /root/miniconda3/envs/nfc/bin/python -u /tmp/stage3_frame_attention_20260909.py --input "$BASE" --output "$OUT/stage3" --gpu 0
  touch "$OUT/STAGE3_DONE"
else
  echo "STAGE3_SCRIPT_MISSING" >&2
  touch "$OUT/STAGE3_WAITING_FOR_SCRIPT"
  exit 2
fi
if [ -f /tmp/stage4_papilla_20260909.py ]; then
  /root/miniconda3/envs/nfc/bin/python -u /tmp/stage4_papilla_20260909.py --input "$BASE" --output "$OUT/stage4" --gpu 0
  touch "$OUT/STAGE4_DONE"
else
  echo "STAGE4_SCRIPT_MISSING" >&2
  touch "$OUT/STAGE4_WAITING_FOR_SCRIPT"
  exit 2
fi
echo "CHAIN_DONE $(date)"
