#!/usr/bin/env bash
set -euo pipefail
BASE=/root/nailfold/artifacts/experiments/unified_protocol_20260908
OUT=/root/nailfold/artifacts/experiments/downstream_20260909
mkdir -p "$OUT"
echo "WAITING_FOR_UNIFIED" > "$OUT/STATUS"
while true; do
  pgrep -f '/tmp/unified_protocol_20260908.py' >/dev/null || true
  ready=1
  for f in frozen_cls frozen_cls_patch lora_r4 lora_r8 lora_r16; do
    test -s "$BASE/${f}_results.json" || ready=0
  done
  if [ "$ready" = 1 ]; then break; fi
  sleep 60
done
echo "UNIFIED_DONE" > "$OUT/STATUS"
# Pick the best fixed-threshold development configuration from the completed runs.
CFG=$(/root/miniconda3/envs/nfc/bin/python - "$BASE" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]); best=None
for p in base.glob('*_results.json'):
    d=json.loads(p.read_text()); score=d.get('mean_ba_0_5',-1)
    if best is None or score>best[0]: best=(score,p.stem.replace('_results',''))
print(best[1])
PY
)
echo "$CFG" > "$OUT/SELECTED_CONFIG"
echo "RUNNING_DOWNSTREAM" > "$OUT/STATUS"
/root/miniconda3/envs/nfc/bin/python -u /tmp/downstream_experiments_20260909.py --config "$CFG" > "$OUT/downstream.log" 2>&1
echo "RUNNING_PAPILLA_CE" > "$OUT/STATUS"
/root/miniconda3/envs/nfc/bin/python -u /tmp/papilla_weighted_ce_20260909.py --rank 8 --lr 2e-4 > "$OUT/papilla_ce.log" 2>&1
echo "DOWNSTREAM_DONE" > "$OUT/STATUS"
