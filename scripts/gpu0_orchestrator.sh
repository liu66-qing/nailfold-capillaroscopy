#!/bin/bash
# Master orchestrator for GPU 0 — runs after EXP-1 completes
# Chain: EXP-1 (already running) → EXP-1b (ordinal) → EXP-3 (calibration) → comparison
#
# GPU 1 runs EXP-2 independently (exp2_pipeline.sh)
set -e
PYTHON=/root/miniconda3/bin/python

echo "[$(date)] GPU0 orchestrator started. Waiting for EXP-1 to finish..."

# Wait for EXP-1 to finish (check for results JSON)
while true; do
    if [ -f "/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json" ]; then
        echo "[$(date)] EXP-1 complete!"
        break
    fi
    # Also check if process died
    if ! pgrep -f "exp1_multiclass.py" > /dev/null 2>&1; then
        echo "[$(date)] EXP-1 process not found. Checking if completed..."
        if [ -f "/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json" ]; then
            echo "[$(date)] EXP-1 results found."
            break
        else
            echo "[$(date)] ERROR: EXP-1 died without producing results!"
            # Check log for errors
            tail -30 /tmp/exp1_gpu0.log
            echo "[$(date)] Continuing with remaining experiments anyway..."
            break
        fi
    fi
    sleep 60
done

# Print EXP-1 summary
echo ""
echo "=========================================="
echo "EXP-1 RESULTS SUMMARY"
echo "=========================================="
if [ -f "/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json" ]; then
    $PYTHON -c "
import json
r = json.load(open('/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json'))
print(f'Total Score MAE: {r.get(\"total_score_mae\", \"N/A\")}')
print(f'Overall Assessment Acc: {r.get(\"overall_assessment_acc\", \"N/A\")}')
for f, pf in r.get('per_field', {}).items():
    print(f'  {f:30s} BA={pf.get(\"ba\",0):.3f}  sMAE={pf.get(\"smae\",0):.3f}')
"
fi

# ── EXP-1b: Ordinal loss variant ──
echo ""
echo "=========================================="
echo "[$(date)] Starting EXP-1b (ordinal loss) on GPU 0"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 $PYTHON /tmp/exp1b_ordinal.py --gpu 0 --epochs 20 2>&1 | tee /tmp/exp1b_gpu0.log

echo ""
echo "=========================================="
echo "EXP-1b RESULTS SUMMARY"
echo "=========================================="
if [ -f "/root/nailfold/artifacts/experiments/exp1b_ordinal/exp1b_results.json" ]; then
    $PYTHON -c "
import json
r = json.load(open('/root/nailfold/artifacts/experiments/exp1b_ordinal/exp1b_results.json'))
print(f'Total Score MAE: {r.get(\"total_score_mae\", \"N/A\")}')
print(f'Overall Assessment Acc: {r.get(\"overall_assessment_acc\", \"N/A\")}')
for f, pf in r.get('per_field', {}).items():
    print(f'  {f:30s} BA={pf.get(\"ba\",0):.3f}  sMAE={pf.get(\"smae\",0):.3f}')
"
fi

# ── EXP-3: Calibration exploration ──
echo ""
echo "=========================================="
echo "[$(date)] Starting EXP-3 calibration exploration (CPU only)"
echo "=========================================="
$PYTHON /tmp/exp3_calibration.py 2>&1 | tee /tmp/exp3_calibration.log

# ── Final comparison ──
echo ""
echo "=========================================="
echo "[$(date)] Running experiment comparison"
echo "=========================================="
$PYTHON /tmp/compare_experiments.py 2>&1 | tee /tmp/comparison.log

echo ""
echo "[$(date)] GPU0 orchestrator DONE."
echo "All results in /root/nailfold/artifacts/experiments/"
