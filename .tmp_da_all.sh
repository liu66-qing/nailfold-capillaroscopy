set -e
P=/c/Users/liujunqing/anaconda3/envs/pytorch_gpu/python.exe
for e in dinov2b dinov2l retfound medsiglip; do
  echo "=== $e $(date +%H:%M)"
  PYTHONIOENCODING=utf-8 $P scripts/adapt_encoder_domain_external.py \
     --encoder $e --epochs 99 --max-steps 6000 --batch-size 8 \
     --unfreeze-blocks 4 2>&1 | grep -E "step[0-9]*000 loss|last_100|runtime_minutes|Error|Traceback" | tail -4
done
