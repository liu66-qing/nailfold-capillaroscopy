from pathlib import Path
s=Path('/root/nailfold/scripts/finetune_dinov2_lora_rank4_lr1e4_cv.py').read_text()
print(s[-3500:])
