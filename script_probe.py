from pathlib import Path
p=Path('/root/nailfold/scripts/finetune_dinov2_lora_rank4_lr1e4_cv.py')
print(p.exists())
print(p.read_text()[:1200])
