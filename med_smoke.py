import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
p='/root/autodl-tmp/vlm_direct_20260827/models/medgemma_4b'
print('loading', flush=True)
proc=AutoProcessor.from_pretrained(p, local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(p, local_files_only=True, device_map='auto', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print('loaded', type(model).__name__, flush=True)
