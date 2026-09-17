import torch
import torch.nn as nn
def _get(self): return getattr(self, '_compat_tied', None) or getattr(self, '_tied_weights_keys', {}) or {}
def _set(self, value): self._compat_tied = value
nn.Module.all_tied_weights_keys = property(_get, _set)
from transformers import AutoTokenizer, AutoModel
p='/root/autodl-tmp/vlm_direct_20260827/models/internvl3_8b'
print('loading', flush=True)
AutoTokenizer.from_pretrained(p, trust_remote_code=True, local_files_only=True, use_fast=False)
model=AutoModel.from_pretrained(p, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
print('loaded', type(model).__name__, flush=True)
