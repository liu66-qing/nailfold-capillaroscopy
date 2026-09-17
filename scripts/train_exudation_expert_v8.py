"""
Exudation-only DINOv2 LoRA fine-tuning with anti-overfitting regularization.
v8 — single-field expert, proper augmentation, dropout, label smoothing, early stopping.
"""
import argparse, copy, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Config ──────────────────────────────────────────────────────────────
FIELD = "exudation"
LABEL_MAP = {"无": 0, "+": 1, "++": 1, "+++": 1, "[无]": 0}  # binary: 无 vs 有
NUM_CLASSES = 2
FOLDS = 5
PATIENCE = 5

# ── Dirty-label fixes (skip bad samples) ──
SKIP_RULES = {
    # blood_color field had +++ leak; not relevant here but pattern for future
}

# ── LoRA ────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.b(self.drop(self.a(x))) * self.scale

# ── Dataset ─────────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row.image_path).convert("RGB")
        img = self.transform(img)
        raw = row[FIELD]
        if pd.isna(raw) or raw not in LABEL_MAP:
            label = -1
        else:
            label = LABEL_MAP[raw]
        return img, label, row.exam_case_id

# ── Model ───────────────────────────────────────────────────────────────
class ExudationExpert(nn.Module):
    def __init__(self, backbone, head_dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(768, NUM_CLASSES),
        )

    def forward(self, x):
        z = self.backbone(x)  # (B, 768) CLS token
        return self.head(z)

# ── Predict with configurable aggregation ───────────────────────────────
def predict_cases(model, loader, agg="mean", k=2):
    """
    agg: 'mean' | 'topk'
    For 'topk', uses top-k frames by positive-class probability.
    """
    case_logits = {}
    case_probs = {}
    model.eval()
    with torch.inference_mode():
        for imgs, labels, case_ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs.cuda(non_blocking=True))
            probs = torch.softmax(logits.float(), dim=1)  # (B, 2)
            for cid, lg, pb in zip(case_ids, logits.float().cpu().numpy(),
                                   probs.cpu().numpy()):
                case_logits.setdefault(cid, []).append(lg)
                case_probs.setdefault(cid, []).append(pb[1])  # P(positive)

    preds = {}
    for cid in case_logits:
        if agg == "topk":
            p_sorted = sorted(case_probs[cid], reverse=True)
            top_mean = np.mean(p_sorted[:k])
            preds[cid] = int(top_mean > 0.5)
        else:  # mean logits
            mean_logit = np.mean(case_logits[cid], axis=0)
            preds[cid] = int(np.argmax(mean_logit))
    return preds

# ── Scoring ─────────────────────────────────────────────────────────────
def score_fold(preds, cases, fold):
    sub = cases[cases.development_fold.astype(int) == fold]
    y_true, y_pred = [], []
    for _, row in sub.iterrows():
        raw = row[FIELD]
        if pd.isna(raw) or raw not in LABEL_MAP:
            continue
        cid = row.exam_case_id
        if cid not in preds:
            continue
        y_true.append(LABEL_MAP[raw])
        y_pred.append(preds[cid])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return ba, rec

# ── Transforms ──────────────────────────────────────────────────────────
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf
# PLACEHOLDER_MAIN
