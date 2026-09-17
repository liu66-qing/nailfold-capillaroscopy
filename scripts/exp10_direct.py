"""EXP-10: Direct total_score regression + overall_assessment classification.

Skips per-field classification. Two heads:
  1. Regression: predict total_score (0-20 range)
  2. Classification: predict overall_assessment (5 classes)

Also tries a hybrid variant that keeps per-field heads + total_score auxiliary loss.
"""
import argparse, copy, json, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

SEEDS = [17, 42, 123, 456, 789]
OVERALL_THRESHOLDS = [(1.0, "正常"), (2.0, "大致正常"), (4.0, "轻度异常"), (8.0, "中度异常")]
SEV_MAP = {"正常": 0, "大致正常": 1, "轻度异常": 2, "中度异常": 3, "重度异常": 4}

def score_to_assessment(total):
    for t, l in OVERALL_THRESHOLDS:
        if total < t: return l
    return "重度异常"

class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden); self.u = nn.Linear(dim, hidden); self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), dim=1)
        return (a * x).sum(1)

class DirectModel(nn.Module):
    def __init__(self, backbone, bottleneck_dim=64):
        super().__init__()
        self.b = backbone
        self.patch_pool = GatedPool()
        self.bottleneck = nn.Sequential(nn.Linear(1536, bottleneck_dim), nn.ReLU(), nn.Dropout(0.3))
        self.score_head = nn.Linear(bottleneck_dim, 1)
        self.assess_head = nn.Linear(bottleneck_dim, 5)

    def forward(self, x):
        z = self.b.forward_features(x)
        z = torch.cat([z[:, 0], self.patch_pool(z[:, 1:])], 1)
        z = self.bottleneck(z)
        return self.score_head(z).squeeze(-1), self.assess_head(z)

class Cases(Dataset):
    def __init__(self, ids, frames, labels, root, transform):
        self.ids = list(ids)
        self.groups = {c: g for c, g in frames[frames.exam_case_id.isin(ids)].groupby('exam_case_id')}
        self.labels = labels.set_index('exam_case_id')
        self.root = Path(root); self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        cid = self.ids[i]; rows = self.groups[cid]
        imgs = torch.stack([self.transform(Image.open(self.root / r.image_path).convert('RGB'))
                            for _, r in rows.iterrows()])
        row = self.labels.loc[cid]
        ts = float(row['total_score']) if pd.notna(row.get('total_score')) else -1.0
        oa_str = str(row['overall_assessment']) if pd.notna(row.get('overall_assessment')) else ""
        oa = SEV_MAP.get(oa_str, -1)
        return imgs, ts, oa, cid

def collate(batch):
    return ([x[0] for x in batch],
            torch.tensor([x[1] for x in batch]),
            torch.tensor([x[2] for x in batch]),
            [x[3] for x in batch])

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_model(wp, device):
    b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True), strict=False)
    for p in b.parameters(): p.requires_grad = False
    return DirectModel(b).to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=25)
    args = ap.parse_args()

    out = Path('/root/nailfold/artifacts/experiments/exp10_direct')
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                         dtype={'exam_case_id': str})
    dev = labels[labels.evaluation_role == 'development'].copy()
    dev['development_fold'] = dev['development_fold'].astype(int)

    frames = pd.read_csv('/root/nailfold/artifacts/features/image_index.csv', dtype={'exam_case_id': str})
    frames = frames[frames.exam_case_id.isin(set(dev.exam_case_id))]

    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    device = f'cuda:{args.gpu}'
    wp = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'

    # Class weights for overall_assessment (5 classes)
    oa_counts = dev['overall_assessment'].map(SEV_MAP).dropna().value_counts().sort_index()
    oa_total = oa_counts.sum()
    oa_weights = torch.ones(5)
    for ci in range(5):
        c = oa_counts.get(ci, 1)
        oa_weights[ci] = min(math.sqrt(oa_total / (5 * max(c, 1))), 5.0)

    oof_preds = {}  # case_id → {'score': float, 'assess': int}

    for seed in SEEDS:
        seed_all(seed)
        print(f"\nSEED {seed}", flush=True)
        for fold in range(5):
            valfold = (fold + 1) % 5
            trainids = dev.loc[~dev.development_fold.isin([fold, valfold]), 'exam_case_id'].tolist()
            valids = dev.loc[dev.development_fold.eq(valfold), 'exam_case_id'].tolist()
            testids = dev.loc[dev.development_fold.eq(fold), 'exam_case_id'].tolist()

            train_dl = DataLoader(Cases(trainids, frames, dev, '/root/nailfold/data', tr),
                                  batch_size=4, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
            val_dl = DataLoader(Cases(valids, frames, dev, '/root/nailfold/data', ev),
                                batch_size=4, collate_fn=collate, num_workers=2)
            test_dl = DataLoader(Cases(testids, frames, dev, '/root/nailfold/data', ev),
                                 batch_size=4, collate_fn=collate, num_workers=2)

            model = build_model(wp, device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, weight_decay=0.05)
            sched = CosineAnnealingLR(opt, T_max=args.epochs)

            best = (999, None, 0)  # minimize val MAE
            for epoch in range(1, args.epochs + 1):
                # Train
                model.train()
                for imgs_list, ts_batch, oa_batch, _ in train_dl:
                    loss = 0.; nc = 0
                    for imgs, ts, oa in zip(imgs_list, ts_batch, oa_batch):
                        if ts < 0: continue
                        imgs = imgs.to(device)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            score_out, assess_out = model(imgs)
                        score_out = score_out.float()
                        assess_out = assess_out.float()
                        # Score: MSE on case-mean (normalize to 0-1 range, max~20)
                        case_score = score_out.mean()
                        score_loss = ((case_score - ts.to(device) / 20.0) ** 2)
                        # Overall assessment: CE on case-mean logits
                        if oa >= 0:
                            case_logits = assess_out.mean(0, keepdim=True)
                            assess_loss = nn.functional.cross_entropy(
                                case_logits, oa.unsqueeze(0).to(device), weight=oa_weights.to(device))
                            loss += 0.5 * score_loss + 0.5 * assess_loss
                        else:
                            loss += score_loss
                        nc += 1
                    loss /= max(nc, 1)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()

                # Validate: MAE on total_score
                model.eval()
                val_preds = []; val_true = []
                with torch.inference_mode():
                    for imgs_list, ts_batch, _, ids in val_dl:
                        for imgs, ts, cid in zip(imgs_list, ts_batch, ids):
                            if ts < 0: continue
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                score_out, _ = model(imgs.to(device))
                            pred = score_out.float().mean().item() * 20.0
                            val_preds.append(pred); val_true.append(ts.item())
                val_mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_true))))
                if val_mae < best[0]:
                    best = (val_mae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, epoch)

            # Test
            model.load_state_dict(best[1])
            model.eval()
            with torch.inference_mode():
                for imgs_list, _, _, ids in test_dl:
                    for imgs, cid in zip(imgs_list, ids):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            score_out, assess_out = model(imgs.to(device))
                        ps = score_out.float().mean().item() * 20.0
                        pa = int(assess_out.float().mean(0).argmax().item())
                        oof_preds.setdefault(cid, []).append({'score': ps, 'assess': pa})

            print(f"  f={fold} ep={best[2]} val_MAE={best[0]:.3f}", flush=True)
            del model, opt, sched; torch.cuda.empty_cache()

    # Ensemble
    idx = dev.set_index('exam_case_id')
    results_rows = []
    for cid, entries in oof_preds.items():
        if cid not in idx.index: continue
        true_ts = float(idx.at[cid, 'total_score']) if pd.notna(idx.at[cid, 'total_score']) else None
        true_oa = str(idx.at[cid, 'overall_assessment']) if pd.notna(idx.at[cid, 'overall_assessment']) else None

        # Ensemble: median score, majority vote for assessment
        scores = [e['score'] for e in entries]
        assesses = [e['assess'] for e in entries]
        pred_ts = float(np.median(scores))
        pred_oa_idx = Counter(assesses).most_common(1)[0][0]
        pred_oa_from_score = score_to_assessment(pred_ts)

        results_rows.append({
            'case': cid, 'true_ts': true_ts, 'pred_ts': pred_ts,
            'pred_oa_from_score': pred_oa_from_score,
            'pred_oa_classify': list(SEV_MAP.keys())[pred_oa_idx],
            'true_oa': true_oa,
        })

    rdf = pd.DataFrame(results_rows)
    valid = rdf[rdf.true_ts.notna()].copy()

    # Metrics
    mae = float(np.mean(np.abs(valid.true_ts - valid.pred_ts)))
    sp, _ = spearmanr(valid.true_ts, valid.pred_ts)

    true_sev = valid.true_oa.map(SEV_MAP).values
    pred_sev_score = valid.pred_oa_from_score.map(SEV_MAP).values
    pred_sev_cls = valid.pred_oa_classify.map(SEV_MAP).values

    print(f"\n{'='*60}\nEXP-10 RESULTS\n{'='*60}", flush=True)
    print(f"Total Score MAE:     {mae:.3f}", flush=True)
    print(f"Spearman rho:        {sp:.3f}", flush=True)
    print(f"Pred mean:           {valid.pred_ts.mean():.2f}", flush=True)
    print(f"True mean:           {valid.true_ts.mean():.2f}", flush=True)
    print(f"\nFrom score threshold:", flush=True)
    print(f"  Overall Acc:       {np.mean(true_sev == pred_sev_score):.3f}", flush=True)
    print(f"  Within ±1:         {np.mean(np.abs(true_sev - pred_sev_score) <= 1):.3f}", flush=True)
    print(f"  Severe underest:   {np.mean((true_sev - pred_sev_score) >= 2):.3f}", flush=True)
    print(f"\nFrom classify head:", flush=True)
    print(f"  Overall Acc:       {np.mean(true_sev == pred_sev_cls):.3f}", flush=True)
    print(f"  Within ±1:         {np.mean(np.abs(true_sev - pred_sev_cls) <= 1):.3f}", flush=True)
    print(f"  Severe underest:   {np.mean((true_sev - pred_sev_cls) >= 2):.3f}", flush=True)

    summary = {
        "total_score_mae": mae, "spearman": float(sp),
        "pred_mean": float(valid.pred_ts.mean()), "true_mean": float(valid.true_ts.mean()),
        "overall_acc_score": float(np.mean(true_sev == pred_sev_score)),
        "overall_acc_classify": float(np.mean(true_sev == pred_sev_cls)),
        "within_1_score": float(np.mean(np.abs(true_sev - pred_sev_score) <= 1)),
        "severe_underest_score": float(np.mean((true_sev - pred_sev_score) >= 2)),
    }
    (out / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    valid.to_csv(out / 'oof_predictions.csv', index=False)
    print(f"\nSaved to {out}", flush=True)

if __name__ == '__main__':
    main()