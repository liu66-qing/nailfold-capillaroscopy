"""EXP-7: Backbone unfreeze — last K layers of DINOv2.

Runs on top of EXP-6 best skeleton config (bottleneck + score-weighted loss).
Searches K={2,4,6} × backbone_lr={5e-6, 1e-5} = 6 configs.
Uses stronger augmentation to compensate for overfitting risk.
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
import torchvision.transforms as T

# ── Reuse field definitions from EXP-6 ──
FIELDS = [
    "clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla",
    "capillary_count", "crossing_ratio", "malformation_ratio",
    "flow_state", "rbc_aggregation", "microthrombus", "hemorrhage",
]

MAP = {
    "clarity":        {"清晰": 0, "不清": 1, "模糊": 2},
    "blood_color":    {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 2},
    "exudation":      {"无": 0, "+": 1, "++": 2, "+++": 2},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3},
    "papilla":        {"平坦": 0, "浅波纹状": 1, "波纹状": 2},
    "capillary_count": {"<1": 0, "1--2": 0, "3--4": 1, "5--6": 2, ">=7": 3},
    "crossing_ratio":  {"<=30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 2,
                        "[<30%]": 0, "10--30%": 0},
    "malformation_ratio": {"<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3,
                           "[<10%]": 0},
    "flow_state":     {"线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3,
                       "粒缓流": 4, "粒摆流": 4, "全停": 4,
                       "[线流、线粒流]": 0},
    "rbc_aggregation": {"无": 0, "轻度": 1, "中度": 2, "重度": 3,
                        "[无]": 0, "1--30": -1},
    "microthrombus":  {"无": 0, "1--2": 1, ">2": 2,
                       "[未见]": 0, "个/min": -1},
    "hemorrhage":     {"无": 0, "1--2": 1, "管袢/一指甲襞": -1},
}

N_CLASSES = {f: max(v for v in m.values() if v >= 0) + 1 for f, m in MAP.items()}

SCORE_SPREAD = {
    "clarity": 0.6, "blood_color": 0.8, "exudation": 3.6,
    "subpapillary_venous_plexus": 1.0, "papilla": 1.6,
    "capillary_count": 6.0, "crossing_ratio": 1.2,
    "malformation_ratio": 1.2, "flow_state": 6.0,
    "rbc_aggregation": 3.0, "microthrombus": 4.8, "hemorrhage": 0.8,
}
_SM = np.mean(list(SCORE_SPREAD.values()))
FIELD_LOSS_WEIGHT = {f: SCORE_SPREAD[f] / _SM for f in FIELDS}

IDX_TO_LABEL = {}
for f, m in MAP.items():
    rev = {}
    for label, idx in m.items():
        if idx >= 0 and idx not in rev: rev[idx] = label
    IDX_TO_LABEL[f] = rev

SCORE_MAP = {}
FIXED_FIELDS = {"vasomotion": 0.0, "wbc_count": 0.0, "sweat_duct": 0.0}
MEASUREMENT_BASELINE_SCORES = {}
SEEDS = [17, 42, 123, 456, 789]
OVERALL_THRESHOLDS = [(1.0, "正常"), (2.0, "大致正常"), (4.0, "轻度异常"), (8.0, "中度异常")]

def score_to_assessment(total):
    for t, l in OVERALL_THRESHOLDS:
        if total < t: return l
    return "重度异常"

def pred_to_score(f, ci):
    return SCORE_MAP.get(f, {}).get(ci, 0.0)

def load_score_rules(path):
    rules = json.load(open(path))
    for f in FIELDS:
        fr = rules["field_rules"].get(f)
        if fr is None: continue
        SCORE_MAP[f] = {}
        for ci, label in IDX_TO_LABEL[f].items():
            if fr["type"] == "categorical_lookup":
                SCORE_MAP[f][ci] = fr["mapping"].get(label, 0.0)
    SCORE_MAP["capillary_count"][0] = 5.0
    SCORE_MAP["exudation"][2] = (23*2.4+4*3.6)/27
    SCORE_MAP["crossing_ratio"][2] = (12*0.4+5*1.2)/17
    SCORE_MAP["flow_state"][4] = (13*1.6+2*4.0+1*6.0)/16
    SCORE_MAP["blood_color"][2] = (79*0.4+2*0.8)/81
    return rules

class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden); self.u = nn.Linear(dim, hidden); self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), dim=1)
        return (a * x).sum(1)

class Model(nn.Module):
    def __init__(self, backbone, bottleneck_dim=64):
        super().__init__()
        self.b = backbone
        self.patch_pool = GatedPool()
        self.bottleneck = nn.Sequential(
            nn.Linear(1536, bottleneck_dim), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleDict({f: nn.Linear(bottleneck_dim, N_CLASSES[f]) for f in FIELDS})

    def forward(self, x):
        z = self.b.forward_features(x)
        z = torch.cat([z[:, 0], self.patch_pool(z[:, 1:])], 1)
        z = self.bottleneck(z)
        return {f: h(z) for f, h in self.heads.items()}

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
        y = torch.tensor([MAP[f].get(str(row[f]), -1) if pd.notna(row[f]) else -1 for f in FIELDS])
        return imgs, y, cid

def collate(batch):
    return [x[0] for x in batch], torch.stack([x[1] for x in batch]), [x[2] for x in batch]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def compute_class_weights(labels, cap=5.0):
    weights = {}
    for f in FIELDS:
        vals = labels[f].dropna().map(MAP[f]); vals = vals[vals >= 0]
        counts = vals.value_counts().sort_index()
        nc = N_CLASSES[f]; w = torch.ones(nc); total = len(vals)
        for ci in range(nc):
            c = counts.get(ci, 1)
            w[ci] = min(math.sqrt(total / (nc * max(c, 1))), cap)
        weights[f] = w
    return weights

def build_model(weights_path, device, unfreeze_k=0):
    b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=False)
    # Freeze all, then selectively unfreeze
    for p in b.parameters(): p.requires_grad = False
    if unfreeze_k > 0:
        n_blocks = len(b.blocks)
        for i in range(max(0, n_blocks - unfreeze_k), n_blocks):
            for p in b.blocks[i].parameters(): p.requires_grad = True
        for p in b.norm.parameters(): p.requires_grad = True
    model = Model(b)
    return model.to(device)

def train_epoch(model, loader, opt, device, class_weights):
    model.train()
    for imgs_list, ys, _ in loader:
        loss = 0.; total_w = 0.
        for imgs, y in zip(imgs_list, ys):
            imgs = imgs.to(device); y = y.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(imgs)
            one = 0.; wsum = 0.
            for i, f in enumerate(FIELDS):
                if y[i] < 0: continue
                fw = FIELD_LOSS_WEIGHT[f]
                w = class_weights[f].to(device)
                logits_f = out[f].float()
                case_logits = logits_f.mean(0, keepdim=True)
                case_ce = nn.functional.cross_entropy(case_logits, y[i:i+1], weight=w)
                target = y[i].expand(len(imgs))
                frame_ce = nn.functional.cross_entropy(logits_f, target, weight=w)
                one += fw * (0.75 * case_ce + 0.25 * frame_ce)
                wsum += fw
            if wsum > 0: loss += one / wsum; total_w += 1
        loss /= max(total_w, 1)
        opt.zero_grad(); loss.backward(); opt.step()

def infer(model, loader, device):
    cases = {f: {} for f in FIELDS}
    model.eval()
    with torch.inference_mode():
        for imgs_list, _, ids in loader:
            for imgs, cid in zip(imgs_list, ids):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(imgs.to(device))
                for f in FIELDS:
                    cases[f][cid] = out[f].float().cpu().mean(0).numpy()
    return cases

def objective(pred, ids, label_index):
    vals = []
    for f in FIELDS:
        use = [c for c in ids if c in pred[f] and pd.notna(label_index.at[c, f])]
        if not use: continue
        y = np.array([MAP[f].get(str(label_index.at[c, f]), -1) for c in use])
        mask = y >= 0; y = y[mask]; use = [u for u, m in zip(use, mask) if m]
        if len(set(y)) < 2: continue
        p = np.array([int(np.argmax(pred[f][c])) for c in use])
        vals.append(balanced_accuracy_score(y, p))
    return float(np.mean(vals)) if vals else 0.0

def compute_total_score(fp):
    total = sum(pred_to_score(f, fp[f]) for f in FIELDS if f in fp)
    total += sum(FIXED_FIELDS.values())
    total += sum(MEASUREMENT_BASELINE_SCORES.values())
    return total

def evaluate_full(oof_classes, labels_idx):
    results = {}
    for f in FIELDS:
        cases = [c for c in oof_classes[f] if c in labels_idx.index and pd.notna(labels_idx.at[c, f])]
        if not cases: continue
        y = np.array([MAP[f].get(str(labels_idx.at[c, f]), -1) for c in cases])
        p = np.array([oof_classes[f][c] for c in cases])
        mask = y >= 0; y, p = y[mask], p[mask]
        if len(y) == 0: continue
        ba = balanced_accuracy_score(y, p) if len(set(y)) >= 2 else float(np.mean(y == p))
        s_true = np.array([pred_to_score(f, int(yt)) for yt in y])
        s_pred = np.array([pred_to_score(f, int(yp)) for yp in p])
        results[f] = {"ba": ba, "smae": float(np.mean(np.abs(s_true - s_pred)))}

    ts_cases = [c for c in labels_idx.index
                if pd.notna(labels_idx.at[c, "total_score"])
                and all(c in oof_classes.get(f, {}) for f in FIELDS)]
    if ts_cases:
        true_ts = np.array([float(labels_idx.at[c, "total_score"]) for c in ts_cases])
        pred_ts = np.array([compute_total_score({f: oof_classes[f][c] for f in FIELDS}) for c in ts_cases])
        results["total_score_mae"] = float(np.mean(np.abs(true_ts - pred_ts)))
        results["pred_mean"] = float(pred_ts.mean())
        results["true_mean"] = float(true_ts.mean())
        sp, _ = spearmanr(true_ts, pred_ts)
        results["spearman"] = float(sp) if not np.isnan(sp) else 0.0
        sev_map = {"正常": 0, "大致正常": 1, "轻度异常": 2, "中度异常": 3, "重度异常": 4}
        oa_cases = [c for c in ts_cases if pd.notna(labels_idx.at[c, "overall_assessment"])]
        if oa_cases:
            true_oa = [str(labels_idx.at[c, "overall_assessment"]) for c in oa_cases]
            pred_oa = [score_to_assessment(compute_total_score({f: oof_classes[f][c] for f in FIELDS})) for c in oa_cases]
            true_s = np.array([sev_map.get(t, -1) for t in true_oa])
            pred_s = np.array([sev_map.get(p, -1) for p in pred_oa])
            results["overall_acc"] = float(np.mean(true_s == pred_s))
            results["within_1"] = float(np.mean(np.abs(true_s - pred_s) <= 1))
            results["severe_underest"] = float(np.mean((true_s - pred_s) >= 2))
    return results

# Search configs: K × backbone_lr
SEARCH = [
    {"K": 2, "bb_lr": 5e-6},
    {"K": 2, "bb_lr": 1e-5},
    {"K": 4, "bb_lr": 5e-6},
    {"K": 4, "bb_lr": 1e-5},
    {"K": 6, "bb_lr": 1e-5},
    {"K": 6, "bb_lr": 2e-5},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()

    labels = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                         dtype={'exam_case_id': str})
    dev = labels[labels.evaluation_role == 'development'].copy()
    dev['development_fold'] = dev['development_fold'].astype(int)

    score_rules = load_score_rules('/root/nailfold/artifacts/labels/score_rules_v3.json')
    global MEASUREMENT_BASELINE_SCORES
    idx = dev.set_index('exam_case_id')
    for mf in ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]:
        vals = pd.to_numeric(idx[mf], errors='coerce').dropna()
        median_val = float(vals.median()) if len(vals) > 0 else 0
        fr = score_rules["field_rules"].get(mf)
        if fr and fr["type"] == "numeric_tree":
            node = fr["tree"]
            while "threshold" in node:
                node = node["left"] if median_val <= node["threshold"] else node["right"]
            MEASUREMENT_BASELINE_SCORES[mf] = node["value"]
        else:
            MEASUREMENT_BASELINE_SCORES[mf] = 0.0

    frames = pd.read_csv('/root/nailfold/artifacts/features/image_index.csv', dtype={'exam_case_id': str})
    frames = frames[frames.exam_case_id.isin(set(dev.exam_case_id))]

    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    # Stronger augmentation for fine-tuning
    base_tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0.2, hflip=0.5, vflip=0.5)
    ev = timm.data.create_transform(**dc, is_training=False)
    # Add random affine + erasing
    tr = T.Compose(list(base_tr.transforms) + [
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.RandomErasing(p=0.1),
    ])

    device = f'cuda:{args.gpu}'
    class_weights = compute_class_weights(dev)
    wp = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'

    all_results = []

    for si, scfg in enumerate(SEARCH):
        K, bb_lr = scfg["K"], scfg["bb_lr"]
        tag = f"K{K}_lr{bb_lr}"
        print(f"\n{'='*60}\nEXP-7 [{si+1}/{len(SEARCH)}] K={K} bb_lr={bb_lr}\n{'='*60}", flush=True)

        out = Path(f'/root/nailfold/artifacts/experiments/exp7_{tag}')
        out.mkdir(parents=True, exist_ok=True)

        oof_logits = {f: {} for f in FIELDS}

        for seed in SEEDS:
            seed_all(seed)
            print(f"  SEED {seed}", flush=True)
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

                model = build_model(wp, device, unfreeze_k=K)
                # Separate param groups: backbone (low lr) vs head (high lr)
                backbone_params = [p for n, p in model.named_parameters()
                                   if p.requires_grad and n.startswith('b.')]
                head_params = [p for n, p in model.named_parameters()
                               if p.requires_grad and not n.startswith('b.')]
                opt = torch.optim.AdamW([
                    {"params": backbone_params, "lr": bb_lr},
                    {"params": head_params, "lr": 1e-3},
                ], weight_decay=0.1)
                sched = CosineAnnealingLR(opt, T_max=args.epochs)

                best = (-1, None, 0)
                for epoch in range(1, args.epochs + 1):
                    train_epoch(model, train_dl, opt, device, class_weights)
                    sched.step()
                    vp = infer(model, val_dl, device)
                    vp_cls = {f: {c: int(np.argmax(v)) for c, v in vp[f].items()} for f in FIELDS}
                    obj = objective(vp_cls, valids, dev.set_index('exam_case_id'))
                    if obj > best[0]:
                        best = (obj, {k: v.cpu().clone() for k, v in model.state_dict().items()}, epoch)

                model.load_state_dict(best[1])
                tp = infer(model, test_dl, device)
                for f in FIELDS:
                    for cid, v in tp[f].items():
                        oof_logits[f].setdefault(cid, []).append((seed, v))
                print(f"    f={fold} ep={best[2]} val_BA={best[0]:.4f}", flush=True)
                del model, opt, sched; torch.cuda.empty_cache()

        oof_cls = {}
        for f in FIELDS:
            oof_cls[f] = {}
            for cid, entries in oof_logits[f].items():
                votes = [int(np.argmax(v)) for _, v in entries]
                oof_cls[f][cid] = Counter(votes).most_common(1)[0][0]

        results = evaluate_full(oof_cls, dev.set_index('exam_case_id'))
        results["config"] = scfg
        all_results.append(results)

        print(f"\nK={K} lr={bb_lr}:", flush=True)
        print(f"  Total Score MAE: {results.get('total_score_mae', 'N/A')}", flush=True)
        print(f"  Spearman: {results.get('spearman', 'N/A')}", flush=True)
        print(f"  Overall Acc: {results.get('overall_acc', 'N/A')}", flush=True)
        print(f"  Severe underest: {results.get('severe_underest', 'N/A')}", flush=True)
        (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

    # Summary
    print(f"\n{'='*60}\nEXP-7 SUMMARY\n{'='*60}", flush=True)
    print(f"{'Config':20s} {'MAE':>6} {'Spear':>6} {'OA_Acc':>7} {'SevU':>6}", flush=True)
    for r in all_results:
        c = r['config']
        tag = f"K={c['K']} lr={c['bb_lr']}"
        print(f"{tag:20s} {r.get('total_score_mae',99):.3f} {r.get('spearman',0):.3f} "
              f"{r.get('overall_acc',0):.3f}   {r.get('severe_underest',1):.3f}", flush=True)

    summary_path = Path('/root/nailfold/artifacts/experiments/exp7_summary.json')
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + '\n')
    print(f"\nSaved summary to {summary_path}", flush=True)

if __name__ == '__main__':
    main()