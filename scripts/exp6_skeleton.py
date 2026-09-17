"""EXP-6: Skeleton fixes — bottleneck + score-weighted loss + ordinal MSE.

4 configurations (controlled by --config):
  0: EXP-1 baseline (reproduce, sanity check)
  1: +bottleneck (1536→64→N, dropout 0.3)
  2: +bottleneck +score-weighted loss
  3: +bottleneck +score-weighted loss +MSE ordinal regression

Based on exp1_multiclass.py with minimal diffs.
"""
import argparse, copy, gzip, json, math, random
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

# ── Field definitions (identical to EXP-1) ──
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

# Score spread per field (from score_rules_v3.json)
SCORE_SPREAD = {
    "clarity": 0.6, "blood_color": 0.8, "exudation": 3.6,
    "subpapillary_venous_plexus": 1.0, "papilla": 1.6,
    "capillary_count": 6.0, "crossing_ratio": 1.2,
    "malformation_ratio": 1.2, "flow_state": 6.0,
    "rbc_aggregation": 3.0, "microthrombus": 4.8, "hemorrhage": 0.8,
}
_SPREAD_MEAN = np.mean(list(SCORE_SPREAD.values()))
FIELD_LOSS_WEIGHT = {f: SCORE_SPREAD[f] / _SPREAD_MEAN for f in FIELDS}

IDX_TO_LABEL = {}
for f, m in MAP.items():
    rev = {}
    for label, idx in m.items():
        if idx >= 0 and idx not in rev:
            rev[idx] = label
    IDX_TO_LABEL[f] = rev

SCORE_MAP = {}  # {field: {class_idx: score}}

FIXED_FIELDS = {
    "vasomotion": ("0--1", 0.0),
    "wbc_count": ("1--30", 0.0),
    "sweat_duct": ("0--2", 0.0),
}
MEASUREMENT_BASELINE_SCORES = {}
SEEDS = [17, 42, 123, 456, 789]

OVERALL_THRESHOLDS = [(1.0, "正常"), (2.0, "大致正常"), (4.0, "轻度异常"), (8.0, "中度异常")]

def score_to_assessment(total):
    for thresh, label in OVERALL_THRESHOLDS:
        if total < thresh:
            return label
    return "重度异常"

def pred_to_score(field, class_idx):
    if field in SCORE_MAP:
        return SCORE_MAP[field].get(class_idx, 0.0)
    return 0.0

def load_score_rules(path):
    rules = json.load(open(path))
    for f in FIELDS:
        fr = rules["field_rules"].get(f)
        if fr is None: continue
        SCORE_MAP[f] = {}
        for ci, label in IDX_TO_LABEL[f].items():
            if fr["type"] == "categorical_lookup":
                SCORE_MAP[f][ci] = fr["mapping"].get(label, 0.0)
            else:
                SCORE_MAP[f][ci] = 0.0
    SCORE_MAP["capillary_count"][0] = 5.0
    SCORE_MAP["exudation"][2] = (23 * 2.4 + 4 * 3.6) / 27
    SCORE_MAP["crossing_ratio"][2] = (12 * 0.4 + 5 * 1.2) / 17
    SCORE_MAP["flow_state"][4] = (13 * 1.6 + 2 * 4.0 + 1 * 6.0) / 16
    SCORE_MAP["blood_color"][2] = (79 * 0.4 + 2 * 0.8) / 81
    return rules

# ── Model ──
class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), dim=1)
        return (a * x).sum(1)

class Model(nn.Module):
    """Configurable model with optional bottleneck and MSE heads."""
    def __init__(self, backbone, use_bottleneck=False, use_mse=False, bottleneck_dim=64):
        super().__init__()
        self.b = backbone
        self.patch_pool = GatedPool()
        self.use_bottleneck = use_bottleneck
        self.use_mse = use_mse

        feat_dim = 1536
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(feat_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            head_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Dropout(0.1)
            head_dim = feat_dim

        if use_mse:
            # MSE ordinal: 1 output per field, predict normalized class index
            self.heads = nn.ModuleDict({f: nn.Linear(head_dim, 1) for f in FIELDS})
        else:
            self.heads = nn.ModuleDict({f: nn.Linear(head_dim, N_CLASSES[f]) for f in FIELDS})

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
        self.root = Path(root)
        self.transform = transform
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

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(weights, device, use_bottleneck=False, use_mse=False):
    b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=False)
    for p in b.parameters(): p.requires_grad = False
    return Model(b, use_bottleneck=use_bottleneck, use_mse=use_mse).to(device)

def compute_class_weights(labels, cap=5.0):
    weights = {}
    for f in FIELDS:
        vals = labels[f].dropna().map(MAP[f])
        vals = vals[vals >= 0]
        counts = vals.value_counts().sort_index()
        nc = N_CLASSES[f]
        w = torch.ones(nc)
        total = len(vals)
        for cls_idx in range(nc):
            c = counts.get(cls_idx, 1)
            w[cls_idx] = min(math.sqrt(total / (nc * max(c, 1))), cap)
        weights[f] = w
    return weights

def train_epoch(model, loader, opt, device, class_weights, use_score_weight, use_mse):
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
                fw = FIELD_LOSS_WEIGHT[f] if use_score_weight else 1.0

                if use_mse:
                    pred = out[f].float().squeeze(-1)  # (num_frames,)
                    # normalize target to [0, 1]
                    nc = N_CLASSES[f]
                    target_norm = y[i].float() / max(nc - 1, 1)
                    case_pred = pred.mean()
                    case_loss = (case_pred - target_norm) ** 2
                    frame_loss = ((pred - target_norm) ** 2).mean()
                    one += fw * (0.75 * case_loss + 0.25 * frame_loss)
                else:
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

def infer(model, loader, device, use_mse):
    cases = {f: {} for f in FIELDS}
    model.eval()
    with torch.inference_mode():
        for imgs_list, _, ids in loader:
            for imgs, cid in zip(imgs_list, ids):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(imgs.to(device))
                for f in FIELDS:
                    raw = out[f].float().cpu()
                    if use_mse:
                        # mean across frames, then store scalar
                        cases[f][cid] = raw.squeeze(-1).mean().item()
                    else:
                        cases[f][cid] = raw.mean(0).numpy()
    return cases

def preds_to_classes(preds, use_mse):
    """Convert raw predictions to class indices."""
    result = {}
    for f in FIELDS:
        result[f] = {}
        for cid, v in preds[f].items():
            if use_mse:
                nc = N_CLASSES[f]
                cls_idx = int(round(v * max(nc - 1, 1)))
                cls_idx = max(0, min(nc - 1, cls_idx))
                result[f][cid] = cls_idx
            else:
                result[f][cid] = int(np.argmax(v))
    return result

def objective(pred_classes, ids, label_index):
    vals = []
    for f in FIELDS:
        use = [c for c in ids if c in pred_classes[f] and pd.notna(label_index.at[c, f])]
        if not use: continue
        y = np.array([MAP[f].get(str(label_index.at[c, f]), -1) for c in use])
        mask = y >= 0; y = y[mask]; use = [u for u, m in zip(use, mask) if m]
        if len(set(y)) < 2: continue
        p = np.array([pred_classes[f][c] for c in use])
        vals.append(balanced_accuracy_score(y, p))
    return float(np.mean(vals)) if vals else 0.0

def compute_total_score(field_preds):
    total = 0.0
    for f in FIELDS:
        if f in field_preds:
            total += pred_to_score(f, field_preds[f])
    for f, (label, score) in FIXED_FIELDS.items():
        total += score
    for f, score in MEASUREMENT_BASELINE_SCORES.items():
        total += score
    return total

def evaluate_full(oof_classes, labels_idx):
    """Full evaluation: per-field BA/sMAE, total score MAE, Spearman, severity metrics."""
    results = {}

    # Per-field
    for f in FIELDS:
        cases = [c for c in oof_classes[f] if c in labels_idx.index and pd.notna(labels_idx.at[c, f])]
        if not cases: continue
        y = np.array([MAP[f].get(str(labels_idx.at[c, f]), -1) for c in cases])
        p = np.array([oof_classes[f][c] for c in cases])
        mask = y >= 0
        y, p = y[mask], p[mask]
        if len(y) == 0: continue
        ba = balanced_accuracy_score(y, p) if len(set(y)) >= 2 else float(np.mean(y == p))
        s_true = np.array([pred_to_score(f, int(yt)) for yt in y])
        s_pred = np.array([pred_to_score(f, int(yp)) for yp in p])
        smae = float(np.mean(np.abs(s_true - s_pred)))
        results[f] = {"ba": ba, "smae": smae}

    # Total score
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

        # Overall assessment
        sev_map = {"正常": 0, "大致正常": 1, "轻度异常": 2, "中度异常": 3, "重度异常": 4}
        oa_cases = [c for c in ts_cases if pd.notna(labels_idx.at[c, "overall_assessment"])]
        if oa_cases:
            true_oa = [str(labels_idx.at[c, "overall_assessment"]) for c in oa_cases]
            pred_oa = [score_to_assessment(compute_total_score({f: oof_classes[f][c] for f in FIELDS}))
                       for c in oa_cases]
            true_sev = np.array([sev_map.get(t, -1) for t in true_oa])
            pred_sev = np.array([sev_map.get(p, -1) for p in pred_oa])
            results["overall_acc"] = float(np.mean(true_sev == pred_sev))
            results["within_1"] = float(np.mean(np.abs(true_sev - pred_sev) <= 1))
            results["underest_rate"] = float(np.mean(pred_sev < true_sev))
            results["severe_underest"] = float(np.mean((true_sev - pred_sev) >= 2))
    return results

# ── Config definitions ──
CONFIGS = {
    0: {"name": "baseline",      "bottleneck": False, "score_weight": False, "mse": False},
    1: {"name": "bottleneck",    "bottleneck": True,  "score_weight": False, "mse": False},
    2: {"name": "bn+scoreW",     "bottleneck": True,  "score_weight": True,  "mse": False},
    3: {"name": "bn+scoreW+mse", "bottleneck": True,  "score_weight": True,  "mse": True},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--config', type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    print(f"EXP-6 config={args.config} ({cfg['name']})", flush=True)
    print(f"  bottleneck={cfg['bottleneck']}, score_weight={cfg['score_weight']}, mse={cfg['mse']}", flush=True)

    out = Path(f'/root/nailfold/artifacts/experiments/exp6_c{args.config}')
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                         dtype={'exam_case_id': str})
    dev = labels[labels.evaluation_role == 'development'].copy()
    dev['development_fold'] = dev['development_fold'].astype(int)

    score_rules = load_score_rules('/root/nailfold/artifacts/labels/score_rules_v3.json')

    global MEASUREMENT_BASELINE_SCORES
    measure_fields = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
    idx = dev.set_index('exam_case_id')
    for mf in measure_fields:
        vals = pd.to_numeric(idx[mf], errors='coerce').dropna()
        if len(vals) == 0:
            MEASUREMENT_BASELINE_SCORES[mf] = 0.0; continue
        median_val = float(vals.median())
        fr = score_rules["field_rules"].get(mf)
        if fr and fr["type"] == "numeric_tree":
            node = fr["tree"]
            while "threshold" in node:
                node = node["left"] if median_val <= node["threshold"] else node["right"]
            MEASUREMENT_BASELINE_SCORES[mf] = node["value"]
        else:
            MEASUREMENT_BASELINE_SCORES[mf] = 0.0

    frames = pd.read_csv('/root/nailfold/artifacts/features/image_index.csv',
                         dtype={'exam_case_id': str})
    frames = frames[frames.exam_case_id.isin(set(dev.exam_case_id))]

    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    device = f'cuda:{args.gpu}'
    class_weights = compute_class_weights(dev)
    weights_path = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'

    oof_logits = {f: {} for f in FIELDS}

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

            model = build_model(weights_path, device,
                                use_bottleneck=cfg['bottleneck'], use_mse=cfg['mse'])
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, weight_decay=0.05)
            sched = CosineAnnealingLR(opt, T_max=args.epochs)

            best = (-1, None, 0)
            for epoch in range(1, args.epochs + 1):
                train_epoch(model, train_dl, opt, device, class_weights,
                            use_score_weight=cfg['score_weight'], use_mse=cfg['mse'])
                sched.step()
                vp = infer(model, val_dl, device, use_mse=cfg['mse'])
                vp_cls = preds_to_classes(vp, cfg['mse'])
                obj = objective(vp_cls, valids, dev.set_index('exam_case_id'))
                if obj > best[0]:
                    best = (obj, {k: v.cpu().clone() for k, v in model.state_dict().items()}, epoch)

            model.load_state_dict(best[1])
            test_preds = infer(model, test_dl, device, use_mse=cfg['mse'])

            for f in FIELDS:
                for cid, v in test_preds[f].items():
                    oof_logits[f].setdefault(cid, []).append((seed, v))

            print(f"  s={seed} f={fold} ep={best[2]} val_BA={best[0]:.4f}", flush=True)
            del model, opt, sched; torch.cuda.empty_cache()

    # Ensemble: majority vote
    oof_classes = {}
    for f in FIELDS:
        oof_classes[f] = {}
        for cid, entries in oof_logits[f].items():
            votes = []
            for seed, v in entries:
                if cfg['mse']:
                    nc = N_CLASSES[f]
                    cls_idx = int(round(v * max(nc - 1, 1)))
                    cls_idx = max(0, min(nc - 1, cls_idx))
                    votes.append(cls_idx)
                else:
                    votes.append(int(np.argmax(v)))
            oof_classes[f][cid] = Counter(votes).most_common(1)[0][0]

    # Evaluate
    results = evaluate_full(oof_classes, dev.set_index('exam_case_id'))
    results["config"] = cfg

    print(f"\n{'='*60}", flush=True)
    print(f"EXP-6 config={args.config} ({cfg['name']}) RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total Score MAE: {results.get('total_score_mae', 'N/A')}", flush=True)
    print(f"Spearman rho:    {results.get('spearman', 'N/A')}", flush=True)
    print(f"Overall Acc:     {results.get('overall_acc', 'N/A')}", flush=True)
    print(f"Within ±1:       {results.get('within_1', 'N/A')}", flush=True)
    print(f"Underest rate:   {results.get('underest_rate', 'N/A')}", flush=True)
    print(f"Severe underest: {results.get('severe_underest', 'N/A')}", flush=True)
    print(f"Pred mean:       {results.get('pred_mean', 'N/A')}", flush=True)
    print(f"True mean:       {results.get('true_mean', 'N/A')}", flush=True)

    print(f"\n{'Field':30s} {'BA':>6} {'sMAE':>6}", flush=True)
    for f in FIELDS:
        pf = results.get(f, {})
        if pf:
            print(f"  {f:28s} {pf['ba']:.3f}  {pf['smae']:.3f}", flush=True)

    (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(f"\nSaved to {out}", flush=True)

if __name__ == '__main__':
    main()
