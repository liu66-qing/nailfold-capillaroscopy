"""EXP-1b: Ordinal-aware loss variant of EXP-1.

Key change: for ordered fields, replace hard CE with score-distance-weighted
soft label loss. A prediction of "+" when truth is "++" is penalized less
than predicting "无" when truth is "++", proportional to score distance.

For non-ordered fields (hemorrhage: 2 classes), keeps standard CE.
"""
import argparse, copy, gzip, json, math, random
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

# ── 12-field multi-class mapping (with merges for tiny classes) ──────────────
FIELDS = [
    "clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla",
    "capillary_count", "crossing_ratio", "malformation_ratio",
    "flow_state", "rbc_aggregation", "microthrombus", "hemorrhage",
]

MAP = {
    "clarity":        {"清晰": 0, "不清": 1, "模糊": 2},                          # 3
    "blood_color":    {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 2},               # 3 (暗紫→暗红, only 2 cases)
    "exudation":      {"无": 0, "+": 1, "++": 2, "+++": 2},                       # 3 (+++→++, only 4 cases)
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3},  # 4
    "papilla":        {"平坦": 0, "浅波纹状": 1, "波纹状": 2},                     # 3
    "capillary_count": {"<1": 0, "1--2": 0, "3--4": 1, "5--6": 2, ">=7": 3},      # 4 (<1,1-2→<=4 merged, ~16 cases)
    "crossing_ratio":  {"<=30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 2,        # 3 (>80%→60-80%, only 5)
                        "[<30%]": 0, "10--30%": 0},                                # dirty labels
    "malformation_ratio": {"<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3,     # 4
                           "[<10%]": 0},                                            # dirty label
    "flow_state":     {"线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3,
                       "粒缓流": 4, "粒摆流": 4, "全停": 4,                        # 5 (粒摆流+全停→粒缓流+)
                       "[线流、线粒流]": 0},                                        # dirty → 线流
    "rbc_aggregation": {"无": 0, "轻度": 1, "中度": 2, "重度": 3,
                        "[无]": 0, "1--30": -1},                                   # 4 (1--30 is wrong, skip)
    "microthrombus":  {"无": 0, "1--2": 1, ">2": 2,
                       "[未见]": 0, "个/min": -1},                                  # 3
    "hemorrhage":     {"无": 0, "1--2": 1,
                       "管袢/一指甲襞": -1},                                        # 2 (dirty → skip)
}

N_CLASSES = {f: max(v for v in m.values() if v >= 0) + 1 for f, m in MAP.items()}

# ── Score rules (loaded at runtime) ─────────────────────────────────────────
# Reverse mapping: class_index → label string (pick first match)
IDX_TO_LABEL = {}
for f, m in MAP.items():
    rev = {}
    for label, idx in m.items():
        if idx >= 0 and idx not in rev:
            rev[idx] = label
    IDX_TO_LABEL[f] = rev

# ── Score mapping from class index to score ──────────────────────────────────
# Built from score_rules_v3.json + IDX_TO_LABEL at runtime
SCORE_MAP = {}  # populated in main()

# For merged classes, use the WORSE (higher) score as conservative estimate
MERGE_SCORE_OVERRIDE = {
    # blood_color class 2 = 暗红(0.4) + 暗紫(0.8) → use population-weighted avg
    # Since 79 暗红 vs 2 暗紫, essentially 0.4
    ("blood_color", 2): None,  # use 暗红 = 0.4
    # exudation class 2 = ++(2.4) + +++(3.6) → weighted by 23 vs 4
    ("exudation", 2): None,    # use ++ = 2.4
    # capillary_count class 0 = <1(6.0) + 1-2(4.0) → 2 cases total, use 4.0
    ("capillary_count", 0): None,  # use 1--2 = 4.0
    # crossing_ratio class 2 = 60-80%(0.8) + >80%(1.2) → use 60-80% = 0.8
    ("crossing_ratio", 2): None,
    # flow_state class 4 = 粒缓流(4.0) + 粒摆流(5.0) + 全停(6.0) → use 粒缓流=4.0
    ("flow_state", 4): None,
}

# Fixed fields (always predict mode)
FIXED_FIELDS = {
    "vasomotion": ("0--1", 0.0),     # 97% same
    "wbc_count": ("1--30", 0.0),     # 98% same
    "sweat_duct": ("0--2", 0.0),     # 99% same
}

SEEDS = [17, 42, 123, 456, 789]

# ── Model ────────────────────────────────────────────────────────────────────
class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden); self.u = nn.Linear(dim, hidden); self.w = nn.Linear(hidden, 1)
    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), dim=1)
        return (a * x).sum(1)

class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone; self.patch_pool = GatedPool(); self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({f: nn.Linear(1536, N_CLASSES[f]) for f in FIELDS})
    def forward(self, x):
        z = self.b.forward_features(x)
        z = torch.cat([z[:, 0], self.patch_pool(z[:, 1:])], 1)
        z = self.drop(z)
        return {f: h(z) for f, h in self.heads.items()}

class Cases(Dataset):
    def __init__(self, ids, frames, labels, root, transform):
        self.ids = list(ids)
        self.groups = {c: g for c, g in frames[frames.exam_case_id.isin(ids)].groupby('exam_case_id')}
        self.labels = labels.set_index('exam_case_id'); self.root = Path(root); self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        cid = self.ids[i]; rows = self.groups[cid]
        imgs = torch.stack([self.transform(Image.open(self.root / r.image_path).convert('RGB')) for _, r in rows.iterrows()])
        paths = rows.image_path.tolist(); row = self.labels.loc[cid]
        y = torch.tensor([MAP[f].get(str(row[f]), -1) if pd.notna(row[f]) else -1 for f in FIELDS])
        return imgs, y, cid, paths

def collate(batch):
    return [x[0] for x in batch], torch.stack([x[1] for x in batch]), [x[2] for x in batch], [x[3] for x in batch]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build(weights, device):
    b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=False)
    for p in b.parameters(): p.requires_grad = False
    return Model(b).to(device)

def compute_class_weights(labels, cap=5.0):
    """Sqrt inverse-frequency class weights, capped at `cap`."""
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

# ── Ordinal-aware score distance matrices ────────────────────────────────────
# For each field, precompute |score_i - score_j| between all class pairs.
# Used to weight soft labels: closer classes get higher soft target mass.
SCORE_DIST_MATRIX = {}  # populated after SCORE_MAP is loaded

def build_score_distance_matrices():
    """Build per-field score distance matrices from SCORE_MAP."""
    for f in FIELDS:
        if f not in SCORE_MAP: continue
        nc = N_CLASSES[f]
        if nc <= 2:  # binary fields, no ordinal structure
            continue
        scores = np.array([SCORE_MAP[f].get(i, 0.0) for i in range(nc)])
        # Distance matrix: |s_i - s_j|
        dist = np.abs(scores[:, None] - scores[None, :])
        # Convert to soft target weights: closer = higher weight
        # Use exponential: w_ij = exp(-dist_ij / temperature)
        # Temperature = median nonzero distance
        nonzero = dist[dist > 0]
        temp = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        soft_w = np.exp(-dist / temp)
        # Normalize rows to sum to 1
        soft_w /= soft_w.sum(axis=1, keepdims=True)
        SCORE_DIST_MATRIX[f] = torch.tensor(soft_w, dtype=torch.float32)

def ordinal_ce(logits, target_idx, field, class_weights_f, device):
    """Score-distance-weighted soft label cross entropy.
    For fields with ordinal structure, uses soft targets based on score proximity.
    For binary fields, falls back to standard CE.
    """
    if field not in SCORE_DIST_MATRIX:
        return nn.functional.cross_entropy(logits, target_idx, weight=class_weights_f)

    soft_matrix = SCORE_DIST_MATRIX[field].to(device)
    # soft_targets: (batch, nc) - soft distribution centered on true class
    soft_targets = soft_matrix[target_idx]  # (batch, nc)

    # Weighted log-softmax
    log_probs = nn.functional.log_softmax(logits, dim=1)
    # Apply class weights to log_probs
    if class_weights_f is not None:
        w = class_weights_f.to(device)
        log_probs = log_probs * w.unsqueeze(0)

    # Soft CE: -sum(soft_target * log_prob)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss

def train_epoch(model, loader, opt, device, class_weights):
    model.train()
    for imgs_list, ys, _, _ in loader:
        loss = 0.; ncases = 0
        for imgs, y in zip(imgs_list, ys):
            imgs = imgs.to(device); y = y.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(imgs)
            one = 0.; nt = 0
            for i, f in enumerate(FIELDS):
                if y[i] < 0: continue
                w = class_weights[f].to(device)
                logits_f = out[f].float()
                case_logits = logits_f.mean(0, keepdim=True)
                case_loss = ordinal_ce(case_logits, y[i:i+1], f, w, device)
                target = y[i].expand(len(imgs))
                frame_loss = ordinal_ce(logits_f, target, f, w, device)
                one += 0.75 * case_loss + 0.25 * frame_loss; nt += 1
            if nt: loss += one / nt; ncases += 1
        loss /= max(ncases, 1)
        opt.zero_grad(); loss.backward(); opt.step()

def infer(model, loader, device):
    """Returns per-field per-case logits dict: {field: {case_id: np.array(n_classes)}}."""
    cases = {f: {} for f in FIELDS}
    model.eval()
    with torch.inference_mode():
        for imgs_list, _, ids, _ in loader:
            for imgs, cid in zip(imgs_list, ids):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(imgs.to(device))
                for f in FIELDS:
                    logits = out[f].float().cpu()
                    # Case-level: mean logits then softmax
                    cases[f][cid] = logits.mean(0).numpy()
    return cases

def infer_with_frames(model, loader, device, seed, fold):
    """Returns (case_logits, frame_rows) for detailed analysis."""
    cases = {f: {} for f in FIELDS}; rows = []
    model.eval()
    with torch.inference_mode():
        for imgs_list, _, ids, paths_list in loader:
            for imgs, cid, paths in zip(imgs_list, ids, paths_list):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(imgs.to(device))
                for f in FIELDS:
                    logits = out[f].float().cpu()
                    cases[f][cid] = logits.mean(0).numpy()
                    for p, z in zip(paths, logits.numpy()):
                        row = {"seed": seed, "test_fold": fold, "exam_case_id": cid,
                               "image_path": p, "field": f}
                        for ci in range(N_CLASSES[f]):
                            row[f"logit_{ci}"] = float(z[ci])
                        rows.append(row)
    return cases, rows

def objective(pred, ids, label_index):
    """Mean multi-class BA across fields (for epoch selection)."""
    vals = []
    for f in FIELDS:
        use = [c for c in ids if c in pred[f] and pd.notna(label_index.at[c, f])]
        if not use: continue
        y = np.array([MAP[f].get(str(label_index.at[c, f]), -1) for c in use])
        mask = y >= 0; y = y[mask]; use = [u for u, m in zip(use, mask) if m]
        if len(set(y)) < 2: continue
        p = np.array([pred[f][c].argmax() for c in use])
        vals.append(balanced_accuracy_score(y, p))
    return float(np.mean(vals)) if vals else 0.0

def load_score_rules(path):
    """Load score_rules_v3.json and build SCORE_MAP: {field: {class_idx: score}}.
    For merged classes, uses population-weighted average score.
    """
    rules = json.load(open(path))
    global SCORE_MAP
    for f in FIELDS:
        fr = rules["field_rules"].get(f)
        if fr is None: continue
        SCORE_MAP[f] = {}
        for ci, label in IDX_TO_LABEL[f].items():
            if fr["type"] == "categorical_lookup":
                SCORE_MAP[f][ci] = fr["mapping"].get(label, 0.0)
            else:
                SCORE_MAP[f][ci] = 0.0
    # Override merged-class scores with population-weighted averages
    # capillary_count class 0: <1(6.0, n=1) + 1-2(4.0, n=1) → 5.0
    SCORE_MAP["capillary_count"][0] = 5.0
    # exudation class 2: ++(2.4, n=23) + +++(3.6, n=4) → 2.578
    SCORE_MAP["exudation"][2] = (23 * 2.4 + 4 * 3.6) / 27
    # crossing_ratio class 2: 60-80%(0.4, n=12) + >80%(1.2, n=5) → 0.635
    SCORE_MAP["crossing_ratio"][2] = (12 * 0.4 + 5 * 1.2) / 17
    # flow_state class 4: 粒缓流(1.6, n=13) + 粒摆流(4.0, n=2) + 全停(6.0, n=1) → 2.175
    SCORE_MAP["flow_state"][4] = (13 * 1.6 + 2 * 4.0 + 1 * 6.0) / 16
    # blood_color class 2: 暗红(0.4, n=79) + 暗紫(0.8, n=2) → 0.41
    SCORE_MAP["blood_color"][2] = (79 * 0.4 + 2 * 0.8) / 81
    return rules

# TRUE score lookup: maps original label string → exact score (no merging)
TRUE_SCORE_CACHE = {}  # populated by load_true_score_map

def load_true_score_map(rules):
    """Build label-string → exact score mapping for ground truth evaluation."""
    global TRUE_SCORE_CACHE
    for f in FIELDS:
        fr = rules["field_rules"].get(f)
        if fr is None: continue
        if fr["type"] == "categorical_lookup":
            TRUE_SCORE_CACHE[f] = dict(fr["mapping"])

def pred_to_score(field, class_idx):
    """Convert predicted class index to score for a field."""
    if field in SCORE_MAP:
        return SCORE_MAP[field].get(class_idx, 0.0)
    return 0.0

def compute_total_score(field_preds, score_rules):
    """Compute total score from per-field class predictions.
    field_preds: {field: class_idx} for modeled fields.
    Returns total_score float.
    """
    total = 0.0
    # Modeled categorical fields
    for f in FIELDS:
        if f in field_preds:
            total += pred_to_score(f, field_preds[f])
    # Fixed fields
    for f, (label, score) in FIXED_FIELDS.items():
        total += score
    # Measurement fields: use median baseline scores (no model yet)
    # These are populated in main() from the dev set
    for f, score in MEASUREMENT_BASELINE_SCORES.items():
        total += score
    # Missing fields (output_input_ratio, flow_speed_um_s): score = 0
    return total

MEASUREMENT_BASELINE_SCORES = {}  # populated in main()

OVERALL_THRESHOLDS = [(1.0, "正常"), (2.0, "大致正常"), (4.0, "轻度异常"), (8.0, "中度异常")]

def score_to_assessment(total):
    for thresh, label in OVERALL_THRESHOLDS:
        if total < thresh:
            return label
    return "重度异常"

def evaluate_scores(oof_preds, labels, score_rules):
    """Compute Total Score MAE and Overall Assessment Acc from OOF predictions.
    oof_preds: {field: {case_id: class_idx}} (ensemble predictions)
    labels: DataFrame with exam_case_id index
    Returns dict with metrics.
    """
    results = {"per_field": {}}
    case_ids = None

    # Per-field metrics
    for f in FIELDS:
        if f not in oof_preds: continue
        preds_f = oof_preds[f]
        cases = [c for c in preds_f if c in labels.index and pd.notna(labels.at[c, f])]
        if not cases: continue
        if case_ids is None:
            case_ids = set(cases)
        else:
            case_ids &= set(cases)
        y_true = np.array([MAP[f].get(str(labels.at[c, f]), -1) for c in cases])
        y_pred = np.array([preds_f[c] for c in cases])
        mask = y_true >= 0
        y_true, y_pred, cases_f = y_true[mask], y_pred[mask], [c for c, m in zip(cases, mask) if m]

        ba = balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) >= 2 else float(np.mean(y_true == y_pred))
        # sMAE: score-level MAE
        s_true = np.array([pred_to_score(f, int(yt)) for yt in y_true])
        s_pred = np.array([pred_to_score(f, int(yp)) for yp in y_pred])
        smae = float(np.mean(np.abs(s_true - s_pred)))
        results["per_field"][f] = {"ba": ba, "smae": smae, "n": len(y_true)}

    # Total Score MAE and Overall Assessment Acc
    # Use all cases that have labels for ALL modeled fields? No — use union, fill missing with mode score.
    # Actually: for fair comparison, compute on cases that have true total_score in the label CSV.
    if "total_score" in labels.columns:
        ts_cases = [c for c in labels.index if pd.notna(labels.at[c, "total_score"]) and
                    all(c in oof_preds.get(f, {}) for f in FIELDS)]
        true_ts = []; pred_ts = []
        for c in ts_cases:
            true_ts.append(float(labels.at[c, "total_score"]))
            fp = {f: oof_preds[f][c] for f in FIELDS}
            pred_ts.append(compute_total_score(fp, score_rules))
        true_ts = np.array(true_ts); pred_ts = np.array(pred_ts)
        results["total_score_mae"] = float(np.mean(np.abs(true_ts - pred_ts)))
        results["pred_total_mean"] = float(np.mean(pred_ts))
        results["pred_total_std"] = float(np.std(pred_ts))
        results["true_total_mean"] = float(np.mean(true_ts))
        results["true_total_std"] = float(np.std(true_ts))
        results["n_total"] = len(ts_cases)
        # Overall assessment
        if "overall_assessment" in labels.columns:
            oa_cases = [c for c in ts_cases if pd.notna(labels.at[c, "overall_assessment"])]
            pred_oa = [score_to_assessment(compute_total_score({f: oof_preds[f][c] for f in FIELDS}, score_rules)) for c in oa_cases]
            true_oa = [str(labels.at[c, "overall_assessment"]) for c in oa_cases]
            acc = float(np.mean([p == t for p, t in zip(pred_oa, true_oa)]))
            results["overall_assessment_acc"] = acc
            results["n_overall"] = len(oa_cases)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--fields-split', type=int, default=0, choices=[0, 1],
                    help='0=first 6 fields, 1=last 6 fields (for dual-GPU parallel)')
    args = ap.parse_args()

    out = Path('/root/nailfold/artifacts/experiments/exp1b_ordinal')
    out.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                         dtype={'exam_case_id': str})
    dev = labels[labels.evaluation_role == 'development'].copy()
    dev['development_fold'] = dev['development_fold'].astype(int)

    # Load score rules
    score_rules = load_score_rules('/root/nailfold/artifacts/labels/score_rules_v3.json')
    load_true_score_map(score_rules)
    build_score_distance_matrices()
    print("Score distance matrices built for:", list(SCORE_DIST_MATRIX.keys()), flush=True)

    # Compute measurement field baseline scores (median)
    global MEASUREMENT_BASELINE_SCORES
    measure_fields = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
    idx = dev.set_index('exam_case_id')
    for mf in measure_fields:
        vals = pd.to_numeric(idx[mf], errors='coerce').dropna()
        if len(vals) == 0:
            MEASUREMENT_BASELINE_SCORES[mf] = 0.0
            continue
        median_val = float(vals.median())
        # Apply numeric_tree from score_rules
        fr = score_rules["field_rules"].get(mf)
        if fr and fr["type"] == "numeric_tree":
            node = fr["tree"]
            while "threshold" in node:
                if median_val <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            MEASUREMENT_BASELINE_SCORES[mf] = node["value"]
        else:
            MEASUREMENT_BASELINE_SCORES[mf] = 0.0
    print("Measurement baseline scores:", MEASUREMENT_BASELINE_SCORES, flush=True)

    # Image index
    frames = pd.read_csv('/root/nailfold/artifacts/features/image_index.csv',
                         dtype={'exam_case_id': str})
    frames = frames[frames.exam_case_id.isin(set(dev.exam_case_id))]

    # Transforms
    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    device = f'cuda:{args.gpu}'
    class_weights = compute_class_weights(dev)
    print("Class weights:", {f: w.tolist() for f, w in class_weights.items()}, flush=True)

    # Collect OOF predictions across all seeds and folds
    # oof_logits[field][case_id] = list of (seed, logits_array)
    oof_logits = {f: {} for f in FIELDS}
    all_frame_rows = []
    epoch_selections = []

    for seed in SEEDS:
        seed_all(seed)
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
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

            model = build('/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth', device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, weight_decay=0.05)
            sched = CosineAnnealingLR(opt, T_max=args.epochs)

            best = (-1, None, 0)
            for epoch in range(1, args.epochs + 1):
                train_epoch(model, train_dl, opt, device, class_weights)
                sched.step()
                vp = infer(model, val_dl, device)
                obj = objective(vp, valids, dev.set_index('exam_case_id'))
                if obj > best[0]:
                    best = (obj, {k: v.cpu().clone() for k, v in model.state_dict().items()}, epoch)

            model.load_state_dict(best[1])
            test_cases, frame_rows = infer_with_frames(model, test_dl, device, seed, fold)
            all_frame_rows.extend(frame_rows)
            epoch_selections.append({'seed': seed, 'fold': fold, 'best_epoch': best[2],
                                     'val_mean_ba': round(best[0], 4)})

            # Accumulate OOF logits
            for f in FIELDS:
                for cid, logits in test_cases[f].items():
                    oof_logits[f].setdefault(cid, []).append((seed, logits))

            print(f"  seed={seed} fold={fold} epoch={best[2]} val_BA={best[0]:.4f}", flush=True)
            del model, opt, sched, best; torch.cuda.empty_cache()

    # Save frame-level predictions
    pd.DataFrame(all_frame_rows).to_csv(out / 'exp1b_test_frames.csv.gz', index=False, compression='gzip')

    # ── Ensemble: majority vote on argmax across 5 seeds ──
    oof_preds = {}  # {field: {case_id: class_idx}}
    oof_per_seed = {s: {f: {} for f in FIELDS} for s in SEEDS}

    for f in FIELDS:
        oof_preds[f] = {}
        for cid, seed_logits_list in oof_logits[f].items():
            votes = []
            for seed, logits in seed_logits_list:
                pred_cls = int(np.argmax(logits))
                votes.append(pred_cls)
                oof_per_seed[seed][f][cid] = pred_cls
            # Majority vote
            from collections import Counter
            oof_preds[f][cid] = Counter(votes).most_common(1)[0][0]

    # ── Evaluate ──
    idx = dev.set_index('exam_case_id')
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS (5-seed majority vote)")
    print("=" * 60)

    results = evaluate_scores(oof_preds, idx, score_rules)

    # Print per-field
    print(f"\n{'Field':<28} {'BA':>6} {'sMAE':>6} {'N':>4}")
    print("-" * 50)
    for f in FIELDS:
        pf = results["per_field"].get(f, {})
        print(f"{f:<28} {pf.get('ba',0):.3f}  {pf.get('smae',0):.3f}  {pf.get('n',0):>4}")

    print(f"\nTotal Score MAE: {results.get('total_score_mae', 'N/A')}")
    print(f"Pred total: {results.get('pred_total_mean', 0):.2f} ± {results.get('pred_total_std', 0):.2f}")
    print(f"True total: {results.get('true_total_mean', 0):.2f} ± {results.get('true_total_std', 0):.2f}")
    print(f"Overall Assessment Acc: {results.get('overall_assessment_acc', 'N/A')}")
    print(f"N cases (total score): {results.get('n_total', 0)}")

    # Per-seed BA
    print("\nPer-seed mean BA:")
    for seed in SEEDS:
        bas = []
        for f in FIELDS:
            cases = [c for c in oof_per_seed[seed][f] if c in idx.index and pd.notna(idx.at[c, f])]
            if not cases: continue
            y = np.array([MAP[f].get(str(idx.at[c, f]), -1) for c in cases])
            p = np.array([oof_per_seed[seed][f][c] for c in cases])
            mask = y >= 0
            if mask.sum() > 0 and len(set(y[mask])) >= 2:
                bas.append(balanced_accuracy_score(y[mask], p[mask]))
        print(f"  seed={seed}: mean_BA={np.mean(bas):.4f}")

    # Save full results
    results["epoch_selections"] = epoch_selections
    results["class_weights"] = {f: w.tolist() for f, w in class_weights.items()}
    results["n_classes"] = N_CLASSES
    results["measurement_baseline_scores"] = MEASUREMENT_BASELINE_SCORES
    (out / 'exp1b_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

    # Save OOF predictions as CSV
    oof_rows = []
    for f in FIELDS:
        for cid, cls_idx in oof_preds[f].items():
            if cid not in idx.index: continue
            true_label = str(idx.at[cid, f]) if pd.notna(idx.at[cid, f]) else None
            true_cls = MAP[f].get(true_label, -1) if true_label else -1
            oof_rows.append({
                "exam_case_id": cid, "field": f,
                "true_label": true_label, "true_cls": true_cls,
                "pred_cls": cls_idx,
                "true_score": pred_to_score(f, true_cls) if true_cls >= 0 else None,
                "pred_score": pred_to_score(f, cls_idx),
            })
    pd.DataFrame(oof_rows).to_csv(out / 'exp1b_oof_predictions.csv', index=False)

    print("\nDone. Results saved to:", out)

if __name__ == '__main__':
    main()
