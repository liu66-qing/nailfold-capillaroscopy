"""Evaluate current best models on locked test set (47 cases).

Strategy:
1. For 5 baseline binary fields (clarity/blood_color/SVP/papilla/exudation):
   Train on ALL dev, predict on test, ensemble 5 seeds
2. For EXP-1 multiclass fields (all 12):
   Train on ALL dev, predict on test, ensemble 5 seeds
3. For mode/median fields: use dev set statistics
4. Compute total score and clinical metrics on test set
"""
import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# ── Field definitions (from EXP-1) ──
FIELDS = [
    "clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla",
    "capillary_count", "crossing_ratio", "malformation_ratio",
    "flow_state", "rbc_aggregation", "microthrombus", "hemorrhage",
]

MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 2},
    "blood_color": {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 2},
    "exudation": {"无": 0, "+": 1, "++": 2, "+++": 2},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 2},
    "capillary_count": {"<1": 0, "1--2": 0, "3--4": 1, "5--6": 2, ">=7": 3},
    "crossing_ratio": {"<=30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 2, "[<30%]": 0, "10--30%": 0},
    "malformation_ratio": {"<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3, "[<10%]": 0},
    "flow_state": {"线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3, "粒缓流": 4, "粒摆流": 4, "全停": 4, "[线流、线粒流]": 0},
    "rbc_aggregation": {"无": 0, "轻度": 1, "中度": 2, "重度": 3, "[无]": 0, "1--30": -1},
    "microthrombus": {"无": 0, "1--2": 1, ">2": 2, "[未见]": 0, "个/min": -1},
    "hemorrhage": {"无": 0, "1--2": 1, "管袢/一指甲襞": -1},
}

N_CLASSES = {f: max(v for v in m.values() if v >= 0) + 1 for f, m in MAP.items()}

SEEDS = [17, 42, 123, 456, 789]

# Score mapping
SCORE_RULES = None
SCORE_MAP = {}

def load_score_rules():
    global SCORE_RULES, SCORE_MAP
    SCORE_RULES = json.load(open("/root/nailfold/artifacts/labels/score_rules_v3.json"))
    IDX_TO_LABEL = {}
    for f, m in MAP.items():
        rev = {}
        for label, idx in m.items():
            if idx >= 0 and idx not in rev:
                rev[idx] = label
        IDX_TO_LABEL[f] = rev

    for f in FIELDS:
        fr = SCORE_RULES["field_rules"].get(f)
        if fr is None or fr["type"] != "categorical_lookup":
            continue
        SCORE_MAP[f] = {}
        for ci, label in IDX_TO_LABEL[f].items():
            SCORE_MAP[f][ci] = fr["mapping"].get(label, 0.0)

    # Merged class overrides
    SCORE_MAP["capillary_count"][0] = 5.0
    SCORE_MAP["exudation"][2] = (23 * 2.4 + 4 * 3.6) / 27
    SCORE_MAP["crossing_ratio"][2] = (12 * 0.4 + 5 * 1.2) / 17
    SCORE_MAP["flow_state"][4] = (13 * 1.6 + 2 * 4.0 + 1 * 6.0) / 16
    SCORE_MAP["blood_color"][2] = (79 * 0.4 + 2 * 0.8) / 81

# ── Model ──
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
        self.labels = labels.set_index('exam_case_id') if 'exam_case_id' in labels.columns else labels
        self.root = Path(root); self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        cid = self.ids[i]; rows = self.groups[cid]
        imgs = torch.stack([self.transform(Image.open(self.root / r.image_path).convert('RGB')) for _, r in rows.iterrows()])
        row = self.labels.loc[cid]
        y = torch.tensor([MAP[f].get(str(row[f]), -1) if pd.notna(row[f]) else -1 for f in FIELDS])
        return imgs, y, cid

def collate(batch):
    return [x[0] for x in batch], torch.stack([x[1] for x in batch]), [x[2] for x in batch]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

def train_and_predict(dev_labels, test_ids, frames, device, epochs=20):
    """Train on ALL dev, predict on test. Returns {field: {case: logits_array}}."""
    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    class_weights = compute_class_weights(dev_labels)
    all_preds = {f: {} for f in FIELDS}  # {field: {case: [logits_per_seed]}}

    for seed in SEEDS:
        seed_all(seed)
        print(f"  Seed {seed}...", end="", flush=True)

        b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
        b.load_state_dict(torch.load('/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth',
                                      map_location='cpu', weights_only=True), strict=False)
        for p in b.parameters(): p.requires_grad = False
        model = Model(b).to(device)

        train_ids = dev_labels.exam_case_id.tolist()
        train_dl = DataLoader(Cases(train_ids, frames, dev_labels, '/root/nailfold/data', tr),
                              batch_size=4, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
        test_dl = DataLoader(Cases(test_ids, frames, dev_labels.append(pd.DataFrame()) if False else
                             pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv",
                                         dtype={"exam_case_id": str}).set_index("exam_case_id"),
                             '/root/nailfold/data', ev),
                             batch_size=4, collate_fn=collate, num_workers=2)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.05)
        sched = CosineAnnealingLR(opt, T_max=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            for imgs_list, ys, _ in train_dl:
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
                        case_ce = nn.functional.cross_entropy(case_logits, y[i:i+1], weight=w)
                        target = y[i].expand(len(imgs))
                        frame_ce = nn.functional.cross_entropy(logits_f, target, weight=w)
                        one += 0.75 * case_ce + 0.25 * frame_ce; nt += 1
                    if nt: loss += one / nt; ncases += 1
                loss /= max(ncases, 1)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        # Predict on test
        model.eval()
        with torch.inference_mode():
            for imgs_list, _, ids in test_dl:
                for imgs, cid in zip(imgs_list, ids):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = model(imgs.to(device))
                    for f in FIELDS:
                        logits = out[f].float().cpu().mean(0).numpy()
                        all_preds[f].setdefault(cid, []).append(logits)

        print(f" done", flush=True)
        del model, opt, sched; torch.cuda.empty_cache()

    # Ensemble: majority vote
    ensemble = {f: {} for f in FIELDS}
    for f in FIELDS:
        for cid, logits_list in all_preds[f].items():
            votes = [int(np.argmax(l)) for l in logits_list]
            from collections import Counter
            ensemble[f][cid] = Counter(votes).most_common(1)[0][0]

    return ensemble

def main():
    load_score_rules()

    labels = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv",
                         dtype={"exam_case_id": str})
    dev = labels[labels.evaluation_role == "development"].copy()
    test = labels[labels.evaluation_role == "locked_test"].copy().set_index("exam_case_id")
    frames = pd.read_csv("/root/nailfold/artifacts/features/image_index.csv", dtype={"exam_case_id": str})

    device = "cuda:0"

    # Mode/median baselines from dev
    dev_idx = dev.set_index("exam_case_id")
    mode_preds = {}
    for f in FIELDS:
        vals = dev_idx[f].dropna().map(MAP[f])
        vals = vals[vals >= 0]
        mode_preds[f] = int(vals.mode().iloc[0])

    # Measurement field medians
    measure_fields = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
    measure_medians = {}
    for mf in measure_fields:
        vals = pd.to_numeric(dev_idx[mf], errors="coerce").dropna()
        measure_medians[mf] = float(vals.median())

    print("=" * 60)
    print("LOCKED TEST SET EVALUATION")
    print("=" * 60)

    # Train multiclass model on ALL dev, predict on test
    print("\nTraining EXP-1 multiclass on full dev set (5 seeds)...")
    test_ids = list(test.index)
    model_preds = train_and_predict(dev, test_ids, frames, device, epochs=20)

    # Now compute test set metrics
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    # Per-field metrics
    print(f"\n{'Field':30s} {'Model BA':>10} {'Model sMAE':>12} {'Mode sMAE':>12} {'Winner':>10}")
    print("-" * 80)

    field_scores_model = {}
    field_scores_mode = {}
    for f in FIELDS:
        cases_f = [c for c in test.index if pd.notna(test.at[c, f]) and MAP[f].get(str(test.at[c, f]), -1) >= 0]
        if not cases_f:
            print(f"{f:30s} {'N/A':>10}")
            continue

        y_true = np.array([MAP[f][str(test.at[c, f])] for c in cases_f])

        # Model predictions
        y_model = np.array([model_preds[f].get(c, mode_preds[f]) for c in cases_f])
        ba = balanced_accuracy_score(y_true, y_model) if len(set(y_true)) >= 2 else 0

        # Score MAE
        true_scores = np.array([SCORE_MAP.get(f, {}).get(int(yt), 0) for yt in y_true])
        model_scores = np.array([SCORE_MAP.get(f, {}).get(int(yp), 0) for yp in y_model])
        mode_scores = np.array([SCORE_MAP.get(f, {}).get(mode_preds[f], 0)] * len(y_true))

        model_smae = float(np.mean(np.abs(true_scores - model_scores)))
        mode_smae = float(np.mean(np.abs(true_scores - mode_scores)))

        winner = "Model" if model_smae < mode_smae else "Mode" if mode_smae < model_smae else "Tie"
        print(f"{f:30s} {ba:10.3f} {model_smae:12.3f} {mode_smae:12.3f} {winner:>10}")

        for c in test.index:
            pred_cls = model_preds[f].get(c, mode_preds[f])
            field_scores_model.setdefault(c, {})[f] = SCORE_MAP.get(f, {}).get(pred_cls, 0)
            field_scores_mode.setdefault(c, {})[f] = SCORE_MAP.get(f, {}).get(mode_preds[f], 0)

    # Total score computation
    print("\n" + "=" * 60)
    print("TOTAL SCORE AND CLINICAL ASSESSMENT")
    print("=" * 60)

    # Compute predicted total scores
    thresholds = [(1.0, "正常"), (2.0, "大致正常"), (4.0, "轻度异常"), (8.0, "中度异常")]
    def score_to_assessment(s):
        for t, label in thresholds:
            if s < t: return label
        return "重度异常"

    # Measurement scores from median
    measure_scores = {}
    for mf in measure_fields:
        fr = SCORE_RULES["field_rules"][mf]
        node = fr["tree"]
        val = measure_medians[mf]
        while "threshold" in node:
            node = node["left"] if val <= node["threshold"] else node["right"]
        measure_scores[mf] = node["value"]

    fixed_score = 0.0  # vasomotion + wbc_count + sweat_duct ≈ 0

    results_rows = []
    for c in test.index:
        true_ts = float(test.at[c, "total_score"]) if pd.notna(test.at[c, "total_score"]) else None
        true_oa = str(test.at[c, "overall_assessment"]) if pd.notna(test.at[c, "overall_assessment"]) else None

        # Model predicted total
        model_ts = sum(field_scores_model.get(c, {}).values()) + sum(measure_scores.values()) + fixed_score
        mode_ts = sum(field_scores_mode.get(c, {}).values()) + sum(measure_scores.values()) + fixed_score

        model_oa = score_to_assessment(model_ts)
        mode_oa = score_to_assessment(mode_ts)

        results_rows.append({
            "case": c, "true_ts": true_ts, "model_ts": model_ts, "mode_ts": mode_ts,
            "true_oa": true_oa, "model_oa": model_oa, "mode_oa": mode_oa
        })

    rdf = pd.DataFrame(results_rows)
    valid = rdf[rdf.true_ts.notna()].copy()

    print(f"\nTest cases with total_score: {len(valid)}")
    print(f"\nModel: Total Score MAE = {np.mean(np.abs(valid.true_ts - valid.model_ts)):.3f}")
    print(f"Mode:  Total Score MAE = {np.mean(np.abs(valid.true_ts - valid.mode_ts)):.3f}")
    print(f"\nModel: pred mean={valid.model_ts.mean():.2f}, true mean={valid.true_ts.mean():.2f}")
    print(f"Mode:  pred mean={valid.mode_ts.mean():.2f}")

    # Overall assessment accuracy
    oa_valid = valid[valid.true_oa.notna()]
    model_acc = (oa_valid.model_oa == oa_valid.true_oa).mean()
    mode_acc = (oa_valid.mode_oa == oa_valid.true_oa).mean()
    print(f"\nModel: Overall Assessment Acc = {model_acc:.3f}")
    print(f"Mode:  Overall Assessment Acc = {mode_acc:.3f}")

    # Confusion matrix
    sev_order = ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"]
    print("\nModel confusion matrix (rows=true, cols=pred):")
    print(f"{'':12s}", end="")
    for s in sev_order: print(f"{s:8s}", end="")
    print()
    for true_s in sev_order:
        row = oa_valid[oa_valid.true_oa == true_s]
        print(f"{true_s:12s}", end="")
        for pred_s in sev_order:
            n = (row.model_oa == pred_s).sum()
            print(f"{n:8d}", end="")
        print()

    # CLINICAL SAFETY METRICS
    print("\n" + "=" * 60)
    print("CLINICAL SAFETY METRICS")
    print("=" * 60)
    sev_map = {"正常": 0, "大致正常": 1, "轻度异常": 2, "中度异常": 3, "重度异常": 4}
    oa_valid["true_sev"] = oa_valid.true_oa.map(sev_map)
    oa_valid["model_sev"] = oa_valid.model_oa.map(sev_map)

    underest = (oa_valid.model_sev < oa_valid.true_sev).sum()
    overest = (oa_valid.model_sev > oa_valid.true_sev).sum()
    correct = (oa_valid.model_sev == oa_valid.true_sev).sum()
    adjacent = (np.abs(oa_valid.model_sev - oa_valid.true_sev) <= 1).sum()

    print(f"Exact match: {correct}/{len(oa_valid)} ({correct/len(oa_valid)*100:.1f}%)")
    print(f"Within ±1 level: {adjacent}/{len(oa_valid)} ({adjacent/len(oa_valid)*100:.1f}%)")
    print(f"Underestimated: {underest}/{len(oa_valid)} ({underest/len(oa_valid)*100:.1f}%)")
    print(f"Overestimated: {overest}/{len(oa_valid)} ({overest/len(oa_valid)*100:.1f}%)")

    # Severe underestimation (off by 2+ levels)
    severe_under = ((oa_valid.true_sev - oa_valid.model_sev) >= 2).sum()
    print(f"Severe underestimation (>=2 levels): {severe_under}/{len(oa_valid)} ({severe_under/len(oa_valid)*100:.1f}%)")

    # Binary: can it detect "needs attention" (>= 轻度)?
    true_needs = (oa_valid.true_sev >= 2).astype(int)
    model_needs = (oa_valid.model_sev >= 2).astype(int)
    tp = ((true_needs == 1) & (model_needs == 1)).sum()
    fn = ((true_needs == 1) & (model_needs == 0)).sum()
    fp = ((true_needs == 0) & (model_needs == 1)).sum()
    tn = ((true_needs == 0) & (model_needs == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nBinary 'needs attention' (>=轻度):")
    print(f"  Sensitivity: {tp}/{tp+fn} = {sens:.3f}")
    print(f"  Specificity: {tn}/{tn+fp} = {spec:.3f}")
    print(f"  TP={tp} FN={fn} FP={fp} TN={tn}")

    # Per-case details
    print("\n" + "=" * 60)
    print("PER-CASE DETAILS (test set)")
    print("=" * 60)
    print(f"{'Case':35s} {'True':6s} {'Pred':6s} {'TrueOA':12s} {'PredOA':12s} {'Delta':>6s}")
    for _, r in valid.sort_values("true_ts").iterrows():
        delta = r.model_ts - r.true_ts
        flag = "!!" if abs(delta) > 4 else ""
        print(f"{r.case:35s} {r.true_ts:6.1f} {r.model_ts:6.1f} {r.true_oa:12s} {r.model_oa:12s} {delta:+6.1f} {flag}")

    # Save
    out = Path("/root/nailfold/artifacts/experiments/test_set_evaluation")
    out.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out / "test_predictions.csv", index=False)

    summary = {
        "total_score_mae_model": float(np.mean(np.abs(valid.true_ts - valid.model_ts))),
        "total_score_mae_mode": float(np.mean(np.abs(valid.true_ts - valid.mode_ts))),
        "overall_acc_model": float(model_acc),
        "overall_acc_mode": float(mode_acc),
        "within_1_level": float(adjacent / len(oa_valid)),
        "underestimation_rate": float(underest / len(oa_valid)),
        "severe_underestimation_rate": float(severe_under / len(oa_valid)),
        "sensitivity_needs_attention": float(sens),
        "specificity_needs_attention": float(spec),
    }
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
