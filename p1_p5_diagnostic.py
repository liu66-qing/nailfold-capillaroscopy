"""
P1: Domain shift audit — can frozen DINOv2 features distinguish dev from locked?
P5: Frozen backbone baseline — how well does frozen DINO + linear head classify, on dev OOF?
Also: check dev case metadata for hidden grouping (archive source, fold distribution).
"""
import torch, timm, math
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FIELDS = ["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}

class Frames(Dataset):
    def __init__(self, df, root, tf):
        self.f = df.reset_index(drop=True); self.root = root; self.tf = tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        im = self.tf(Image.open(self.root / r.image_path).convert("RGB"))
        return im, r.exam_case_id

def extract_features(model, loader):
    """Extract frozen CLS token features, aggregate to case level by mean."""
    feats = {}
    model.eval()
    with torch.inference_mode():
        for x, ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x.cuda(non_blocking=True))  # (B, 768)
            for cid, vec in zip(ids, z.float().cpu().numpy()):
                feats.setdefault(cid, []).append(vec)
    # Mean pool frames to case
    return {c: np.mean(vs, axis=0) for c, vs in feats.items()}

def main():
    WEIGHTS = "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"
    INDEX = "/root/nailfold/artifacts/features/image_index.csv"
    IMGROOT = Path("/root/nailfold/data")
    LABELS = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"

    # Load labels
    labels = pd.read_csv(LABELS)
    labels.exam_case_id = labels.exam_case_id.astype(str)
    dev = labels[labels.evaluation_role == "development"].copy()
    locked = labels[labels.evaluation_role == "locked_test"].copy()

    # Load frame index
    idx = pd.read_csv(INDEX)
    idx.exam_case_id = idx.exam_case_id.astype(str)
    all_ids = set(dev.exam_case_id) | set(locked.exam_case_id)
    frames = idx[idx.exam_case_id.isin(all_ids)].copy()

    # Build frozen DINOv2
    print("Loading frozen DINOv2...")
    b = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=False)
    b = b.cuda().eval()
    for p in b.parameters():
        p.requires_grad = False

    cfg = timm.data.resolve_model_data_config(b)
    evtf = timm.data.create_transform(**cfg, is_training=False)
    loader = DataLoader(Frames(frames, IMGROOT, evtf), batch_size=16, num_workers=4)

    print("Extracting features for all cases...")
    case_feats = extract_features(b, loader)
    del b
    torch.cuda.empty_cache()
    print(f"Extracted features for {len(case_feats)} cases")

    # ==========================================
    # P1: DOMAIN SHIFT AUDIT
    # ==========================================
    print("\n" + "=" * 60)
    print("P1: DOMAIN SHIFT — dev vs locked")
    print("=" * 60)

    dev_ids = sorted(dev.exam_case_id.unique())
    locked_ids = sorted(locked.exam_case_id.unique())

    X_dev = np.array([case_feats[c] for c in dev_ids if c in case_feats])
    X_locked = np.array([case_feats[c] for c in locked_ids if c in case_feats])
    X_all = np.vstack([X_dev, X_locked])
    y_domain = np.array([0]*len(X_dev) + [1]*len(X_locked))

    # Stratified 5-fold CV
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.01, max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    domain_probs = cross_val_predict(pipe, X_all, y_domain, cv=cv, method="predict_proba")[:, 1]
    domain_preds = (domain_probs >= 0.5).astype(int)
    domain_auc = roc_auc_score(y_domain, domain_probs)
    domain_ba = balanced_accuracy_score(y_domain, domain_preds)
    print(f"Domain classifier AUC: {domain_auc:.4f}")
    print(f"Domain classifier BA:  {domain_ba:.4f}")
    print(f"(AUC ~0.5 = no shift, AUC >0.7 = significant shift)")

    # Check archive source distribution
    print("\n--- Archive source distribution ---")
    dev["archive"] = dev.exam_case_id.str.split("/").str[0]
    locked["archive"] = locked.exam_case_id.str.split("/").str[0]
    print("Dev:")
    print(dev.archive.value_counts().to_string())
    print("Locked:")
    print(locked.archive.value_counts().to_string())

    # Check class distribution dev vs locked
    print("\n--- Class distribution dev vs locked ---")
    for f in FIELDS:
        y_dev = dev[f].map(MAP[f])
        y_lock = locked[f].map(MAP[f])
        dev_rate = y_dev.dropna().mean()
        lock_rate = y_lock.dropna().mean()
        print(f"  {f:28s}: dev pos_rate={dev_rate:.3f}, locked pos_rate={lock_rate:.3f}, delta={lock_rate-dev_rate:+.3f}")

    # ==========================================
    # P5: FROZEN BACKBONE BASELINE
    # ==========================================
    print("\n" + "=" * 60)
    print("P5: FROZEN DINOv2 + LogisticRegression (dev 5-fold CV)")
    print("=" * 60)

    dev_feat_ids = [c for c in dev_ids if c in case_feats]
    X_dev_feat = np.array([case_feats[c] for c in dev_feat_ids])
    dev_sub = dev[dev.exam_case_id.isin(dev_feat_ids)].set_index("exam_case_id").loc[dev_feat_ids]
    folds = dev_sub.development_fold.values.astype(int)

    for f in FIELDS:
        y = dev_sub[f].map(MAP[f])
        valid = y.notna()
        X_f = X_dev_feat[valid]
        y_f = y[valid].values.astype(int)
        folds_f = folds[valid]

        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
        preds = cross_val_predict(pipe, X_f, y_f, groups=folds_f, cv=LeaveOneGroupOut())
        ba = balanced_accuracy_score(y_f, preds)

        # Compare with LoRA baseline v2
        print(f"  {f:28s}: frozen_BA={ba:.4f}")

    # ==========================================
    # AGGREGATION COMPARISON (mean vs median vs trimmed mean)
    # ==========================================
    print("\n" + "=" * 60)
    print("AGGREGATION: per-frame feature stats")
    print("=" * 60)

    # For each dev case, compute frame-level feature stats
    frame_counts = {}
    for cid in dev_feat_ids:
        frames_for_case = idx[idx.exam_case_id == cid]
        frame_counts[cid] = len(frames_for_case)
    counts = [frame_counts.get(c, 0) for c in dev_feat_ids]
    print(f"Frames per case: mean={np.mean(counts):.1f}, median={np.median(counts):.1f}, min={np.min(counts)}, max={np.max(counts)}")

if __name__ == "__main__":
    main()
