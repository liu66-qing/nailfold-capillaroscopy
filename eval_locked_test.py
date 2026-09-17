"""
Evaluate v2 routing models on locked test set (47 cases, never seen during training).
For each routing field, load best model's 5 fold checkpoints, ensemble predict on locked cases.
"""
import torch, timm, json, math
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

FIELDS = ["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}

class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.a = torch.nn.Linear(base.in_features, rank, bias=False)
        self.b = torch.nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        torch.nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale

class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone
        self.heads = torch.nn.ModuleDict({f: torch.nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        z = self.b(x)
        return {f: h(z) for f, h in self.heads.items()}

class Frames(Dataset):
    def __init__(self, df, root, tf):
        self.f = df.reset_index(drop=True)
        self.root = root
        self.tf = tf
    def __len__(self):
        return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        im = self.tf(Image.open(self.root / r.image_path).convert("RGB"))
        return im, r.exam_case_id

def build_model(weights_path, rank, ckpt_path):
    b = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
    for p in b.parameters():
        p.requires_grad = False
    for block in b.blocks[-4:]:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank)
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank)
    m = Model(b)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        m.load_state_dict(state["state_dict"])
    else:
        m.load_state_dict(state)
    return m.cuda().eval()

def predict_soft(model, loader):
    """Returns {field: {case_id: probability_of_class_1}}"""
    vals = {f: {} for f in FIELDS}
    with torch.inference_mode():
        for x, ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.cuda(non_blocking=True))
            for f in FIELDS:
                probs = torch.softmax(o[f].float(), dim=1)[:, 1].cpu().numpy()
                for c, p in zip(ids, probs):
                    vals[f].setdefault(c, []).append(float(p))
    # Mean across frames per case
    return {f: {c: float(np.mean(ps)) for c, ps in v.items()} for f, v in vals.items()}

def main():
    WEIGHTS = "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"
    INDEX = "/root/nailfold/artifacts/features/image_index.csv"
    IMGROOT = Path("/root/nailfold/data")
    LABELS = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
    BASE = "/root/nailfold/artifacts/experiments/retrain_reviewed"

    # Routing: field -> (model_dir, rank)
    ROUTING = {
        "clarity": ("rank8", 8),
        "blood_color": ("rank8", 8),
        "exudation": ("rank4_lr1e4", 4),
        "subpapillary_venous_plexus": ("rank4_lr1e4", 4),
        "papilla": ("rank16", 16),
    }

    # Load locked test cases
    labels = pd.read_csv(LABELS)
    labels.exam_case_id = labels.exam_case_id.astype(str)
    locked = labels[labels.evaluation_role == "locked_test"].copy()
    print(f"Locked test cases: {len(locked)}")

    # Load frame index, filter to locked cases
    idx = pd.read_csv(INDEX)
    idx.exam_case_id = idx.exam_case_id.astype(str)
    locked_ids = set(locked.exam_case_id)
    locked_frames = idx[idx.exam_case_id.isin(locked_ids)].copy()
    print(f"Locked test frames: {len(locked_frames)}")

    # Build eval transform
    b_tmp = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(b_tmp)
    evtf = timm.data.create_transform(**cfg, is_training=False)
    del b_tmp

    loader = DataLoader(Frames(locked_frames, IMGROOT, evtf), batch_size=16, num_workers=4)

    # For each routing model, load 5 folds, predict, ensemble
    model_preds = {}  # model_name -> {field: {case: prob}}
    needed_models = set(m for m, r in ROUTING.values())

    for model_name in sorted(set(v[0] for v in ROUTING.values())):
        rank = [v[1] for v in ROUTING.values() if v[0] == model_name][0]
        model_dir = Path(BASE) / model_name
        fold_preds = []
        for fold in range(5):
            ckpt = model_dir / f"fold{fold}.pt"
            print(f"  Loading {model_name}/fold{fold}...", end=" ", flush=True)
            m = build_model(WEIGHTS, rank, ckpt)
            preds = predict_soft(m, loader)
            fold_preds.append(preds)
            del m
            torch.cuda.empty_cache()
            print("done")

        # Ensemble: mean across 5 folds
        ensemble = {f: {} for f in FIELDS}
        for f in FIELDS:
            all_cases = set()
            for fp in fold_preds:
                all_cases.update(fp[f].keys())
            for c in all_cases:
                vals = [fp[f][c] for fp in fold_preds if c in fp[f]]
                ensemble[f][c] = float(np.mean(vals))
        model_preds[model_name] = ensemble

    # Evaluate
    print("\n" + "=" * 80)
    print("LOCKED TEST EVALUATION — BASELINE V2")
    print("=" * 80)
    print(f"\n{'field':28s} | {'model':14s} | n    | pos/neg | BA     | recall_0 | recall_1")
    print("-" * 95)

    results = {}
    total_ba = 0
    for f in FIELDS:
        model_name, rank = ROUTING[f]
        preds = model_preds[model_name]

        y = locked[f].map(MAP[f])
        valid = y.notna()
        y_valid = y[valid].values.astype(int)
        case_ids = locked.loc[valid, "exam_case_id"].values

        probs = np.array([preds[f].get(str(c), np.nan) for c in case_ids])
        mask = ~np.isnan(probs)
        y_f = y_valid[mask]
        p_f = probs[mask]

        pred_hard = (p_f >= 0.5).astype(int)
        ba = balanced_accuracy_score(y_f, pred_hard)
        cm = confusion_matrix(y_f, pred_hard, labels=[0, 1])
        rec = recall_score(y_f, pred_hard, labels=[0, 1], average=None, zero_division=0)

        n_pos = int(y_f.sum())
        n_neg = len(y_f) - n_pos

        total_ba += ba
        results[f] = {
            "model": model_name,
            "n": int(mask.sum()),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "ba": round(ba, 6),
            "recall_0": round(float(rec[0]), 4),
            "recall_1": round(float(rec[1]), 4),
            "confusion_matrix": cm.tolist(),
            "threshold": 0.5,
        }
        print(f"{f:28s} | {model_name:14s} | {mask.sum():4d} | {n_pos:3d}/{n_neg:3d} | {ba:.4f} | {rec[0]:.4f}   | {rec[1]:.4f}")

    print("-" * 95)
    print(f"{'MEAN':28s} | {'':14s} |      |         | {total_ba/5:.4f}")

    # Also try threshold optimization per field (using locked probs)
    print("\n--- With threshold search on locked (CAUTION: overfitting on 47 cases) ---")
    for f in FIELDS:
        model_name, rank = ROUTING[f]
        preds = model_preds[model_name]
        y = locked[f].map(MAP[f])
        valid = y.notna()
        y_valid = y[valid].values.astype(int)
        case_ids = locked.loc[valid, "exam_case_id"].values
        probs = np.array([preds[f].get(str(c), np.nan) for c in case_ids])
        mask = ~np.isnan(probs)
        y_f, p_f = y_valid[mask], probs[mask]

        best_ba, best_th = 0, 0.5
        for th in np.arange(0.2, 0.8, 0.01):
            ba = balanced_accuracy_score(y_f, (p_f >= th).astype(int))
            if ba > best_ba:
                best_ba, best_th = ba, th
        print(f"  {f:28s} | BA@0.5={results[f]['ba']:.4f} | best={best_ba:.4f}@{best_th:.2f}")

    # Save results
    out = {
        "version": "v2_locked_test_20260905",
        "description": "Locked test evaluation of baseline v2 routing models. 5-fold ensemble, threshold=0.5.",
        "n_locked_cases": len(locked),
        "labels": LABELS,
        "per_field": results,
        "mean_ba": round(total_ba / 5, 6),
    }
    out_path = "/root/nailfold/artifacts/locked_test_v2_20260905.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
