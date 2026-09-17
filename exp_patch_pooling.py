"""
Experiment: representation & input-scale ablations on frozen DINOv2.
Baseline to beat: A_frozen from progressive complexity (mean BA=0.736).

Configs (select with --config):
  A_cls_only    : CLS token only (control, reproduces A_frozen)
  B_cls_patch   : CLS + gated attention pooling over patch tokens
  C_patch_only  : gated attention pooling over patch tokens only
  D_tta         : CLS only + test-time augmentation (hflip + 2 rotations)
  E_multicrop   : CLS on 5 crops (center + 4 quadrants at native scale)

All configs: 5 seeds, 5-fold CV, 20 epochs, batch=4 cases, dropout=0.1,
case-level loss (0.75*case + 0.25*frame), CosineAnnealingLR, frozen backbone.
Reports: per-field BA + AUC + recall, per-seed std, paired case-level diff vs A.

Usage: python exp_patch_pooling.py --config B_cls_patch --gpu 0
"""
import torch, torch.nn as nn, timm, math, json, os, sys, argparse
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

FIELDS = ["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
SEEDS = [17, 42, 123, 456, 789]

# ── Attention Pooling ──
class GatedAttentionPool(nn.Module):
    """Gated attention pooling over patch tokens. Returns pooled 768-dim vector."""
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1)
    def forward(self, patches):
        # patches: (N_patches, dim)
        a = self.V(patches) * self.U(patches)  # (N, hidden)
        a = self.w(a)  # (N, 1)
        a = torch.softmax(a, dim=0)  # (N, 1)
        return (a * patches).sum(dim=0)  # (dim,)

# ── Models ──
class CLSModel(nn.Module):
    """A: CLS token only (same as A_frozen)."""
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.b = backbone
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        z = self.b(x)  # (B, 768) CLS token
        z = self.drop(z)
        return {f: h(z) for f, h in self.heads.items()}

class CLSPatchModel(nn.Module):
    """B: CLS + gated attention over patch tokens → concat → classify."""
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.b = backbone
        self.pool = GatedAttentionPool(768, 256)
        self.drop = nn.Dropout(dropout)
        # CLS (768) + pooled patches (768) = 1536
        self.heads = nn.ModuleDict({f: nn.Linear(1536, 2) for f in FIELDS})
    def forward(self, x):
        # Get both CLS and patch tokens
        features = self.b.forward_features(x)  # (B, 1+N_patches, 768)
        cls_token = features[:, 0, :]  # (B, 768)
        patch_tokens = features[:, 1:, :]  # (B, N_patches, 768)
        # Pool patches per image
        pooled = torch.stack([self.pool(patch_tokens[i]) for i in range(patch_tokens.size(0))])
        z = torch.cat([cls_token, pooled], dim=1)  # (B, 1536)
        z = self.drop(z)
        return {f: h(z) for f, h in self.heads.items()}

class PatchOnlyModel(nn.Module):
    """C: Gated attention over patch tokens only (no CLS)."""
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.b = backbone
        self.pool = GatedAttentionPool(768, 256)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        features = self.b.forward_features(x)  # (B, 1+N_patches, 768)
        patch_tokens = features[:, 1:, :]  # (B, N_patches, 768)
        pooled = torch.stack([self.pool(patch_tokens[i]) for i in range(patch_tokens.size(0))])
        z = self.drop(pooled)
        return {f: h(z) for f, h in self.heads.items()}

class MultiCropModel(nn.Module):
    """E: CLS token averaged over 5 crops (center + 4 quadrants).
    Each crop is 518x518 taken from the original image at higher effective
    resolution, so local detail is preserved instead of downscaled away.
    Crops are produced in the dataset; this model just pools their CLS tokens.
    """
    def __init__(self, backbone, dropout=0.1, n_crops=5):
        super().__init__()
        self.b = backbone
        self.n_crops = n_crops
        self.pool = GatedAttentionPool(768, 256)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        # x: (B*n_crops, 3, 518, 518) — crops are stacked along batch dim
        z = self.b(x)  # (B*n_crops, 768)
        b = z.size(0) // self.n_crops
        z = z.view(b, self.n_crops, 768)
        # Attention pool across the 5 crops
        pooled = torch.stack([self.pool(z[i]) for i in range(b)])  # (B, 768)
        pooled = self.drop(pooled)
        return {f: h(pooled) for f, h in self.heads.items()}

# ── Dataset (same as progressive_complexity) ──
def make_crops(img, size=518):
    """Return 5 crops (center + 4 quadrants) from a PIL image at native scale.
    Each crop is `size`x`size` taken from the original resolution, so local
    detail is retained instead of being lost to a global downscale.
    """
    W, H = img.size
    # If the image is smaller than the crop window, upscale minimally so
    # a full-size window fits.
    if W < size or H < size:
        scale = max(size / W, size / H)
        img = img.resize((max(size, int(W * scale)), max(size, int(H * scale))), Image.BICUBIC)
        W, H = img.size
    boxes = [
        ((W - size) // 2, (H - size) // 2),      # center
        (0, 0),                                   # top-left
        (W - size, 0),                            # top-right
        (0, H - size),                            # bottom-left
        (W - size, H - size),                     # bottom-right
    ]
    return [img.crop((x, y, x + size, y + size)) for x, y in boxes]

class CaseDataset(Dataset):
    def __init__(self, case_ids, frame_df, labels_df, root, tf, multicrop=False):
        self.cases = case_ids
        self.frame_df = frame_df
        self.labels_df = labels_df.set_index("exam_case_id")
        self.root = Path(root)
        self.tf = tf
        self.multicrop = multicrop
    def __len__(self):
        return len(self.cases)
    def __getitem__(self, idx):
        cid = self.cases[idx]
        rows = self.frame_df[self.frame_df.exam_case_id == cid]
        imgs = []
        for _, r in rows.iterrows():
            p = self.root / r.image_path
            img = Image.open(p).convert("RGB")
            if self.multicrop:
                # 5 crops per frame, flattened into the frame dimension
                for crop in make_crops(img, 518):
                    imgs.append(self.tf(crop))
            else:
                imgs.append(self.tf(img))
        imgs = torch.stack(imgs)
        y = []
        for f in FIELDS:
            raw = self.labels_df.loc[cid, f] if cid in self.labels_df.index else None
            mapped = MAP[f].get(str(raw), -1) if pd.notna(raw) else -1
            y.append(mapped)
        return imgs, torch.tensor(y, dtype=torch.long), cid

def case_collate(batch):
    imgs_list = [b[0] for b in batch]
    ys = torch.stack([b[1] for b in batch])
    cids = [b[2] for b in batch]
    return imgs_list, ys, cids

# ── Training ──
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for imgs_list, ys, cids in loader:
        batch_loss = 0
        for case_idx, imgs in enumerate(imgs_list):
            imgs = imgs.to(device)
            y = ys[case_idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)
            case_loss = 0
            n_tasks = 0
            for fi, f in enumerate(FIELDS):
                if y[fi] == -1:
                    continue
                logits = out[f]
                target = y[fi].unsqueeze(0).expand(logits.size(0)).long().to(device)
                mean_logits = logits.mean(dim=0, keepdim=True)
                case_ce = nn.functional.cross_entropy(mean_logits, y[fi].unsqueeze(0).long().to(device))
                frame_ce = nn.functional.cross_entropy(logits, target)
                case_loss += 0.75 * case_ce + 0.25 * frame_ce
                n_tasks += 1
            if n_tasks > 0:
                batch_loss += case_loss / n_tasks
        batch_loss = batch_loss / len(imgs_list)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        n += 1
    return total_loss / max(n, 1)

def tta_views(imgs):
    """Return TTA views of a frame batch: identity, hflip, vflip, hvflip."""
    return [
        imgs,
        torch.flip(imgs, dims=[3]),        # horizontal
        torch.flip(imgs, dims=[2]),        # vertical
        torch.flip(imgs, dims=[2, 3]),     # both
    ]

def predict_soft(model, loader, device, tta=False):
    model.eval()
    results = {f: {} for f in FIELDS}
    with torch.inference_mode():
        for imgs_list, ys, cids in loader:
            for case_idx, imgs in enumerate(imgs_list):
                imgs = imgs.to(device)
                views = tta_views(imgs) if tta else [imgs]
                # Average logits across TTA views, then across frames
                acc = {f: [] for f in FIELDS}
                for v in views:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(v)
                    for f in FIELDS:
                        acc[f].append(out[f].float())
                for f in FIELDS:
                    logits = torch.stack(acc[f]).mean(dim=0)  # avg over views
                    mean_logits = logits.mean(dim=0)          # avg over frames
                    prob = torch.softmax(mean_logits, dim=0)[1].item()
                    results[f][cids[case_idx]] = prob
    return results

def build_model(weights_path, config_name, device):
    b = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing = b.load_state_dict(sd, strict=False)
    for p in b.parameters():
        p.requires_grad = False

    if config_name in ("A_cls_only", "D_tta"):
        m = CLSModel(b, dropout=0.1)
    elif config_name == "B_cls_patch":
        m = CLSPatchModel(b, dropout=0.1)
    elif config_name == "C_patch_only":
        m = PatchOnlyModel(b, dropout=0.1)
    elif config_name == "E_multicrop":
        m = MultiCropModel(b, dropout=0.1, n_crops=5)
    else:
        raise ValueError(f"Unknown config: {config_name}")
    return m.to(device)

def run_config(config_name, labels, frame_df, imgroot, weights_path, device):
    use_tta = (config_name == "D_tta")
    use_multicrop = (config_name == "E_multicrop")
    dev = labels[labels.evaluation_role == "development"].copy()
    dev.exam_case_id = dev.exam_case_id.astype(str)

    cfg_tmp = timm.data.resolve_model_data_config(
        timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0))
    trtf = timm.data.create_transform(**cfg_tmp, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    evtf = timm.data.create_transform(**cfg_tmp, is_training=False)

    all_oof = {f: {} for f in FIELDS}  # {field: {case_id: [prob_per_seed]}}

    seed_results = []  # per-seed metrics for std calculation

    for seed in SEEDS:
        print(f"  Seed {seed}...", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_oof = {f: {} for f in FIELDS}

        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            train_folds = [i for i in range(5) if i not in (test_fold, val_fold)]

            train_cases = dev[dev.development_fold.astype(int).isin(train_folds)].exam_case_id.tolist()
            val_cases = dev[dev.development_fold.astype(int).eq(val_fold)].exam_case_id.tolist()
            test_cases = dev[dev.development_fold.astype(int).eq(test_fold)].exam_case_id.tolist()

            # Multicrop makes each case 5x larger in the frame dim, so halve
            # the case batch to keep peak memory comparable.
            bs = 2 if use_multicrop else 4
            train_loader = DataLoader(
                CaseDataset(train_cases, frame_df, dev, imgroot, trtf, multicrop=use_multicrop),
                batch_size=bs, shuffle=True, collate_fn=case_collate, num_workers=2, pin_memory=True)
            val_loader = DataLoader(
                CaseDataset(val_cases, frame_df, dev, imgroot, evtf, multicrop=use_multicrop),
                batch_size=bs, shuffle=False, collate_fn=case_collate, num_workers=2)
            test_loader = DataLoader(
                CaseDataset(test_cases, frame_df, dev, imgroot, evtf, multicrop=use_multicrop),
                batch_size=bs, shuffle=False, collate_fn=case_collate, num_workers=2)

            model = build_model(weights_path, config_name, device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.05)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)

            best_obj = -1
            best_state = None

            for epoch in range(1, 21):
                train_one_epoch(model, train_loader, optimizer, device)
                scheduler.step()
                val_preds = predict_soft(model, val_loader, device, tta=use_tta)
                field_bas = []
                for f in FIELDS:
                    y = dev[dev.development_fold.astype(int).eq(val_fold)].set_index("exam_case_id")[f].map(MAP[f])
                    valid_cases = [c for c in val_cases if c in val_preds[f] and pd.notna(y.get(c, np.nan))]
                    if len(valid_cases) < 5:
                        continue
                    y_true = np.array([y[c] for c in valid_cases])
                    y_pred = np.array([int(val_preds[f][c] >= 0.5) for c in valid_cases])
                    field_bas.append(balanced_accuracy_score(y_true, y_pred))
                obj = np.mean(field_bas) if field_bas else 0
                if obj > best_obj:
                    best_obj = obj
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            test_preds = predict_soft(model, test_loader, device, tta=use_tta)
            for f in FIELDS:
                for c in test_cases:
                    if c in test_preds[f]:
                        seed_oof[f][c] = test_preds[f][c]

            del model, optimizer, scheduler, best_state
            torch.cuda.empty_cache()

        # Per-seed metrics
        seed_metrics = {}
        for f in FIELDS:
            y = dev.set_index("exam_case_id")[f].map(MAP[f])
            valid = [(c, y[c]) for c in seed_oof[f] if pd.notna(y.get(c, np.nan))]
            if not valid:
                continue
            cases, y_true = zip(*valid)
            y_true = np.array(y_true, dtype=int)
            y_prob = np.array([seed_oof[f][c] for c in cases])
            y_pred = (y_prob >= 0.5).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = float('nan')
            seed_metrics[f] = {"ba": ba, "auc": auc}
        seed_results.append(seed_metrics)

        # Accumulate
        for f in FIELDS:
            for c, p in seed_oof[f].items():
                all_oof[f].setdefault(c, []).append(p)

    # Ensemble
    ensemble_oof = {f: {c: np.mean(ps) for c, ps in v.items()} for f, v in all_oof.items()}

    # Final metrics
    print(f"\n  Config: {config_name}")
    print(f"  {'field':28s} | n    | BA_ens | AUC_ens | BA_std  | recall_0 | recall_1")
    results = {}
    for f in FIELDS:
        y = dev.set_index("exam_case_id")[f].map(MAP[f])
        valid = [(c, y[c]) for c in ensemble_oof[f] if pd.notna(y.get(c, np.nan))]
        if not valid:
            continue
        cases, y_true = zip(*valid)
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array([ensemble_oof[f][c] for c in cases])
        y_pred = (y_prob >= 0.5).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = float('nan')
        recall_0 = np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).sum() > 0 else float('nan')
        recall_1 = np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).sum() > 0 else float('nan')
        ba_std = np.std([sr[f]["ba"] for sr in seed_results if f in sr])
        results[f] = {
            "ba_ensemble": round(ba, 6), "auc_ensemble": round(auc, 6),
            "ba_std": round(ba_std, 6),
            "recall_0": round(recall_0, 4), "recall_1": round(recall_1, 4),
            "n": len(valid)
        }
        print(f"  {f:28s} | {len(valid):4d} | {ba:.4f} | {auc:.4f}  | {ba_std:.4f}  | {recall_0:.4f}   | {recall_1:.4f}")

    mean_ba = np.mean([v["ba_ensemble"] for v in results.values()])
    mean_auc = np.mean([v["auc_ensemble"] for v in results.values()])
    print(f"  {'MEAN':28s} |      | {mean_ba:.4f} | {mean_auc:.4f}")

    return {
        "config": config_name, "results": results,
        "mean_ba": round(mean_ba, 6), "mean_auc": round(mean_auc, 6),
        "oof": {f: {c: round(p, 6) for c, p in v.items()} for f, v in ensemble_oof.items()},
        "seed_results": seed_results
    }

# ── Paired comparison vs A ──
def paired_comparison(oof_a, oof_b, labels):
    """Case-level paired difference: B minus A."""
    dev = labels[labels.evaluation_role == "development"].copy()
    dev.exam_case_id = dev.exam_case_id.astype(str)
    comparison = {}
    for f in FIELDS:
        y = dev.set_index("exam_case_id")[f].map(MAP[f])
        common = [c for c in oof_a[f] if c in oof_b[f] and pd.notna(y.get(c, np.nan))]
        if not common:
            continue
        y_true = np.array([y[c] for c in common], dtype=int)
        correct_a = np.array([(int(oof_a[f][c] >= 0.5) == y_true[i]) for i, c in enumerate(common)])
        correct_b = np.array([(int(oof_b[f][c] >= 0.5) == y_true[i]) for i, c in enumerate(common)])
        diff = correct_b.astype(int) - correct_a.astype(int)
        comparison[f] = {
            "n": len(common),
            "b_wins": int((diff > 0).sum()),
            "a_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "net_cases": int(diff.sum()),
        }
    return comparison

# ── Main ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["A_cls_only", "B_cls_patch", "C_patch_only"],
                        help="Which configs to run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    OUTDIR = Path("/root/nailfold/artifacts/experiments/patch_pooling")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    INDEX = "/root/nailfold/artifacts/features/image_index.csv"
    LABELS = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
    IMGROOT = "/root/nailfold/data"
    WEIGHTS = "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"

    frame_df = pd.read_csv(INDEX, dtype={"exam_case_id": str})
    labels = pd.read_csv(LABELS, dtype={"exam_case_id": str})

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using GPU {args.gpu}, configs: {args.configs}", flush=True)

    all_results = []

    for cfg in args.configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Running config: {cfg} on GPU {args.gpu}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run_config(cfg, labels, frame_df, IMGROOT, WEIGHTS, device)
        all_results.append(result)

        save = {k: v for k, v in result.items() if k != "oof"}
        with open(OUTDIR / f"{cfg}_results.json", "w") as fp:
            json.dump(save, fp, indent=2)
        with open(OUTDIR / f"{cfg}_oof.json", "w") as fp:
            json.dump(result["oof"], fp)
        print(f"  Saved {cfg}", flush=True)

    # Paired comparisons only when A is included
    a_oof_path = OUTDIR / "A_cls_only_oof.json"
    if a_oof_path.exists():
        with open(a_oof_path) as fp:
            oof_a = json.load(fp)
        for r in all_results:
            if r["config"] == "A_cls_only":
                continue
            comp = paired_comparison(oof_a, r["oof"], labels)
            comp_name = f"{r['config']}_vs_A_cls_only"
            with open(OUTDIR / f"paired_{comp_name}.json", "w") as fp:
                json.dump(comp, fp, indent=2)
            print(f"\n  Paired: {comp_name}")
            for f, v in comp.items():
                print(f"    {f:28s}: +{v['b_wins']} -{v['a_wins']} ={v['ties']} net={v['net_cases']:+d}")

    # Summary (merge with existing if partial)
    summary_path = OUTDIR / "summary.json"
    existing = {}
    if summary_path.exists():
        with open(summary_path) as fp:
            for item in json.load(fp):
                existing[item["config"]] = item
    for r in all_results:
        existing[r["config"]] = {
            "config": r["config"],
            "mean_ba": r["mean_ba"],
            "mean_auc": r["mean_auc"],
            "per_field": r["results"]
        }
    with open(summary_path, "w") as fp:
        json.dump(list(existing.values()), fp, indent=2)

    print(f"\nDone (GPU {args.gpu}). Saved to {OUTDIR}", flush=True)
