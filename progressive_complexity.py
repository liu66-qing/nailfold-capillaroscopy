"""
Experiment: Progressive complexity on dev OOF, case-level training.
Compare 4 configs, all with:
  - 5 seeds (17,42,123,456,789)
  - 5-fold CV, all 5 fields in val objective
  - CosineAnnealingLR, head dropout=0.1
  - Case-level loss: 0.75*CE(mean_logits) + 0.25*mean(CE(frame_logits))
  - Aggregation comparison: mean vs median logits

Configs:
  A. Frozen backbone + linear head (no LoRA)
  B. LoRA rank=2, last 2 blocks
  C. LoRA rank=4, last 4 blocks (reduced from baseline v2)
  D. LoRA rank=8, last 4 blocks (same as baseline v2, but with all improvements)

Outputs: OOF soft predictions per config, per-field BA, bootstrap CI.
"""
import torch, torch.nn as nn, timm, math, json, argparse, os, sys
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

FIELDS = ["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
SEEDS = [17, 42, 123, 456, 789]

class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale

class Model(nn.Module):
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.b = backbone
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({f: nn.Linear(768, 2) for f in FIELDS})
    def forward(self, x):
        z = self.b(x)
        z = self.drop(z)
        return {f: h(z) for f, h in self.heads.items()}

class CaseDataset(Dataset):
    """Returns all frames for a single case."""
    def __init__(self, case_ids, frame_df, labels_df, root, tf):
        self.cases = case_ids
        self.frame_df = frame_df
        self.labels_df = labels_df.set_index("exam_case_id")
        self.root = root
        self.tf = tf
    def __len__(self):
        return len(self.cases)
    def __getitem__(self, i):
        cid = self.cases[i]
        frames = self.frame_df[self.frame_df.exam_case_id == cid]
        imgs = []
        for _, r in frames.iterrows():
            im = self.tf(Image.open(self.root / r.image_path).convert("RGB"))
            imgs.append(im)
        imgs = torch.stack(imgs)  # (N_frames, 3, H, W)
        row = self.labels_df.loc[cid]
        y = torch.tensor([MAP[f].get(row[f], -1) if pd.notna(row[f]) else -1 for f in FIELDS])
        return imgs, y, cid

def case_collate(batch):
    """Variable-length frame batches."""
    imgs_list, ys, cids = zip(*batch)
    return list(imgs_list), torch.stack(ys), list(cids)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for imgs_list, ys, cids in loader:
        ys = ys.to(device)
        batch_loss = 0
        for case_idx, imgs in enumerate(imgs_list):
            imgs = imgs.to(device)
            y = ys[case_idx]  # (5,)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)  # each head: (N_frames, 2)
            case_loss = 0
            n_tasks = 0
            for fi, f in enumerate(FIELDS):
                if y[fi] == -1:
                    continue
                logits = out[f]  # (N_frames, 2)
                target = y[fi].unsqueeze(0).expand(logits.size(0)).long()
                # Case-level: CE on mean logits
                mean_logits = logits.mean(dim=0, keepdim=True)
                case_ce = nn.functional.cross_entropy(mean_logits, y[fi].unsqueeze(0).long())
                # Frame-level: mean of per-frame CE
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

def predict_soft(model, loader, device):
    """Returns {field: {case_id: prob_class1}} using mean logits."""
    model.eval()
    results = {f: {} for f in FIELDS}
    with torch.inference_mode():
        for imgs_list, ys, cids in loader:
            for case_idx, imgs in enumerate(imgs_list):
                imgs = imgs.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(imgs)
                for f in FIELDS:
                    logits = out[f].float()  # (N_frames, 2)
                    mean_logits = logits.mean(dim=0)
                    prob = torch.softmax(mean_logits, dim=0)[1].item()
                    results[f][cids[case_idx]] = prob
    return results

def build_model(weights_path, config, device):
    b = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
    for p in b.parameters():
        p.requires_grad = False

    if config["lora_blocks"] > 0:
        rank = config["rank"]
        n_blocks = config["lora_blocks"]
        for block in b.blocks[-n_blocks:]:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank)
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank)

    m = Model(b, dropout=config.get("dropout", 0.1))
    return m.to(device)

def run_config(config, labels, frame_df, imgroot, weights_path, device):
    dev = labels[labels.evaluation_role == "development"].copy()
    dev.exam_case_id = dev.exam_case_id.astype(str)

    cfg_tmp = timm.data.resolve_model_data_config(
        timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0))
    trtf = timm.data.create_transform(**cfg_tmp, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    evtf = timm.data.create_transform(**cfg_tmp, is_training=False)

    all_oof = {f: {} for f in FIELDS}  # {field: {case_id: [prob_per_seed]}}

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

            train_loader = DataLoader(
                CaseDataset(train_cases, frame_df, dev, imgroot, trtf),
                batch_size=4, shuffle=True, collate_fn=case_collate, num_workers=2, pin_memory=True)
            val_loader = DataLoader(
                CaseDataset(val_cases, frame_df, dev, imgroot, evtf),
                batch_size=4, shuffle=False, collate_fn=case_collate, num_workers=2)
            test_loader = DataLoader(
                CaseDataset(test_cases, frame_df, dev, imgroot, evtf),
                batch_size=4, shuffle=False, collate_fn=case_collate, num_workers=2)

            model = build_model(weights_path, config, device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=0.05)
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

            best_obj = -1
            best_state = None

            for epoch in range(1, config["epochs"] + 1):
                train_one_epoch(model, train_loader, optimizer, device)
                scheduler.step()

                # Validate: all 5 fields
                val_preds = predict_soft(model, val_loader, device)
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

            # Load best and predict test fold
            model.load_state_dict(best_state)
            test_preds = predict_soft(model, test_loader, device)
            for f in FIELDS:
                for c in test_cases:
                    if c in test_preds[f]:
                        seed_oof[f][c] = test_preds[f][c]

            del model, optimizer, scheduler, best_state
            torch.cuda.empty_cache()

        # Accumulate seed results
        for f in FIELDS:
            for c, p in seed_oof[f].items():
                all_oof[f].setdefault(c, []).append(p)

    # Ensemble: mean across seeds
    ensemble_oof = {f: {c: np.mean(ps) for c, ps in v.items()} for f, v in all_oof.items()}

    # Evaluate
    print(f"\n  Config: {config['name']}")
    print(f"  {'field':28s} | n    | BA_ensemble | BA_single_seed17")
    results = {}
    for f in FIELDS:
        y = dev.set_index("exam_case_id")[f].map(MAP[f])
        valid = [(c, y[c]) for c in ensemble_oof[f] if pd.notna(y.get(c, np.nan))]
        if not valid:
            continue
        cases, y_true = zip(*valid)
        y_true = np.array(y_true, dtype=int)
        y_pred_ens = np.array([int(ensemble_oof[f][c] >= 0.5) for c in cases])
        ba_ens = balanced_accuracy_score(y_true, y_pred_ens)

        # Single seed 17
        y_pred_s17 = np.array([int(all_oof[f][c][0] >= 0.5) for c in cases if len(all_oof[f][c]) > 0])
        y_true_s17 = y_true[:len(y_pred_s17)]
        ba_s17 = balanced_accuracy_score(y_true_s17, y_pred_s17)

        results[f] = {"ba_ensemble": round(ba_ens, 6), "ba_seed17": round(ba_s17, 6), "n": len(valid)}
        print(f"  {f:28s} | {len(valid):4d} | {ba_ens:.4f}      | {ba_s17:.4f}")

    mean_ens = np.mean([v["ba_ensemble"] for v in results.values()])
    print(f"  {'MEAN':28s} |      | {mean_ens:.4f}")

    return {"config": config, "results": results, "mean_ba": round(mean_ens, 6), "oof": ensemble_oof}

def main():
    WEIGHTS = "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"
    INDEX = "/root/nailfold/artifacts/features/image_index.csv"
    IMGROOT = Path("/root/nailfold/data")
    LABELS = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
    OUTDIR = Path("/root/nailfold/artifacts/experiments/progressive_complexity")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(LABELS)
    labels.exam_case_id = labels.exam_case_id.astype(str)
    frame_df = pd.read_csv(INDEX)
    frame_df.exam_case_id = frame_df.exam_case_id.astype(str)
    dev_ids = set(labels[labels.evaluation_role == "development"].exam_case_id)
    frame_df = frame_df[frame_df.exam_case_id.isin(dev_ids)].copy()

    device = "cuda"

    CONFIGS = [
        {"name": "A_frozen", "lora_blocks": 0, "rank": 0, "lr": 1e-3, "epochs": 20, "dropout": 0.1},
        {"name": "B_lora_r2_b2", "lora_blocks": 2, "rank": 2, "lr": 1e-4, "epochs": 20, "dropout": 0.1},
        {"name": "C_lora_r4_b4", "lora_blocks": 4, "rank": 4, "lr": 1e-4, "epochs": 20, "dropout": 0.1},
        {"name": "D_lora_r8_b4", "lora_blocks": 4, "rank": 8, "lr": 1e-4, "epochs": 20, "dropout": 0.1},
    ]

    all_results = []
    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running config: {config['name']}")
        print(f"{'='*60}")
        result = run_config(config, labels, frame_df, IMGROOT, WEIGHTS, device)
        all_results.append(result)

        # Save intermediate
        save = {k: v for k, v in result.items() if k != "oof"}
        with open(OUTDIR / f"{config['name']}_results.json", "w") as fp:
            json.dump(save, fp, indent=2)
        with open(OUTDIR / f"{config['name']}_oof.json", "w") as fp:
            json.dump(result["oof"], fp)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'config':20s} | {'mean BA':>8s} | {'clarity':>8s} | {'blood':>8s} | {'exud':>8s} | {'SVP':>8s} | {'papilla':>8s}")
    for r in all_results:
        c = r["config"]["name"]
        res = r["results"]
        vals = [res.get(f, {}).get("ba_ensemble", 0) for f in FIELDS]
        print(f"{c:20s} | {r['mean_ba']:.4f}   | {vals[0]:.4f}   | {vals[1]:.4f}   | {vals[2]:.4f}   | {vals[3]:.4f}   | {vals[4]:.4f}")

    # Save summary
    summary = [{"config": r["config"]["name"], "mean_ba": r["mean_ba"], "per_field": r["results"]} for r in all_results]
    with open(OUTDIR / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved to {OUTDIR}")

if __name__ == "__main__":
    main()
