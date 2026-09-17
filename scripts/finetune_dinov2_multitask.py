from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


FIELDS = ("clarity", "blood_color", "crossing_ratio", "malformation_ratio", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")


class Images(Dataset):
    def __init__(self, frame, root, transform, mappings, case_counts, class_weights):
        self.frame, self.root, self.transform = frame.reset_index(drop=True), root, transform
        self.mappings, self.case_counts, self.class_weights = mappings, case_counts, class_weights

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.image_path) as source: image = self.transform(source.convert("RGB"))
        targets, weights = {}, {}
        for field, mapping in self.mappings.items():
            value = str(row[field]) if pd.notna(row[field]) else None
            targets[field] = mapping.get(value, -1)
            weights[field] = (self.class_weights[field].get(value, 0.0) / self.case_counts[row.exam_case_id]) if value in mapping else 0.0
        return image, targets, weights, row.exam_case_id


class MultiTask(nn.Module):
    def __init__(self, backbone, mappings):
        super().__init__(); self.backbone = backbone; self.heads = nn.ModuleDict({field: nn.Linear(768, len(mapping)) for field, mapping in mappings.items()})
    def forward(self, values):
        features = self.backbone(values); return {field: head(features) for field, head in self.heads.items()}


def predict_cases(model, frame, root, transform, mappings, batch_size, workers):
    counts = frame.exam_case_id.value_counts().to_dict(); neutral = {field: {value: 1.0 for value in mapping} for field, mapping in mappings.items()}
    loader = DataLoader(Images(frame, root, transform, mappings, counts, neutral), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    values = {field: {} for field in FIELDS}; model.eval()
    with torch.inference_mode():
        for images, _, _, case_ids in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16): outputs = model(images.cuda(0, non_blocking=True))
            for field in FIELDS:
                batch = outputs[field].float().cpu().numpy()
                for case_id, value in zip(case_ids, batch): values[field].setdefault(case_id, []).append(value)
    predictions = {}
    for field, mapping in mappings.items():
        reverse = np.asarray(list(mapping.keys())); predictions[field] = {case_id: reverse[np.mean(logits, axis=0).argmax()] for case_id, logits in values[field].items()}
    model.train(); return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--image-root", type=Path, required=True); parser.add_argument("--weights", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--nested-early-stopping", action="store_true")
    args = parser.parse_args(); torch.manual_seed(20260816); np.random.seed(20260816)
    index = pd.read_csv(args.index); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"]); roles = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role"); labels = pd.read_csv(args.labels, usecols=["exam_case_id", *FIELDS]); cases = roles.merge(labels, on="exam_case_id", validate="one_to_one"); frame = index.merge(cases, on="exam_case_id", validate="many_to_one")
    mappings = {}
    for field in FIELDS:
        support = cases[field].dropna().astype(str).value_counts(); classes = sorted(support[support >= 5].index); mappings[field] = {value: position for position, value in enumerate(classes)}
    args.output_dir.mkdir(parents=True, exist_ok=True); oof = {field: {} for field in FIELDS}; peak = 0; selected_epochs = []
    for fold in range(5):
        val_fold = (fold + 1) % 5
        excluded_folds = [fold, val_fold] if args.nested_early_stopping else [fold]
        train_frame = frame[~frame.development_fold.astype(int).isin(excluded_folds)].copy(); val_frame = frame[frame.development_fold.astype(int).eq(val_fold)].copy(); test_frame = frame[frame.development_fold.astype(int).eq(fold)].copy(); train_cases = cases[~cases.development_fold.astype(int).isin(excluded_folds)]
        case_counts = train_frame.exam_case_id.value_counts().to_dict(); class_weights = {}
        for field, mapping in mappings.items():
            counts = train_cases[field].dropna().astype(str).value_counts(); class_weights[field] = {value: len(train_cases) / (len(mapping) * counts.get(value, 1)) for value in mapping}
        backbone = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0); state = torch.load(args.weights, map_location="cpu", weights_only=True); backbone.load_state_dict(state, strict=False)
        for parameter in backbone.parameters(): parameter.requires_grad = False
        for block in backbone.blocks[-2:]:
            for parameter in block.parameters(): parameter.requires_grad = True
        for parameter in backbone.norm.parameters(): parameter.requires_grad = True
        model = MultiTask(backbone, mappings).cuda(0)
        config = timm.data.resolve_model_data_config(backbone); train_transform = timm.data.create_transform(**config, is_training=True, color_jitter=0.0, hflip=0.5, vflip=0.0); eval_transform = timm.data.create_transform(**config, is_training=False)
        train_loader = DataLoader(Images(train_frame, args.image_root, train_transform, mappings, case_counts, class_weights), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        optimizer = torch.optim.AdamW([{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5}, {"params": model.heads.parameters(), "lr": 3e-4}], weight_decay=0.05)
        model.train(); best_score, best_state, best_epoch = -1.0, None, args.epochs
        for epoch in range(args.epochs):
            total = 0.0
            for images, targets, weights, _ in train_loader:
                images = images.cuda(0, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images); losses = []
                    for field in FIELDS:
                        target = targets[field].cuda(0); weight = weights[field].cuda(0); valid = target.ge(0)
                        if valid.any(): losses.append((nn.functional.cross_entropy(logits[field][valid], target[valid], reduction="none") * weight[valid]).sum() / weight[valid].sum().clamp_min(1e-6))
                    loss = torch.stack(losses).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss.detach())
                peak = max(peak, torch.cuda.max_memory_allocated(0))
            print(f"fold={fold} epoch={epoch + 1}/{args.epochs} loss={total / len(train_loader):.4f} peak_gib={peak / 2**30:.3f}", flush=True)
            if args.nested_early_stopping:
                predictions = predict_cases(model, val_frame, args.image_root, eval_transform, mappings, args.batch_size, args.workers); scores = []
                val_cases = cases[cases.development_fold.astype(int).eq(val_fold)]
                for field, mapping in mappings.items():
                    valid = val_cases[field].notna() & val_cases[field].astype(str).isin(mapping); truth = val_cases.loc[valid, field].astype(str).to_numpy(); predicted = np.asarray([predictions[field][case_id] for case_id in val_cases.loc[valid, "exam_case_id"]]); scores.append(balanced_accuracy_score(truth, predicted))
                score = float(np.mean(scores)); print(f"fold={fold} epoch={epoch + 1} validation_mean_ba={score:.4f}", flush=True)
                if score > best_score: best_score, best_epoch, best_state = score, epoch + 1, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        selected_epochs.append({"fold": fold, "epoch": best_epoch, "validation_mean_ba": best_score if args.nested_early_stopping else None})
        predictions = predict_cases(model, test_frame, args.image_root, eval_transform, mappings, args.batch_size, args.workers)
        for field in FIELDS: oof[field].update(predictions[field])
        torch.save({"model": model.state_dict(), "mappings": mappings}, args.output_dir / f"fold{fold}.pt"); del model, backbone, optimizer, train_loader; torch.cuda.empty_cache()
    reports = {}
    for field, mapping in mappings.items():
        valid = cases[field].notna() & cases[field].astype(str).isin(mapping); truth = cases.loc[valid, field].astype(str).to_numpy(); prediction = np.asarray([oof[field][case_id] for case_id in cases.loc[valid, "exam_case_id"]]); majority = pd.Series(truth).value_counts().index[0]; reports[field] = {"cases": len(truth), "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "baseline": float(balanced_accuracy_score(truth, np.repeat(majority, len(truth))))}
    report = {"schema_version": "dinov2-multitask-finetune-oof/2.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "epochs": args.epochs, "nested_early_stopping": args.nested_early_stopping, "selected_epochs": selected_epochs, "peak_vram_gib": peak / 2**30, "fields": reports}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
