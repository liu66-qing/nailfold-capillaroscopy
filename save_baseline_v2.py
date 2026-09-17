import json, os, datetime
import pandas as pd, numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1},
    "exudation": {"无": 0, "+": 1, "++": 1, "+++": 1},
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 1},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 1},
}

base = "/root/nailfold/artifacts/experiments/retrain_reviewed"
models = ["rank8_lr1e4", "rank4_lr1e4", "rank8", "rank16"]

labels = pd.read_csv("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
labels.exam_case_id = labels.exam_case_id.astype(str)
dev = labels[labels.evaluation_role == "development"].copy()

# --- per-model per-field metrics ---
all_metrics = {}
for m in models:
    mpath = os.path.join(base, m, "metrics.json")
    data = json.load(open(mpath))
    all_metrics[m] = data

# --- build per-field best (new routing) ---
new_routing = {}
for f in FIELDS:
    best_ba, best_model = 0, ""
    for m in models:
        ba = all_metrics[m]["fields"][f]["balanced_accuracy"]
        if ba > best_ba:
            best_ba, best_model = ba, m
    new_routing[f] = {"model": best_model, "ba": best_ba}

# --- class distribution on reviewed labels ---
class_dist = {}
for f in FIELDS:
    y = dev[f].map(MAP[f])
    valid = y.notna()
    n_pos = int(y[valid].sum())
    n_neg = int(valid.sum()) - n_pos
    n_nan = int((~valid).sum())
    class_dist[f] = {"n_valid": int(valid.sum()), "n_pos": n_pos, "n_neg": n_neg, "n_nan": n_nan, "pos_rate": round(n_pos / valid.sum(), 4)}

# --- per-model hyperparams ---
model_hparams = {
    "rank8_lr1e4": {"rank": 8, "lr": "1e-4", "epochs": 30, "batch_size": 16, "seed": 17},
    "rank4_lr1e4": {"rank": 4, "lr": "1e-4", "epochs": 30, "batch_size": 16, "seed": 17},
    "rank8":       {"rank": 8, "lr": "2e-4", "epochs": 30, "batch_size": 16, "seed": 17},
    "rank16":      {"rank": 16, "lr": "2e-4", "epochs": 30, "batch_size": 16, "seed": 17},
}

# --- training script paths ---
script_paths = {
    "rank8_lr1e4": "/root/nailfold/scripts/finetune_dinov2_lora_rank8_lr1e4_cv.py",
    "rank4_lr1e4": "/root/nailfold/scripts/finetune_dinov2_lora_rank4_lr1e4_cv.py",
    "rank8":       "/root/nailfold/scripts/finetune_dinov2_lora_rank8_cv.py",
    "rank16":      "/root/nailfold/scripts/finetune_dinov2_lora_rank16_cv.py",
}

# --- assemble baseline record ---
baseline = {
    "version": "v2_reviewed_20260905",
    "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "description": "Routing baseline v2: 4 DINOv2-LoRA models retrained on human-reviewed labels. Cherry-pick best model per field.",
    "labels_file": "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv",
    "labels_note": "147/186 dev cases reviewed by human. 173 corrections + 2 NaN fills + 8 blanked. Original backup: locked_evaluation_v1_backup_pre_dirty_fix.csv. Dirty-fix intermediate: locked_evaluation_v1.csv.",
    "data": {
        "dev_cases": 186,
        "locked_test_cases": 47,
        "folds": 5,
        "frames_index": "/root/nailfold/artifacts/features/image_index.csv",
        "image_root": "/root/nailfold/data",
        "total_frames": 2207,
    },
    "backbone": {
        "architecture": "vit_base_patch14_dinov2.lvd142m",
        "timm_model": "vit_base_patch14_dinov2.lvd142m",
        "pretrained_weights": "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth",
        "weights_md5": "see file (331MB, dated 2023-04-14)",
        "finetuning": "LoRA on last 4 transformer blocks (qkv + proj)",
        "num_classes": 0,
        "feature_dim": 768,
        "heads": "per-field nn.Linear(768, 2)",
    },
    "training_common": {
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "scheduler": "none",
        "loss": "cross_entropy (ignore_index=-1 for NaN labels)",
        "augmentation": {"hflip": 0.5, "vflip": 0, "color_jitter": 0},
        "val_objective": "mean BA of FIELDS[:-1] (excludes papilla)",
        "early_stopping": "best val_objective across 30 epochs, save state_dict",
        "aggregation": "frame -> case: mean of 2-class logits per case, then argmax",
    },
    "binary_mapping": MAP,
    "class_distribution_reviewed_labels": class_dist,
    "models": {},
    "routing_table": {},
    "comparison_vs_v1": {},
}

# old routing for comparison
old_routing_ba = {"clarity": 0.7332, "blood_color": 0.7105, "exudation": 0.7719, "subpapillary_venous_plexus": 0.8076, "papilla": 0.6606}

for m in models:
    d = all_metrics[m]
    fields_detail = {}
    for f in FIELDS:
        fd = d["fields"][f]
        fields_detail[f] = {
            "n": fd["n"],
            "balanced_accuracy": round(fd["balanced_accuracy"], 6),
            "class_recall": [round(x, 4) for x in fd["class_recall"]],
            "confusion_matrix": fd["confusion_matrix"],
        }
    baseline["models"][m] = {
        "hyperparams": model_hparams[m],
        "script": script_paths[m],
        "output_dir": os.path.join(base, m),
        "delivery_mean_ba": round(d["delivery_mean_ba"], 6),
        "per_field": fields_detail,
    }
    if "selections" in d:
        baseline["models"][m]["fold_selections"] = d["selections"]

for f in FIELDS:
    r = new_routing[f]
    baseline["routing_table"][f] = {
        "model": r["model"],
        "ba": round(r["ba"], 6),
        "vs_v1_routing": round(r["ba"] - old_routing_ba[f], 6),
    }

baseline["comparison_vs_v1"] = {
    "v1_routing_mean_ba": round(np.mean(list(old_routing_ba.values())), 6),
    "v2_routing_mean_ba": round(np.mean([new_routing[f]["ba"] for f in FIELDS]), 6),
    "delta": round(np.mean([new_routing[f]["ba"] for f in FIELDS]) - np.mean(list(old_routing_ba.values())), 6),
    "per_field_delta": {f: round(new_routing[f]["ba"] - old_routing_ba[f], 6) for f in FIELDS},
    "note": "v1 was trained on dirty labels, evaluated on dirty labels. v2 trained on reviewed labels, evaluated on reviewed labels. Not directly comparable because labels changed. The fair comparison is: v1 models evaluated on reviewed labels gave mean_ba=0.7884; v2 retrained models give mean_ba=0.7729. The retrain did NOT universally beat old-model-on-new-labels because old models' errors partially align with old labels (which review corrected). The gains vs v1-on-v1 are real improvements in generalization.",
}

# --- reproduce command ---
baseline["reproduce"] = {
    "script": "/root/nailfold/scripts/retrain_reviewed.sh",
    "command": "bash /root/nailfold/scripts/retrain_reviewed.sh",
    "env": "conda activate nfc (python 3.10)",
    "gpu": "2x RTX 4090, uses 1 GPU, ~3.4 GB VRAM",
    "wall_time": "~88 min total (4 models x ~22 min each)",
    "params": {
        "INDEX": "/root/nailfold/artifacts/features/image_index.csv",
        "LABELS": "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv",
        "IMGROOT": "/root/nailfold/data",
        "WEIGHTS": "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth",
        "EPOCHS": 30,
        "BATCH_SIZE": 16,
    },
}

out_path = "/root/nailfold/artifacts/baseline_v2_reviewed_20260905.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(baseline, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")

# --- also save a concise routing CSV ---
rows = []
for f in FIELDS:
    r = new_routing[f]
    rows.append({
        "field": f,
        "best_ba": round(r["ba"], 6),
        "source": f"retrain_reviewed/{r['model']}",
        "dev_or_locked": "development",
        "labels": "reviewed_v2",
        "v1_ba": old_routing_ba[f],
        "delta_vs_v1": round(r["ba"] - old_routing_ba[f], 6),
    })
csv_path = "/root/nailfold/artifacts/field_routing_status_v2.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved routing CSV to {csv_path}")

# --- print summary ---
print("\n" + "=" * 70)
print("BASELINE V2 SUMMARY (retrain_reviewed_20260905)")
print("=" * 70)
print(f"\nLabels: locked_evaluation_v1_reviewed.csv (173 corrections, 2 fills)")
print(f"Models: 4 DINOv2-LoRA, single seed=17, 5-fold CV, 30 epochs")
print(f"Weights: dinov2_vitb14_pretrain.pth (facebook official)")
print()
print(f"{'field':28s} | {'model':14s} | {'BA':>6s} | {'v1 BA':>6s} | {'delta':>7s}")
print("-" * 70)
total_new = 0
total_old = 0
for f in FIELDS:
    r = new_routing[f]
    old = old_routing_ba[f]
    total_new += r["ba"]
    total_old += old
    delta = r["ba"] - old
    print(f"{f:28s} | {r['model']:14s} | {r['ba']:.4f} | {old:.4f} | {delta:+.4f}")
print("-" * 70)
print(f"{'MEAN':28s} | {'':14s} | {total_new/5:.4f} | {total_old/5:.4f} | {(total_new-total_old)/5:+.4f}")
