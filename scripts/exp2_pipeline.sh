#!/bin/bash
# Master scheduler: runs EXP-2 on GPU 1 while EXP-1 runs on GPU 0
# EXP-2 data prep can start immediately; training starts right away since
# it uses completely different data (detection boxes, not case-level labels)
set -e

echo "[$(date)] EXP-2 pipeline starting on GPU 1"

# Step 1: Unpack images (should already be uploaded)
IMG_TAR=/tmp/det_images.tar
IMG_DST=/root/nailfold/data/yolo_det_3class/images

if [ ! -d "$IMG_DST" ] || [ "$(ls $IMG_DST/*.jpg 2>/dev/null | wc -l)" -lt 11000 ]; then
    echo "[$(date)] Unpacking images..."
    mkdir -p $IMG_DST
    cd /tmp && tar xf $IMG_TAR
    mv /tmp/image/*.jpg $IMG_DST/
    echo "[$(date)] Images unpacked: $(ls $IMG_DST/*.jpg | wc -l) files"
else
    echo "[$(date)] Images already present: $(ls $IMG_DST/*.jpg | wc -l) files"
fi

# Step 2: Create image symlinks for each fold
YOLO_BASE=/root/nailfold/data/yolo_det_3class
for fold in 0 1 2 3 4; do
    for split in train val; do
        DST=$YOLO_BASE/fold${fold}/${split}/images
        mkdir -p "$DST"
        count=0
        for label in $YOLO_BASE/fold${fold}/${split}/labels/*.txt; do
            stem=$(basename "$label" .txt)
            src="$IMG_DST/${stem}.jpg"
            if [ -f "$src" ] && [ ! -e "$DST/${stem}.jpg" ]; then
                ln -s "$src" "$DST/${stem}.jpg"
            fi
            count=$((count+1))
        done
        echo "  Fold $fold $split: $count labels -> images linked"
    done
done

# Step 3: Install ultralytics if needed
/root/miniconda3/bin/python -c "from ultralytics import YOLO" 2>/dev/null || {
    echo "[$(date)] Installing ultralytics..."
    /root/miniconda3/bin/pip install ultralytics -q
}

# Step 4: Train YOLO 5-fold on GPU 1
export CUDA_VISIBLE_DEVICES=1
OUT_BASE=/root/nailfold/artifacts/experiments/exp2_yolo_det
mkdir -p $OUT_BASE

for fold in 0 1 2 3 4; do
    echo "========================================"
    echo "[$(date)] EXP-2 Training fold $fold on GPU 1"
    echo "========================================"
    /root/miniconda3/bin/python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
results = model.train(
    data='/root/nailfold/data/yolo_det_3class/fold${fold}/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='${OUT_BASE}',
    name='fold${fold}',
    exist_ok=True,
    patience=15,
    save=True,
    plots=True,
    workers=4,
    seed=42,
    verbose=True,
)
"
    echo "[$(date)] Fold $fold done"
done

echo "[$(date)] EXP-2 training complete. Starting OOF inference..."

# Step 5: OOF inference
/root/miniconda3/bin/python << 'PYEOF'
import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

out_base = Path("/root/nailfold/artifacts/experiments/exp2_yolo_det")
yolo_base = Path("/root/nailfold/data/yolo_det_3class")

from ultralytics import YOLO

oof_results = {}
for fold in range(5):
    model_path = out_base / f"fold{fold}" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"WARN: {model_path} not found")
        continue
    model = YOLO(str(model_path))
    val_img_dir = yolo_base / f"fold{fold}" / "val" / "images"
    for img_path in sorted(val_img_dir.glob("*.jpg")):
        results = model(str(img_path), verbose=False, conf=0.25)
        counts = {0: 0, 1: 0, 2: 0}
        for r in results:
            if r.boxes is not None:
                for cls in r.boxes.cls.cpu().numpy():
                    counts[int(cls)] += 1
        stem = img_path.stem
        case_id = stem.split("_")[0]
        oof_results[stem] = {
            "case_id": case_id, "fold": fold,
            "vessel": counts[0], "malformed_vessel": counts[1],
            "cross_vessel": counts[2],
            "total_vessel": counts[0] + counts[1] + counts[2],
        }

case_agg = defaultdict(lambda: {"vessel": [], "malformed": [], "cross": [], "total": []})
for stem, r in oof_results.items():
    cid = r["case_id"]
    case_agg[cid]["vessel"].append(r["vessel"])
    case_agg[cid]["malformed"].append(r["malformed_vessel"])
    case_agg[cid]["cross"].append(r["cross_vessel"])
    case_agg[cid]["total"].append(r["total_vessel"])

case_predictions = {}
for cid, counts in case_agg.items():
    total = sum(counts["total"])
    cross_ratio = sum(counts["cross"]) / total if total else 0.0
    malform_ratio = sum(counts["malformed"]) / total if total else 0.0
    case_predictions[cid] = {
        "crossing_ratio_pct": cross_ratio * 100,
        "malformation_ratio_pct": malform_ratio * 100,
        "mean_vessel_per_frame": float(np.mean(counts["total"])),
        "n_frames": len(counts["total"]),
        "total_vessels": total,
    }

with open(out_base / "oof_case_predictions.json", "w") as f:
    json.dump(case_predictions, f, indent=2)
print(f"OOF saved for {len(case_predictions)} cases")
PYEOF

echo "[$(date)] EXP-2 pipeline DONE"
