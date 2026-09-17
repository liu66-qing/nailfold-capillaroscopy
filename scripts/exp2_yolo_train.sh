#!/bin/bash
# EXP-2: YOLO11m 3-class detection, 5-fold CV on GPU 1
# Runs after EXP-1 completes or independently on GPU 1
# Usage: bash /tmp/exp2_yolo_train.sh

set -e

export CUDA_VISIBLE_DEVICES=1
PYTHON=/root/miniconda3/bin/python
YOLO_BASE=/root/nailfold/data/yolo_det_3class
OUT_BASE=/root/nailfold/artifacts/experiments/exp2_yolo_det
mkdir -p $OUT_BASE

# Check if ultralytics is installed
$PYTHON -c "from ultralytics import YOLO; print('ultralytics OK')" || {
    echo "Installing ultralytics..."
    /root/miniconda3/bin/pip install ultralytics -q
}

# First, symlink images into each fold directory
echo "Setting up image symlinks..."
for fold in 0 1 2 3 4; do
    FOLD_DIR=$YOLO_BASE/fold${fold}
    # Symlink images dir into train/ and val/
    if [ ! -L "$FOLD_DIR/train/images" ] && [ ! -d "$FOLD_DIR/train/images" ]; then
        # Images are in a shared location, create symlinks based on label files
        echo "Creating image links for fold $fold..."
    fi
done

# The images need to be in fold{N}/{train,val}/images/
# We'll create symlinks from the master image dir
IMG_SRC=/root/nailfold/data/yolo_det_3class/images
for fold in 0 1 2 3 4; do
    for split in train val; do
        IMG_DST=$YOLO_BASE/fold${fold}/${split}/images
        if [ ! -d "$IMG_DST" ] || [ -z "$(ls -A $IMG_DST 2>/dev/null)" ]; then
            mkdir -p "$IMG_DST"
            # Link images that have corresponding labels
            for label in $YOLO_BASE/fold${fold}/${split}/labels/*.txt; do
                stem=$(basename "$label" .txt)
                src="$IMG_SRC/${stem}.jpg"
                if [ -f "$src" ]; then
                    ln -sf "$src" "$IMG_DST/${stem}.jpg"
                fi
            done
            echo "Fold $fold $split: $(ls $IMG_DST | wc -l) images linked"
        fi
    done
done

# Train 5 folds
for fold in 0 1 2 3 4; do
    echo "========================================"
    echo "Training fold $fold"
    echo "========================================"

    $PYTHON -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
results = model.train(
    data='/root/nailfold/data/yolo_det_3class/fold${fold}/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # CUDA_VISIBLE_DEVICES=1 makes this GPU 1
    project='$OUT_BASE',
    name='fold${fold}',
    exist_ok=True,
    patience=15,
    save=True,
    plots=True,
    workers=4,
    seed=42,
    verbose=True,
)
print(f'Fold ${fold} done. mAP50={results.results_dict.get(\"metrics/mAP50(B)\", \"N/A\")}')
"
    echo "Fold $fold training complete"
done

echo "All 5 folds done. Starting OOF inference..."

# OOF inference: for each fold, load best.pt and predict on val set
$PYTHON << 'PYEOF'
import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

out_base = Path("/root/nailfold/artifacts/experiments/exp2_yolo_det")
yolo_base = Path("/root/nailfold/data/yolo_det_3class")

from ultralytics import YOLO

# Collect OOF predictions
oof_results = {}  # {image_stem: {vessel: N, malformed: N, cross: N}}

for fold in range(5):
    model_path = out_base / f"fold{fold}" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"WARN: {model_path} not found, skipping fold {fold}")
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
            "case_id": case_id,
            "fold": fold,
            "vessel": counts[0],
            "malformed_vessel": counts[1],
            "cross_vessel": counts[2],
            "total_vessel": counts[0] + counts[1] + counts[2],
        }

# Aggregate to case level
case_agg = defaultdict(lambda: {"vessel": [], "malformed": [], "cross": [], "total": []})
for stem, r in oof_results.items():
    cid = r["case_id"]
    case_agg[cid]["vessel"].append(r["vessel"])
    case_agg[cid]["malformed"].append(r["malformed_vessel"])
    case_agg[cid]["cross"].append(r["cross_vessel"])
    case_agg[cid]["total"].append(r["total_vessel"])

# Compute ratios
case_predictions = {}
for cid, counts in case_agg.items():
    total = sum(counts["total"])
    if total == 0:
        cross_ratio = 0.0
        malform_ratio = 0.0
    else:
        cross_ratio = sum(counts["cross"]) / total
        malform_ratio = sum(counts["malformed"]) / total

    n_frames = len(counts["total"])
    mean_count_per_frame = np.mean(counts["total"])

    case_predictions[cid] = {
        "crossing_ratio": cross_ratio,
        "malformation_ratio": malform_ratio,
        "mean_vessel_per_frame": float(mean_count_per_frame),
        "n_frames": n_frames,
        "total_vessels": total,
    }

# Save
with open(out_base / "oof_case_predictions.json", "w") as f:
    json.dump(case_predictions, f, indent=2)

print(f"OOF predictions saved for {len(case_predictions)} cases")
PYEOF

echo "EXP-2 complete."
