"""
Train the locked-safe 3-class vessel detector (stage S2).

Purpose
-------
The only reason to train this detector is `malformation_ratio`. The oracle
test (scripts/oracle_crossing_human_boxes.py) showed that the ratio computed
from HUMAN boxes reaches AUROC 0.749 [0.585, 0.894] against our clinical
label, while `crossing_ratio` reaches 0.639 [0.464, 0.808] -- CI spanning
chance -- so crossing is not pursued. 0.749 is a CEILING: it is what a
perfect detector would score. This run can only approach it, never exceed it.

Domain gaps that are handled here deliberately, because they were measured
before training rather than discovered after:
  - brightness: train images (血管数据集) mean HSV value 0.424, evaluation
    images (recovered_archive) 0.618  -> hsv_v augmentation is widened.
  - orientation: 26/60 sampled evaluation images are 680x750 portrait while
    training images are landscape -> flips stay on, and rotation is allowed.
  - saturation is NOT a problem between these two (0.448 vs 0.377), unlike
    the HF dataset (0.097), which is why HF is only ever an ablation arm.

Governance
----------
- Trains only on data/det_stage_lockedsafe, built by
  scripts/build_locked_safe_det_folds.py, which excludes 32 known-locked and
  24 near-duplicate-of-locked original_ids and keeps augmentation groups
  whole. This script re-asserts that no excluded id is present before it
  starts, so a stale stage directory cannot silently reintroduce one.
- No locked-47 image is read. No image leaves the machine.
- mAP here is an INTERNAL detector metric on augmented copies of the same
  nailfolds; it is NOT a product number and must never be quoted as one.
  The lesson is already on record: a segmenter reached mAP 0.952 while the
  downstream clinical sign was reversed.
"""

import os
import re
import csv
import json
import time
import argparse

ROOT = r"E:\甲劈微循环"
STAGE = os.path.join(ROOT, "data", "det_stage_lockedsafe")
EXCLUDE = os.path.join(
    ROOT, "artifacts", "evidence", "leakcheck_unmapped_20260920",
    "exclude_original_ids.txt")
MAPPING = os.path.join(
    ROOT, "artifacts", "audits", "vascular_dataset_governance_20260830",
    "source_case_mapping.csv")
OUT_DIR = os.path.join(ROOT, "artifacts", "experiments", "det_malformation_20260920")
WEIGHTS = os.path.join(ROOT, "weights", "yolo11s.pt")


def excluded_ids():
    import pandas as pd
    mp = pd.read_csv(MAPPING)
    mp["original_id"] = mp["original_id"].astype(str)
    ids = set(mp.loc[mp["source_mapping_status"] == "EXACT_RECOVERED_LOCKED",
                     "original_id"])
    if os.path.exists(EXCLUDE):
        with open(EXCLUDE, encoding="utf-8") as fh:
            ids |= {ln.strip() for ln in fh
                    if ln.strip() and not ln.startswith("#")}
    return ids


def assert_stage_clean(fold):
    """Refuse to train if an excluded original_id is present in the stage."""
    bad = excluded_ids()
    seen = 0
    for sp in ("train", "val"):
        d = os.path.join(STAGE, f"fold{fold}", sp, "images")
        for fn in os.listdir(d):
            oid = fn.split("_")[0]
            seen += 1
            if oid in bad:
                raise AssertionError(
                    f"excluded original_id {oid} present in fold{fold}/{sp} "
                    f"({fn}) -- rebuild the stage")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=WEIGHTS)
    ap.add_argument("--tag", default="s768")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    n = assert_stage_clean(args.fold)
    print(f"[gate] fold{args.fold} stage clean, {n} images, "
          f"{len(excluded_ids())} ids on the exclusion list")
    if args.check_only:
        return

    # Ultralytics' AMP check downloads a probe checkpoint (yolo26n.pt) on
    # first run. GitHub asset downloads are effectively blocked on this
    # machine (~8 KB/s, SSL handshake timeouts), so the check would stall
    # training for ~11 minutes per run. AMP itself is kept on; only the
    # download-dependent self-check is skipped.
    from ultralytics import YOLO
    import ultralytics.engine.trainer as _trainer
    # The trainer does `from ultralytics.utils.checks import check_amp` at
    # module level and calls that local name, so patching the checks module
    # after import is too late -- the reference must be replaced here.
    _trainer.check_amp = lambda *a, **k: True
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"fold{args.fold}_{args.tag}"
    t0 = time.time()

    model = YOLO(args.model)
    model.train(
        data=os.path.join(STAGE, f"fold{args.fold}", "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=OUT_DIR,
        name=name,
        exist_ok=True,
        seed=20260920,
        deterministic=True,
        val=True,
        plots=False,
        # the source pool is ALREADY 20x augmented per original, so heavy
        # geometric augmentation on top mostly wastes epochs. What is kept is
        # aimed at the two measured domain gaps only.
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.5,      # train 0.424 vs eval 0.618 mean HSV value
        degrees=10.0,   # eval set contains portrait frames
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.2,
        mosaic=0.5,
        mixup=0.0,
        patience=15,
        amp=True,
        cache=False,
    )

    res = model.val(data=os.path.join(STAGE, f"fold{args.fold}", "dataset.yaml"),
                    imgsz=args.imgsz, batch=args.batch, device=0, plots=False)
    out = {
        "fold": args.fold, "tag": args.tag, "epochs": args.epochs,
        "imgsz": args.imgsz, "batch": args.batch,
        "weights_init": os.path.basename(args.model),
        "minutes": round((time.time() - t0) / 60, 1),
        "gpu": torch.cuda.get_device_name(0),
        "map50": round(float(res.box.map50), 4),
        "map50_95": round(float(res.box.map), 4),
        "per_class_map50": {k: round(float(v), 4) for k, v in zip(
            ["vessel", "malformed_vessel", "cross_vessel"],
            list(res.box.ap50))},
        "note": ("INTERNAL detector metric on augmented copies of the same "
                 "nailfolds. NOT a product number. The field verdict comes "
                 "only from the case-level malformation_ratio evaluation on "
                 "the 119 DEV cases that have no human boxes."),
        "governance": {"locked_cases_seen": 0, "images_transmitted": 0},
    }
    with open(os.path.join(OUT_DIR, f"{name}_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
