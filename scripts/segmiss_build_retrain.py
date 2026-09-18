#!/usr/bin/env python
"""Phase 4: build YOLO-seg fold datasets = existing human labels UNION AI-proposed misses.

Data provenance discipline
--------------------------
AI proposals are NEVER written into an existing annotation file and never
overwrite one. The existing full-field seg dataset (data/anfc_yolo_seg) is
copied by SYMLINK into a fresh tree under /root/autodl-tmp/segmiss_v1/, and the
AI-proposed polygons are appended as additional label lines in the NEW tree
only. Every image whose labels were augmented is recorded in
ai_augmented_images.csv so the union can be audited and reversed.

Case-level fold discipline
--------------------------
Development images inherit the fold of their case from development_fold. A case
in fold k is held out of fold k's training set, matching the original recipe so
the retrained weights can be used for fold-matched inference. locked-47 cases
never enter any split (they have no development_fold) and are asserted absent.

The human crop annotations (vascular_train_ready, median 136x54, 1.04 shapes per
image) are NOT the full-field source the original folds trained on; they are
single-vessel crops. The full-field source is data/anfc_yolo_seg.
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


def poly_to_yolo(poly, w, h):
    p = np.asarray(poly, dtype=float)
    p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)
    return "0 " + " ".join("%.6f" % v for v in p.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anfc", default="/root/autodl-tmp/nailfold/data/anfc_yolo_seg")
    ap.add_argument("--misses", required=True, help="ai_proposed_misses.jsonl")
    ap.add_argument("--yolo-polys", default="",
                    help="yolo_c025.jsonl; confident detections that must be kept "
                         "as positives, else the ~51 already-found vessels per "
                         "image would be taught as background")
    ap.add_argument("--data-root", default="/root/autodl-tmp/nailfold/data")
    ap.add_argument("--manifest",
                    default="/root/autodl-tmp/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-area", type=float, default=150.0,
                    help="drop tiny proposals; they add label noise not structure")
    ap.add_argument("--max-per-image", type=int, default=25)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])
    dev_fold = (man.dropna(subset=["development_fold"])
                   .set_index("exam_case_id")["development_fold"].astype(int))
    assert not (set(dev_fold.index) & locked)

    # group AI proposals by image, keeping the largest ones
    by_img = defaultdict(list)
    meta_img = {}
    with open(args.misses) as f:
        for line in f:
            d = json.loads(line)
            cid = str(d["exam_case_id"])
            assert cid not in locked, f"locked case {cid} in proposals"
            by_img[d["image_path"]].append(d["poly"])
            meta_img[d["image_path"]] = (cid, int(d["fold"]))
    n_miss_polys = sum(len(v) for v in by_img.values())
    print(f"proposal images={len(by_img)} total_miss_polys={n_miss_polys}")

    # union with the confident YOLO detections, tracked separately for auditing
    yolo_by_img = defaultdict(list)
    if args.yolo_polys:
        with open(args.yolo_polys) as f:
            for line in f:
                d = json.loads(line)
                cid = str(d["exam_case_id"])
                assert cid not in locked, f"locked case {cid} in yolo polys"
                yolo_by_img[d["image_path"]] = d["polys"]
                meta_img.setdefault(d["image_path"], (cid, int(d["fold"])))
        print(f"yolo images={len(yolo_by_img)} "
              f"total_yolo_polys={sum(len(v) for v in yolo_by_img.values())}")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for k in range(5):
        for sub in ("train", "val"):
            os.makedirs(f"{args.out}/fold{k}/images/{sub}", exist_ok=True)
            os.makedirs(f"{args.out}/fold{k}/labels/{sub}", exist_ok=True)

    # 1) existing human full-field annotations: present in EVERY fold's train set
    #    (they carry no project case identity, so they cannot leak a dev case)
    human = []
    for sub in ("train", "valid"):
        idir = f"{args.anfc}/{sub}/images"
        ldir = f"{args.anfc}/{sub}/labels"
        if not os.path.isdir(idir):
            continue
        for fn in sorted(os.listdir(idir)):
            stem = os.path.splitext(fn)[0]
            lp = f"{ldir}/{stem}.txt"
            if os.path.exists(lp):
                human.append((f"{idir}/{fn}", lp, stem))
    print(f"human full-field annotated images: {len(human)}")

    # 2) AI-proposed images: go to train of the folds that are NOT their own fold
    for k in range(5):
        for ip, lp, stem in human:
            dst_i = f"{args.out}/fold{k}/images/train/H_{stem}{os.path.splitext(ip)[1]}"
            dst_l = f"{args.out}/fold{k}/labels/train/H_{stem}.txt"
            if not os.path.exists(dst_i):
                os.symlink(ip, dst_i)
            shutil.copyfile(lp, dst_l)

    n_added = 0
    n_yolo_kept = 0
    all_imgs = set(by_img) | set(yolo_by_img)
    for img_rel in sorted(all_imgs):
        cid, fold = meta_img[img_rel]
        src = os.path.join(args.data_root, img_rel.replace("\\", "/"))
        if not os.path.exists(src):
            continue
        im = cv2.imread(src)
        if im is None:
            continue
        h, w = im.shape[:2]
        polys = by_img.get(img_rel, [])
        keep = [p for p in polys
                if cv2.contourArea(np.asarray(p, dtype=np.int32)) >= args.min_area]
        keep = sorted(keep, key=lambda p: -cv2.contourArea(np.asarray(p, np.int32)))
        keep = keep[:args.max_per_image]
        base = [p for p in yolo_by_img.get(img_rel, [])
                if len(p) >= 3 and cv2.contourArea(np.asarray(p, np.int32)) >= args.min_area]
        if not keep and not base:
            continue
        n_yolo_kept += len(base)
        lines = [poly_to_yolo(p, w, h) for p in base + keep]
        stem = "AI_" + img_rel.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
        ext = os.path.splitext(src)[1]
        for k in range(5):
            sub = "val" if k == fold else "train"
            # AI pseudo-labels are for TRAINING signal only; never validate on them
            if sub == "val":
                continue
            di = f"{args.out}/fold{k}/images/train/{stem}{ext}"
            dl = f"{args.out}/fold{k}/labels/train/{stem}.txt"
            if not os.path.exists(di):
                os.symlink(src, di)
            with open(dl, "w") as f:
                f.write("\n".join(lines) + "\n")
        rows.append({"image_path": img_rel, "exam_case_id": cid, "fold": fold,
                     "n_ai_miss_polys": len(keep), "n_yolo_polys": len(base),
                     "source": "AI_PROPOSED_NOT_GOLD"})
        n_added += len(keep)

    # 3) validation set per fold = the human full-field val split, untouched
    for k in range(5):
        vdir_i = f"{args.anfc}/valid/images"
        vdir_l = f"{args.anfc}/valid/labels"
        for fn in sorted(os.listdir(vdir_i)):
            stem = os.path.splitext(fn)[0]
            lp = f"{vdir_l}/{stem}.txt"
            if not os.path.exists(lp):
                continue
            di = f"{args.out}/fold{k}/images/val/V_{stem}{os.path.splitext(fn)[1]}"
            dl = f"{args.out}/fold{k}/labels/val/V_{stem}.txt"
            if not os.path.exists(di):
                os.symlink(f"{vdir_i}/{fn}", di)
            shutil.copyfile(lp, dl)
            # and remove it from that fold's train copy to avoid val leakage
            for p in (f"{args.out}/fold{k}/images/train/H_{stem}"
                      f"{os.path.splitext(fn)[1]}",
                      f"{args.out}/fold{k}/labels/train/H_{stem}.txt"):
                if os.path.exists(p):
                    os.remove(p)

    for k in range(5):
        with open(f"{args.out}/fold{k}/dataset.yaml", "w") as f:
            f.write(f"path: {args.out}/fold{k}\ntrain: images/train\n"
                    f"val: images/val\nnc: 1\nnames:\n  0: vessel\n")

    aug = pd.DataFrame(rows)
    aug.to_csv(f"{args.out}/ai_augmented_images.csv", index=False)
    summary = {
        "human_fullfield_images": len(human),
        "ai_augmented_images": int(len(aug)),
        "ai_miss_polygons_added": int(n_added),
        "yolo_pseudo_polygons_kept": int(n_yolo_kept),
        "ai_miss_per_image_mean": round(float(aug.n_ai_miss_polys.mean()), 2) if len(aug) else 0,
        "yolo_per_image_mean": round(float(aug.n_yolo_polys.mean()), 2) if len(aug) else 0,
        "target_definition": ("per dev image: confident YOLO conf=0.25 detections "
                              "UNION class-agnostic SAM misses; both are pseudo-labels"),
        "locked_cases_seen": 0,
        "provenance": ("AI proposals written ONLY into this new tree; no existing "
                       "annotation file modified or overwritten"),
        "fold_rule": ("a dev case in fold k is EXCLUDED from fold k's train set; "
                      "AI pseudo-labels never appear in any val set"),
        "val_set": "human full-field valid split only (no AI labels in validation)",
        "min_area_px": args.min_area, "max_per_image": args.max_per_image,
        "train_counts": {f"fold{k}": len(os.listdir(f'{args.out}/fold{k}/images/train'))
                         for k in range(5)},
        "val_counts": {f"fold{k}": len(os.listdir(f'{args.out}/fold{k}/images/val'))
                       for k in range(5)},
        "limitations": [
            "AI proposals are NOT gold standard and are not human verified",
            "AI labels may include non-vessel structures that passed the colour/size filter",
            "development set only; no locked-47 data anywhere in this tree",
        ],
    }
    with open(f"{args.out}/build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
