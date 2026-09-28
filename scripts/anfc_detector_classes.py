#!/usr/bin/env python
"""ANFC detection stage on our data: 5-class vessel detector -> per-case class shares.

ANFC's pipeline detects every capillary and tags it normal / blur / hemo /
aggregation / abnormal. Our earlier export (instance_npz_dev2) dropped the class
key, so these tags never reached a downstream field model. This arm tests them.

Deviation, stated: their repo uses detectron2 (not vendored in third_party);
here the same ANFC-coco 5-class annotations train an ultralytics YOLO11m-seg at
1024x768 (Roboflow stretch undone), same seed/budget as the segmenter arms.

  python scripts/anfc_detector_classes.py build
  python scripts/anfc_detector_classes.py train
  python scripts/anfc_detector_classes.py infer
Writes E:/nailfold_tmp/seg_geom/ANFCDET/image_geometry.csv (per-image class
counts and shares, conf 0.25), consumed by mendeley_seg_eval.py as a sixth route.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mendeley_seg_build_base import ANFC, imread, imwrite, yolo_line  # noqa: E402

DS = "E:/nailfold_tmp/anfc_cls"
RUNS = "E:/nailfold_tmp/seg_runs"
NAMES = ["abnormal", "aggregation", "blur", "hemo", "normal"]   # coco ids 1..5


def build():
    for split, sub in (("train", "train"), ("valid", "val")):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(DS, sub, d), exist_ok=True)
        d = json.load(open(os.path.join(ANFC, split, "_annotations.coco.json")))
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            img = cv2.resize(imread(os.path.join(ANFC, split, im["file_name"])), (1024, 768),
                             interpolation=cv2.INTER_CUBIC)
            lines = []
            for a in by.get(im["id"], []):
                if not 1 <= a["category_id"] <= 5:
                    continue
                for s in a.get("segmentation") or []:
                    if isinstance(s, list) and len(s) >= 6:
                        p = np.asarray(s, float).reshape(-1, 2)
                        lines.append(str(a["category_id"] - 1) + yolo_line(p, im["width"], im["height"])[1:])
            name = im["file_name"].split("_jpg")[0]
            imwrite(os.path.join(DS, sub, "images", name + ".jpg"), img)
            open(os.path.join(DS, sub, "labels", name + ".txt"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(DS, "data.yaml"), "w") as f:
        f.write("train: %s/train/images\nval: %s/val/images\nnames:\n" % (DS, DS))
        f.write("".join("  %d: %s\n" % (i, n) for i, n in enumerate(NAMES)))
    print("built", DS)


def train(resume=False):
    from ultralytics import YOLO
    last = os.path.join(RUNS, "ANFCDET", "weights", "last.pt")
    if resume and os.path.exists(last):
        YOLO(last).train(resume=True, workers=0)
        return
    YOLO(os.path.join(ROOT, "weights/sam2/yolo11m-seg.pt")).train(
        data=os.path.join(DS, "data.yaml"), imgsz=1024, epochs=40, batch=4, workers=2,
        seed=20260927, deterministic=True, project=RUNS, name="ANFCDET", exist_ok=True,
        degrees=10, fliplr=0.5, mosaic=1.0, close_mosaic=10, plots=False, verbose=False)


def infer(conf):
    import pandas as pd
    from mendeley_seg_geometry import DEV_IX, LOCK_IX, OUT, sha256
    from ultralytics import YOLO
    w = os.path.join(RUNS, "ANFCDET", "weights", "best.pt")
    m = YOLO(w)
    ix = pd.concat([pd.read_csv(DEV_IX, dtype=str), pd.read_csv(LOCK_IX, dtype=str)],
                   ignore_index=True)
    rows = []
    for n, r in enumerate(ix.itertuples(), 1):
        im = imread(os.path.join(ROOT, "data", r.image_path))
        res = m.predict(im, conf=conf, imgsz=1024, verbose=False, device=0)[0]
        c = res.boxes.cls.cpu().numpy().astype(int)
        row = dict(exam_case_id=r.exam_case_id, image_path=r.image_path, n_inst=len(c))
        for k, nm in enumerate(NAMES):
            row["n_" + nm] = int((c == k).sum())
            row["share_" + nm] = float((c == k).mean()) if len(c) else np.nan
        rows.append(row)
        if n % 300 == 0:
            print(n, len(ix), flush=True)
    out = os.path.join(OUT, "ANFCDET")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "image_geometry.csv"), index=False)
    json.dump(dict(arm="ANFCDET", weights=w, sha256=sha256(w), conf=conf, n_images=len(rows)),
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "train", "infer"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    {"build": build, "train": lambda: train(a.resume), "infer": lambda: infer(a.conf)}[a.mode]()
