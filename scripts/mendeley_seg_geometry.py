#!/usr/bin/env python
"""Run one locally trained segmenter arm on our 233 cases and compute the same
per-image geometry as extract_instance_geometry.py (functions imported, not
copied, so the features are defined identically to the historical -0.257).

  python scripts/mendeley_seg_geometry.py --arm S0 --conf 0.05
Writes E:/nailfold_tmp/seg_geom/<arm>_c<conf>/image_geometry.csv
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from extract_instance_geometry import image_row, pairwise_topology, polygon_features  # noqa: E402

DEV_IX = os.path.join(ROOT, "artifacts/experiments/medical_encoder_transfer_20260921/"
                      "features/anchor_dinov2b_deployed/index.csv")
LOCK_IX = os.path.join(ROOT, "artifacts/experiments/locked_consumed_20260924/"
                       "features_locked/index.csv")
RUNS = "E:/nailfold_tmp/seg_runs"
OUT = "E:/nailfold_tmp/seg_geom"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    w = os.path.join(RUNS, a.arm, "weights", "best.pt")
    ix = pd.concat([pd.read_csv(DEV_IX, dtype=str), pd.read_csv(LOCK_IX, dtype=str)],
                   ignore_index=True)
    from ultralytics import YOLO
    import cv2
    model = YOLO(w)
    rows = []
    for n, r in enumerate(ix.itertuples(), 1):
        im = cv2.imdecode(np.fromfile(os.path.join(ROOT, "data", r.image_path), np.uint8), 1)
        res = model.predict(im, conf=a.conf, imgsz=1024, verbose=False, device=0)[0]
        h, wd = res.orig_shape
        inst = []
        if res.masks is not None:
            for poly in res.masks.xy:
                f = polygon_features(np.asarray(poly))
                if f is not None:
                    inst.append(f)
        row = image_row(inst, pairwise_topology(inst, wd, h), wd, h)
        row.update(exam_case_id=r.exam_case_id, image_path=r.image_path,
                   conf_mean=float(res.boxes.conf.mean()) if len(res.boxes) else np.nan)
        rows.append(row)
        if n % 300 == 0:
            print(n, len(ix), flush=True)
    out = os.path.join(OUT, "%s_c%s" % (a.arm, str(a.conf).replace(".", "")))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "image_geometry.csv"), index=False)
    json.dump(dict(arm=a.arm, weights=w, sha256=sha256(w), conf=a.conf, imgsz=1024,
                   n_images=len(rows), n_cases=int(ix.exam_case_id.nunique())),
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
