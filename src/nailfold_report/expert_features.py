"""Inference-time features for the four expert routes of expert_router_v1.

Each function reproduces the table the heads were trained on:
  COL  extract_roi_quality_background_features.features, case mean
  SEG  S0 segmenter (conf 0.05, imgsz 1024) -> extract_instance_geometry
       polygon_features / pairwise_topology / image_row, case mean of numeric
       columns except img_w / img_h
  DET  Capillary-Dataset detector (conf 0.25, imgsz 640) ->
       extract_detector_local_features.per_image_stats + aggregate
A0 is the rag_heads_v1 encoder and is computed by the caller.
Images are decoded from bytes so non-ASCII paths work on Windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DET_CLASSES = {0: "bushy", 1: "crossing", 2: "hairpin", 3: "tortuous"}


def read_bgr(path: Path) -> np.ndarray:
    import cv2
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("cannot decode image")
    return im


def col_features(images: list[np.ndarray]) -> pd.Series:
    import cv2
    import extract_roi_quality_background_features as Q
    vecs = []
    for bgr in images:
        b = cv2.resize(bgr, (512, 384), interpolation=cv2.INTER_AREA)
        h, w = b.shape[:2]
        regions = [b, b[: int(.45 * h)], b[int(.35 * h):],
                   b[int(.1 * h):int(.9 * h), int(.15 * w):int(.85 * w)]]
        v = [Q.region_features(x) for x in regions]
        vecs.append(np.concatenate(v + [v[1] - v[2], v[3] - v[0]]))
    return pd.Series(np.mean(np.stack(vecs).astype(np.float32), 0))


def seg_features(model, images: list[np.ndarray], device) -> pd.Series:
    from extract_instance_geometry import image_row, pairwise_topology, polygon_features
    rows = []
    for im in images:
        res = model.predict(im, conf=0.05, imgsz=1024, verbose=False, device=device)[0]
        h, wd = res.orig_shape
        inst = []
        if res.masks is not None:
            for poly in res.masks.xy:
                f = polygon_features(np.asarray(poly))
                if f is not None:
                    inst.append(f)
        row = image_row(inst, pairwise_topology(inst, wd, h), wd, h)
        row["conf_mean"] = float(res.boxes.conf.mean()) if len(res.boxes) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).drop(columns=["img_w", "img_h"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce").mean()


def det_features(model, images: list[np.ndarray], device) -> pd.Series:
    import extract_detector_local_features as D
    rows = []
    for im in images:
        r = model.predict(im, imgsz=640, conf=0.25, device=device, verbose=False)[0]
        rows.append(dict(exam_case_id="x", image_path="x",
                         **D.per_image_stats(r, 0.25, DET_CLASSES)))
    return D.aggregate(pd.DataFrame(rows)).iloc[0].astype(float)
