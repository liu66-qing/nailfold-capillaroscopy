"""Handcrafted colour / sharpness features for every CAPorg frame of all 233 cases.

Same region_features recipe as extract_roi_quality_background_features.py
(RGB/LAB/HSV/gray stats, Laplacian variance, gradient, entropy, chromatic red
over full / upper / lower / centre regions plus two differences). No labels are
read. Output: artifacts/experiments/expert_routes_20260928/color/{features.npy,index.csv}

    python scripts/extract_color_quality_all233.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import cv2  # noqa: E402
import extract_roi_quality_background_features as Q  # noqa: E402

# cv2.imread cannot open non-ASCII Windows paths; decode from bytes instead
Q.cv2.imread = lambda p, *a: cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
features = Q.features

OUT = ROOT / "artifacts/experiments/expert_routes_20260928/color"


def main():
    _, ixa, _, _, _ = C.load()
    vecs = []
    for n, p in enumerate(ixa.image_path, 1):
        vecs.append(features(ROOT / "data" / p))
        if n % 200 == 0:
            print(n, len(ixa), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "features.npy", np.stack(vecs).astype(np.float32))
    ixa.to_csv(OUT / "index.csv", index=False)
    print("done", len(vecs))


if __name__ == "__main__":
    main()
