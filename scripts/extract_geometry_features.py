from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def distribution(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0] * 8
    return [
        float(values.mean()), float(values.std()), float(values.min()),
        *np.quantile(values, [0.25, 0.5, 0.75, 0.9]).astype(float),
        float(values.max()),
    ]


def image_features(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError("decode failed")
    bgr = cv2.resize(bgr, (512, 384), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Red chromaticity suppresses illumination intensity; LAB-a preserves red/green contrast.
    chromatic_red = rgb[..., 0] / (rgb.sum(axis=2) + 1.0)
    red_excess = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
    a_channel = lab[..., 1]
    normalized = cv2.createCLAHE(2.0, (8, 8)).apply(a_channel)
    base = normalized.astype(np.float32) / 255.0
    ridge_maps = []
    for sigma in (1.0, 2.0, 3.5, 5.0, 7.0):
        smooth = cv2.GaussianBlur(base, (0, 0), sigma)
        hessian_xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        hessian_yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        hessian_xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        trace = hessian_xx + hessian_yy
        determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy
        delta = np.sqrt(np.maximum(trace * trace * 0.25 - determinant, 0))
        lambda1 = trace * 0.5 - delta
        lambda2 = trace * 0.5 + delta
        ridge_maps.append(np.maximum(-lambda1, 0) * (np.abs(lambda1) >= np.abs(lambda2)))
    vesselness = np.max(np.stack(ridge_maps), axis=0)
    vesselness /= vesselness.max() + 1e-6
    features = []
    for channel in (
        bgr[..., 0], bgr[..., 1], bgr[..., 2], lab[..., 0], lab[..., 1],
        lab[..., 2], hsv[..., 0], hsv[..., 1], hsv[..., 2],
        chromatic_red, red_excess, vesselness,
    ):
        features.extend(distribution(channel.reshape(-1)))
    for quantile in (0.8, 0.85, 0.9, 0.93, 0.95):
        mask = (vesselness >= np.quantile(vesselness, quantile)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = stats[1:]
        components = components[components[:, cv2.CC_STAT_AREA] >= 4]
        areas = components[:, cv2.CC_STAT_AREA] if len(components) else np.array([])
        widths = components[:, cv2.CC_STAT_WIDTH] if len(components) else np.array([])
        heights = components[:, cv2.CC_STAT_HEIGHT] if len(components) else np.array([])
        features.extend([float(mask.mean()), float(len(components))])
        features.extend(distribution(areas))
        features.extend(distribution(widths))
        features.extend(distribution(heights))
        features.extend(distribution(heights / np.maximum(widths, 1)))
    gradients_x = cv2.Sobel(base, cv2.CV_32F, 1, 0)
    gradients_y = cv2.Sobel(base, cv2.CV_32F, 0, 1)
    features.extend(distribution(np.hypot(gradients_x, gradients_y).reshape(-1)))
    # Explicit lumen-width candidates. Subtracting a broad local background makes
    # the thresholds robust to the strong illumination/color drift in CAP images.
    width_sources = (
        red_excess - cv2.GaussianBlur(red_excess, (0, 0), 25),
        chromatic_red - cv2.GaussianBlur(chromatic_red, (0, 0), 25),
        a_channel.astype(np.float32)
        - cv2.GaussianBlur(a_channel.astype(np.float32), (0, 0), 25),
    )
    kernel = np.ones((3, 3), np.uint8)
    for source in width_sources:
        for quantile in (0.85, 0.9, 0.93, 0.95):
            mask = (source >= np.quantile(source, quantile)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            local_maximum = distance >= cv2.dilate(distance, kernel) - 1e-5
            center_widths = 2.0 * distance[(local_maximum) & (distance > 0)]
            features.extend([float(mask.mean()), float(center_widths.size)])
            features.extend(distribution(center_widths))
    return np.asarray(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = pd.read_csv(args.files)
    files = files[(files.role == "cap_image") & ~files.is_black_placeholder.astype(bool)]
    vectors, rows, errors = [], [], []
    for index, row in enumerate(files.itertuples(index=False), start=1):
        try:
            vectors.append(image_features(args.image_root / row.path))
            rows.append({"exam_case_id": row.exam_case_id, "path": row.path})
        except Exception as error:
            errors.append({"path": row.path, "error": f"{type(error).__name__}: {error}"})
        if index % 100 == 0:
            print(f"{index}/{len(files)} errors={len(errors)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(vectors)
    np.save(args.output.with_suffix(".npy"), matrix)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"images": len(rows), "errors": errors, "shape": list(matrix.shape)},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
