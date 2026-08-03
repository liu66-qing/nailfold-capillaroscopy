from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image


def _distribution(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0] * 8
    return [
        float(values.mean()), float(values.std()), float(values.min()),
        *np.quantile(values, [0.25, 0.5, 0.75, 0.9]).astype(float),
        float(values.max()),
    ]


def geometry_image_features(image: Image.Image) -> np.ndarray:
    rgb_u8 = cv2.resize(
        np.asarray(image.convert("RGB")), (512, 384), interpolation=cv2.INTER_AREA
    )
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rgb = rgb_u8.astype(np.float32)
    chromatic_red = rgb[..., 0] / (rgb.sum(axis=2) + 1.0)
    red_excess = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
    a_channel = lab[..., 1]
    normalized = cv2.createCLAHE(2.0, (8, 8)).apply(a_channel)
    base = normalized.astype(np.float32) / 255.0
    ridge_maps = []
    for sigma in (1.0, 2.0, 3.5, 5.0, 7.0):
        smooth = cv2.GaussianBlur(base, (0, 0), sigma)
        xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        trace = xx + yy
        determinant = xx * yy - xy * xy
        delta = np.sqrt(np.maximum(trace * trace * 0.25 - determinant, 0))
        first = trace * 0.5 - delta
        second = trace * 0.5 + delta
        ridge_maps.append(np.maximum(-first, 0) * (np.abs(first) >= np.abs(second)))
    vesselness = np.max(np.stack(ridge_maps), axis=0)
    vesselness /= vesselness.max() + 1e-6
    features: list[float] = []
    for channel in (
        bgr[..., 0], bgr[..., 1], bgr[..., 2], lab[..., 0], lab[..., 1],
        lab[..., 2], hsv[..., 0], hsv[..., 1], hsv[..., 2],
        chromatic_red, red_excess, vesselness,
    ):
        features.extend(_distribution(channel.reshape(-1)))
    for quantile in (0.8, 0.85, 0.9, 0.93, 0.95):
        mask = (vesselness >= np.quantile(vesselness, quantile)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = stats[1:]
        components = components[components[:, cv2.CC_STAT_AREA] >= 4]
        areas = components[:, cv2.CC_STAT_AREA] if len(components) else np.array([])
        widths = components[:, cv2.CC_STAT_WIDTH] if len(components) else np.array([])
        heights = components[:, cv2.CC_STAT_HEIGHT] if len(components) else np.array([])
        features.extend([float(mask.mean()), float(len(components))])
        features.extend(_distribution(areas))
        features.extend(_distribution(widths))
        features.extend(_distribution(heights))
        features.extend(_distribution(heights / np.maximum(widths, 1)))
    gradients_x = cv2.Sobel(base, cv2.CV_32F, 1, 0)
    gradients_y = cv2.Sobel(base, cv2.CV_32F, 0, 1)
    features.extend(_distribution(np.hypot(gradients_x, gradients_y).reshape(-1)))
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
            widths = 2.0 * distance[local_maximum & (distance > 0)]
            features.extend([float(mask.mean()), float(widths.size)])
            features.extend(_distribution(widths))
    return np.asarray(features, dtype=np.float32)


class GeometryPredictor:
    def __init__(self, model_dir: Path) -> None:
        self.metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.models = {
            field: joblib.load(model_dir / details["file"])
            for field, details in self.metadata["models"].items()
        }

    def predict(self, paths: list[Path]) -> dict[str, dict[str, Any]]:
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                vectors.append(geometry_image_features(image))
        values = np.stack(vectors)
        case = np.concatenate((values.mean(0), values.std(0), values.max(0)))[None]
        result = {}
        for field, model in self.models.items():
            prediction = model.predict(case)[0]
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(case)[0]
                confidence = float(probabilities.max())
                value: Any = str(prediction)
                status = (
                    "predicted_geometry_route"
                    if confidence >= 0.55
                    else "predicted_geometry_low_confidence"
                )
            else:
                confidence = None
                value = float(prediction)
                status = "predicted_geometry_route"
            result[field] = {
                "value": value,
                "confidence": confidence,
                "status": status,
            }
        return result
