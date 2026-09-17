from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image

from .windows_inference import WindowsCasePredictor


DYNAMIC_FIELDS = ("flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "microthrombus")


def _aggregate(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.mean(0), values.std(0), values.max(0)))


def _flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    summary = [
        float(magnitude.mean()), float(magnitude.std()),
        *np.quantile(magnitude, [0.5, 0.75, 0.9, 0.99]).tolist(),
        float((magnitude > 0.25).mean()), float((magnitude > 0.5).mean()),
        float((magnitude > 1.0).mean()),
    ]
    height, width = magnitude.shape
    for row in range(4):
        for column in range(4):
            summary.append(float(magnitude[
                row * height // 4:(row + 1) * height // 4,
                column * width // 4:(column + 1) * width // 4,
            ].mean()))
    histogram, _ = np.histogram(
        angle, bins=np.linspace(0, 2 * np.pi, 9), weights=magnitude
    )
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-6
    summary.extend(histogram.tolist())
    return np.asarray(summary, dtype=np.float32)


def read_video_10fps(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    step = max(1, round(source_fps / 10.0))
    frames, index = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()
    if len(frames) < 4:
        raise ValueError(f"only {len(frames)} frames decoded from {path}")
    return frames


def video_vector(path: Path, predictor: WindowsCasePredictor) -> np.ndarray:
    frames = read_video_10fps(path)
    positions = np.linspace(0, len(frames) - 1, min(32, len(frames)), dtype=int)
    embeddings = np.stack([
        predictor.encode_image(Image.fromarray(frames[position])) for position in positions
    ])
    appearance = _aggregate(embeddings)
    embedding_delta = _aggregate(np.abs(np.diff(embeddings, axis=0)))
    flows = np.stack([_flow(frames[index - 1], frames[index]) for index in range(1, len(frames))])
    return np.concatenate((appearance, embedding_delta, _aggregate(flows))).astype(np.float32)


class DynamicPredictor:
    def __init__(self, model_dir: Path) -> None:
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.models = {
            field: [
                joblib.load(model_dir / f"{field}_{name}.joblib")
                for name in details["models"]
            ]
            for field, details in self.metadata.items()
        }

    def predict(
        self, video_paths: list[Path], visual_predictor: WindowsCasePredictor
    ) -> dict[str, Any]:
        videos = np.stack([video_vector(path, visual_predictor) for path in video_paths])
        case = _aggregate(videos)[None]
        result = {}
        for field in DYNAMIC_FIELDS:
            class_probabilities = {}
            for model in self.models[field]:
                probability = model.predict_proba(case)[0]
                for label, value in zip(model.classes_, probability):
                    class_probabilities.setdefault(str(label), []).append(float(value))
            averaged = {
                label: float(np.mean(values)) for label, values in class_probabilities.items()
            }
            value = max(averaged, key=averaged.get)
            details = self.metadata[field]
            result[field] = {
                "value": value, "confidence": averaged[value],
                "status": "predicted" if details["reliable_for_claim"]
                else "predicted_low_support",
                "cv_balanced_accuracy": details["cv_balanced_accuracy"],
                "training_class_counts": details["class_counts"],
            }
        return result
