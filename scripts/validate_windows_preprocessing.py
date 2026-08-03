from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from nailfold_report.windows_inference import preprocess_uvc_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--onnx-model-dir", type=Path)
    args = parser.parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    results = []
    for path in args.images:
        with Image.open(path) as source:
            image = source.convert("RGB")
        expected = processor(
            images=[image], return_tensors="np", padding="max_length", max_num_patches=256
        )["pixel_values"]
        actual = preprocess_uvc_image(image)
        row = {
            "path": str(path),
            "shape": list(actual.shape),
            "max_abs_error": float(np.max(np.abs(expected - actual))),
            "mean_abs_error": float(np.mean(np.abs(expected - actual))),
        }
        if args.onnx_model_dir:
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(args.onnx_model_dir / "siglip2_vision.onnx"),
                providers=["CPUExecutionProvider"],
            )
            expected_features = session.run(None, {"pixel_values": expected})[0]
            actual_features = session.run(None, {"pixel_values": actual})[0]
            row["feature_max_abs_error"] = float(
                np.max(np.abs(expected_features - actual_features))
            )
            row["feature_cosine"] = float(
                np.sum(expected_features * actual_features)
                / (np.linalg.norm(expected_features) * np.linalg.norm(actual_features) + 1e-12)
            )
        results.append(row)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if max(row["max_abs_error"] for row in results) > 2.1 / 255:
        raise SystemExit("Windows preprocessing parity failed")
    if args.onnx_model_dir and min(row["feature_cosine"] for row in results) < 0.9999:
        raise SystemExit("Windows preprocessing feature parity failed")


if __name__ == "__main__":
    main()
