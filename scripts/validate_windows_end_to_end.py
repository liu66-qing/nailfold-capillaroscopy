from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nailfold_report.windows_inference import (
    CATEGORICAL, CONTINUOUS, preprocess_uvc_image, select_views,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--onnx-model-dir", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=50)
    args = parser.parse_args()
    import onnxruntime as ort
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.base_model)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    vision = ort.InferenceSession(
        str(args.onnx_model_dir / "siglip2_vision.onnx"), providers=providers
    )
    head = ort.InferenceSession(
        str(args.onnx_model_dir / "case_head.onnx"), providers=providers
    )
    metadata = json.loads(
        (args.onnx_model_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    files = pd.read_csv(args.files)
    files = files[(files.role == "cap_image") & ~files.is_black_placeholder.astype(bool)]
    grouped = list(files.groupby("exam_case_id"))[: args.cases]
    agreements = {field: [] for field in CATEGORICAL}
    probability_deltas = {field: [] for field in CATEGORICAL}
    value_deltas = {field: [] for field in CONTINUOUS}
    feature_cosines = []
    for case_index, (case_id, group) in enumerate(grouped, start=1):
        paths = select_views([args.image_root / path for path in group.path], 8)
        reference_features, deployment_features = [], []
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            expected = processor(
                images=[image], return_tensors="np", padding="max_length",
                max_num_patches=256,
            )["pixel_values"]
            actual = preprocess_uvc_image(image)
            reference = vision.run(None, {"pixel_values": expected})[0][0]
            deployment = vision.run(None, {"pixel_values": actual})[0][0]
            reference_features.append(reference)
            deployment_features.append(deployment)
            feature_cosines.append(float(
                np.dot(reference, deployment)
                / (np.linalg.norm(reference) * np.linalg.norm(deployment) + 1e-12)
            ))
        reference_output = head.run(None, {
            "image_features": np.stack(reference_features)[None].astype(np.float32)
        })
        deployment_output = head.run(None, {
            "image_features": np.stack(deployment_features)[None].astype(np.float32)
        })
        for index, field in enumerate(CATEGORICAL):
            reference = reference_output[index][0]
            deployment = deployment_output[index][0]
            agreements[field].append(int(reference.argmax() == deployment.argmax()))
            probability_deltas[field].append(float(np.max(np.abs(reference - deployment))))
        for offset, field in enumerate(CONTINUOUS, start=len(CATEGORICAL)):
            standard_deviation = metadata["scales"][field][1]
            value_deltas[field].append(float(
                abs(reference_output[offset][0] - deployment_output[offset][0])
                * standard_deviation
            ))
        print(f"{case_index}/{len(grouped)} {case_id}", flush=True)
    result = {
        "cases": len(grouped),
        "images": len(feature_cosines),
        "minimum_feature_cosine": min(feature_cosines),
        "mean_feature_cosine": float(np.mean(feature_cosines)),
        "categorical_agreement": {
            field: float(np.mean(values)) for field, values in agreements.items()
        },
        "maximum_logit_delta": {
            field: max(values) for field, values in probability_deltas.items()
        },
        "continuous_mae_between_preprocessors": {
            field: float(np.mean(values)) for field, values in value_deltas.items()
        },
        "continuous_max_delta": {
            field: max(values) for field, values in value_deltas.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if min(result["categorical_agreement"].values()) < 0.99:
        raise SystemExit("categorical preprocessing regression failed")
    if max(result["continuous_max_delta"].values()) > 0.5:
        raise SystemExit("continuous preprocessing regression failed")


if __name__ == "__main__":
    main()
