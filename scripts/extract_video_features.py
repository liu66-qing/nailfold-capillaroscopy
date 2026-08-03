from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=32)
    parser.add_argument("--max-num-patches", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-sequences", action="store_true")
    parser.add_argument("--deployment-preprocess", action="store_true")
    return parser.parse_args()


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def flow_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    summary = [
        float(magnitude.mean()),
        float(magnitude.std()),
        *np.quantile(magnitude, [0.5, 0.75, 0.9, 0.99]).astype(float).tolist(),
        float((magnitude > 0.25).mean()),
        float((magnitude > 0.5).mean()),
        float((magnitude > 1.0).mean()),
    ]
    height, width = magnitude.shape
    for row in range(4):
        for column in range(4):
            cell = magnitude[
                row * height // 4 : (row + 1) * height // 4,
                column * width // 4 : (column + 1) * width // 4,
            ]
            summary.append(float(cell.mean()))
    bins = np.linspace(0, 2 * np.pi, 9)
    histogram, _ = np.histogram(angle, bins=bins, weights=magnitude)
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-6
    summary.extend(histogram.tolist())
    return np.asarray(summary, dtype=np.float32)


def aggregate_time(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0), values.max(axis=0)], axis=0
    )


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    if args.deployment_preprocess:
        import torch
        from nailfold_report.windows_inference import preprocess_uvc_image

        class DeploymentProcessor:
            def __call__(self, images, **kwargs):
                values = np.concatenate(
                    [preprocess_uvc_image(image) for image in images], axis=0
                )
                batch = len(images)
                mask = np.zeros((batch, 256), dtype=np.int32)
                mask[:, :252] = 1
                shapes = np.tile(np.array([[14, 18]], dtype=np.int64), (batch, 1))
                return {
                    "pixel_values": torch.from_numpy(values),
                    "pixel_attention_mask": torch.from_numpy(mask),
                    "spatial_shapes": torch.from_numpy(shapes),
                }

        processor = DeploymentProcessor()
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in state.items() if key.startswith("backbone.")
        }
        model.load_state_dict(backbone_state, strict=True)
    model.eval()
    paths = sorted(args.video_root.glob("recovered_archive*/*/*.mp4"))
    if not paths:
        raise SystemExit(
            f"no videos found below {args.video_root}; expected "
            "recovered_archive*/<case_id>/*.mp4"
        )
    features: list[np.ndarray] = []
    appearance_sequences: list[np.ndarray] = []
    flow_sequences: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for index, path in enumerate(paths, start=1):
        try:
            frames = read_video(path)
            if len(frames) < 4:
                raise ValueError(f"only {len(frames)} decoded frames")
            sample_indices = np.linspace(
                0, len(frames) - 1, min(args.sample_frames, len(frames)), dtype=int
            )
            images = [Image.fromarray(frames[position]) for position in sample_indices]
            inputs = processor(
                images=images,
                return_tensors="pt",
                padding="max_length",
                max_num_patches=args.max_num_patches,
            )
            inputs = {
                name: value.to(model.device) for name, value in inputs.items()
            }
            with torch.inference_mode():
                embeddings = model.get_image_features(**inputs)
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
                elif not isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings[0]
                embeddings = torch.nn.functional.normalize(
                    embeddings.float(), dim=-1
                ).cpu().numpy()
            appearance = aggregate_time(embeddings)
            embedding_delta = aggregate_time(np.abs(np.diff(embeddings, axis=0)))
            flows = np.stack(
                [
                    flow_vector(frames[position - 1], frames[position])
                    for position in range(1, len(frames))
                ]
            )
            flow_features = aggregate_time(flows)
            if args.save_sequences:
                sampled_flows = np.stack(
                    [
                        flow_vector(
                            frames[sample_indices[position - 1]],
                            frames[sample_indices[position]],
                        )
                        for position in range(1, len(sample_indices))
                    ]
                )
                # Fixed-length sequences make the temporal branch compact.
                while len(embeddings) < args.sample_frames:
                    embeddings = np.concatenate((embeddings, embeddings[-1:]), axis=0)
                while len(sampled_flows) < args.sample_frames - 1:
                    sampled_flows = np.concatenate((sampled_flows, sampled_flows[-1:]), axis=0)
            vector = np.concatenate([appearance, embedding_delta, flow_features])
            relative = path.relative_to(args.video_root).as_posix()
            features.append(vector.astype(np.float16))
            if args.save_sequences:
                appearance_sequences.append(embeddings[: args.sample_frames].astype(np.float16))
                flow_sequences.append(sampled_flows[: args.sample_frames - 1].astype(np.float16))
            rows.append(
                {
                    "exam_case_id": "/".join(relative.split("/")[:2]),
                    "video_path": relative,
                    "decoded_frames": len(frames),
                    "sampled_frames": len(sample_indices),
                }
            )
        except Exception as error:
            errors.append(
                {
                    "video_path": path.relative_to(args.video_root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if index % 10 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} errors={len(errors)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        args.output.with_suffix(".json").write_text(
            json.dumps(
                {"videos": 0, "errors": errors, "feature_shape": None},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"all {len(paths)} videos failed; see {args.output.with_suffix('.json')}")
    matrix = np.stack(features)
    np.save(args.output.with_suffix(".npy"), matrix)
    if args.save_sequences:
        np.savez_compressed(
            args.output.parent / f"{args.output.name}_sequences.npz",
            appearance=np.stack(appearance_sequences),
            flow=np.stack(flow_sequences),
        )
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "videos": len(rows),
                "errors": errors,
                "feature_shape": list(matrix.shape),
                "appearance_sequence_shape": (
                    list(np.stack(appearance_sequences).shape) if args.save_sequences else None
                ),
                "flow_sequence_shape": (
                    list(np.stack(flow_sequences).shape) if args.save_sequences else None
                ),
                "model": str(args.model),
                "sample_frames": args.sample_frames,
                "max_num_patches": args.max_num_patches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
