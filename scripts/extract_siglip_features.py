from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-num-patches", type=int, default=1152)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    split_frame = pd.read_csv(args.splits)
    allowed = set(split_frame["exam_case_id"].astype(str))
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    paths = sorted(args.data_root.glob("recovered_archive*/[0-9]*/CAPorg*.jpg"))
    paths = [
        path
        for path in paths
        if "/".join(path.relative_to(args.data_root).as_posix().split("/")[:2]) in allowed
    ]
    all_features: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    for offset in range(0, len(paths), args.batch_size):
        batch_paths = paths[offset : offset + args.batch_size]
        images: list[Image.Image] = []
        for path in batch_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = processor(
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_num_patches=args.max_num_patches,
        ).to(model.device)
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif not isinstance(features, torch.Tensor):
                features = features[0]
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        all_features.append(features.cpu().numpy())
        for path in batch_paths:
            relative = path.relative_to(args.data_root).as_posix()
            rows.append(
                {
                    "exam_case_id": "/".join(relative.split("/")[:2]),
                    "image_path": relative,
                }
            )
        print(f"{min(offset + args.batch_size, len(paths))}/{len(paths)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.concatenate(all_features, axis=0).astype(np.float16)
    np.save(args.output.with_suffix(".npy"), matrix)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    metadata = {
        "model": args.model,
        "max_num_patches": args.max_num_patches,
        "images": len(rows),
        "feature_shape": list(matrix.shape),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
