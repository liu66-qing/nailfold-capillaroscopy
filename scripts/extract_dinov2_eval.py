from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from numpy.lib.format import open_memmap
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--index", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--image-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--weights", type=Path, required=True); parser.add_argument("--evaluation-role", default="locked_test"); parser.add_argument("--expected-cases", type=int, required=True); args = parser.parse_args()
    index = pd.read_csv(args.index); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"]); index = index.merge(roles, on="exam_case_id", validate="many_to_one"); index = index[index.evaluation_role.eq(args.evaluation_role)].drop(columns="evaluation_role")
    if index.exam_case_id.nunique() != args.expected_cases: raise RuntimeError(f"expected {args.expected_cases} cases, got {index.exam_case_id.nunique()}")
    args.output_dir.mkdir(parents=True, exist_ok=True); model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0).cuda(0); state = torch.load(args.weights, map_location="cpu", weights_only=True); model.load_state_dict(state, strict=False); model.eval(); config = timm.data.resolve_model_data_config(model); transform = timm.data.create_transform(**config, is_training=False); output = open_memmap(args.output_dir / "features.npy", mode="w+", dtype=np.float16, shape=(len(index), 768))
    with torch.inference_mode():
        for position, row in enumerate(index.itertuples(index=False)):
            with Image.open(args.image_root / str(row.image_path)) as source: tensor = transform(source.convert("RGB")).unsqueeze(0).cuda(0)
            output[position] = model(tensor).float().cpu().numpy()[0].astype(np.float16)
            if (position + 1) % 100 == 0 or position + 1 == len(index): output.flush(); print(f"{position + 1}/{len(index)}", flush=True)
    index[["exam_case_id", "image_path"]].to_csv(args.output_dir / "index.csv", index=False); (args.output_dir / "metadata.json").write_text(json.dumps({"schema_version": "dinov2-evaluation-features/1.0", "evaluation_role": args.evaluation_role, "locked_test_examined": args.evaluation_role == "locked_test", "cases": args.expected_cases, "images": len(index), "feature_shape": [len(index), 768]}, indent=2) + "\n")


if __name__ == "__main__": main()
