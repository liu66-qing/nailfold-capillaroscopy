from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from numpy.lib.format import open_memmap
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor


class VisualProjector(nn.Module):
    def __init__(self, vision_size: int, hidden_size: int) -> None:
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(vision_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.readout(values)


def import_symbol(path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def load_visual_modules(model_path: Path, device: torch.device):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    encoder_class = import_symbol(
        model_path / "modeling_hulumed_encoder.py",
        "hulumed_visual_only_encoder",
        "HulumedVisionEncoderModel",
    )
    encoder = encoder_class(config.vision_encoder_config)
    projector = VisualProjector(config.vision_encoder_config.hidden_size, config.hidden_size)

    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    wanted = {
        key: value
        for key, value in weight_map.items()
        if key.startswith("model.vision_encoder.") or key.startswith("model.mm_projector.")
    }
    shards = sorted(set(wanted.values()))
    state = {}
    for shard in shards:
        with safe_open(model_path / shard, framework="pt", device="cpu") as source:
            for key in wanted:
                if wanted[key] == shard:
                    state[key] = source.get_tensor(key)
    encoder_state = {key.removeprefix("model.vision_encoder."): value for key, value in state.items() if key.startswith("model.vision_encoder.")}
    projector_state = {key.removeprefix("model.mm_projector."): value for key, value in state.items() if key.startswith("model.mm_projector.")}
    encoder.load_state_dict(encoder_state, strict=True)
    projector.load_state_dict(projector_state, strict=True)
    return encoder.to(device=device, dtype=torch.bfloat16).eval(), projector.to(device=device, dtype=torch.bfloat16).eval(), shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-role", default="development")
    parser.add_argument("--expected-cases", type=int, default=186)
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"])
    index = index.merge(roles, on="exam_case_id", validate="many_to_one")
    index = index.loc[index.evaluation_role.eq(args.evaluation_role)].drop(columns="evaluation_role")
    if index.exam_case_id.nunique() != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, got {index.exam_case_id.nunique()}")
    if index.image_path.duplicated().any():
        raise ValueError("image index contains duplicate paths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    completed = int(json.loads(progress_path.read_text())["completed"]) if progress_path.exists() else 0
    device = torch.device("cuda:0")
    encoder, projector, loaded_shards = load_visual_modules(args.model, device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dimensions = {"vision_mean": 1152, "vision_mean_std": 2304, "projected_mean": 3584, "projected_mean_std": 7168}
    arrays = {}
    for name, dimension in dimensions.items():
        path = args.output_dir / f"{name}.npy"
        arrays[name] = open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=(len(index), dimension))

    peak_bytes = 0
    for position, row in enumerate(index.itertuples(index=False)):
        if position < completed:
            continue
        with Image.open(args.image_root / str(row.image_path)) as source:
            image = source.convert("RGB")
        inputs = processor.image_processor(images=[image], merge_size=1, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            tokens = encoder(inputs["pixel_values"].to(torch.bfloat16), inputs["grid_sizes"], inputs["merge_sizes"])
            projected = projector(tokens)
        vision_mean, vision_std = tokens.float().mean(0), tokens.float().std(0)
        projected_mean, projected_std = projected.float().mean(0), projected.float().std(0)
        arrays["vision_mean"][position] = vision_mean.cpu().numpy()
        arrays["vision_mean_std"][position] = torch.cat((vision_mean, vision_std)).cpu().numpy()
        arrays["projected_mean"][position] = projected_mean.cpu().numpy()
        arrays["projected_mean_std"][position] = torch.cat((projected_mean, projected_std)).cpu().numpy()
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        if (position + 1) % 25 == 0 or position + 1 == len(index):
            for array in arrays.values():
                array.flush()
            progress_path.write_text(json.dumps({"completed": position + 1, "total": len(index), "peak_vram_bytes": peak_bytes}) + "\n")
            print(f"{position + 1}/{len(index)} peak_vram_gib={peak_bytes / 2**30:.3f}", flush=True)

    output_index = index[["exam_case_id", "image_path"]]
    for name, dimension in dimensions.items():
        prefix = args.output_dir / name
        output_index.to_csv(prefix.with_suffix(".csv"), index=False)
        metadata = {
            "schema_version": "hulumed-visual-only-development-features/2.0",
            "evaluation_role": args.evaluation_role,
            "locked_test_examined": args.evaluation_role == "locked_test",
            "cases": args.expected_cases,
            "images": len(index),
            "feature_shape": [len(index), dimension],
            "representation": name,
            "loaded_weight_shards": loaded_shards,
            "peak_vram_gib": peak_bytes / 2**30,
            "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        }
        prefix.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
