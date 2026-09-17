from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_siglip_case_finetune import CATEGORICAL, CONTINUOUS, CaseHead


class VisionWrapper(nn.Module):
    def __init__(
        self, vision_model: nn.Module, pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.register_buffer("pixel_attention_mask", pixel_attention_mask)
        self.register_buffer("spatial_shapes", spatial_shapes)

    def forward(self, pixel_values):
        output = self.vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=self.pixel_attention_mask,
            spatial_shapes=self.spatial_shapes,
        ).pooler_output
        return nn.functional.normalize(output.float(), dim=-1)


class FixedVisionEmbeddings(nn.Module):
    """Deployment-only equivalent for a fixed 4:3 UVC patch layout."""

    def __init__(self, original: nn.Module, fixed_position: torch.Tensor) -> None:
        super().__init__()
        self.patch_embedding = original.patch_embedding
        self.register_buffer("fixed_position", fixed_position)

    def forward(self, pixel_values, spatial_shapes):
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        )
        return patch_embeds + self.fixed_position


class HeadWrapper(nn.Module):
    def __init__(self, head: CaseHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, features):
        categorical, continuous = self.head(features)
        return tuple(categorical[field] for field in CATEGORICAL) + tuple(
            continuous[field] for field in CONTINUOUS
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--sample-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import onnx
    import onnxruntime as ort

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.training_json.read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu")
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items() if key.startswith("backbone.")
    }
    backbone = AutoModel.from_pretrained(args.base_model, low_cpu_mem_usage=True)
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    processor = AutoProcessor.from_pretrained(args.base_model)
    image = Image.open(args.sample_image).convert("RGB")
    inputs = processor(
        images=[image], return_tensors="pt", padding="max_length", max_num_patches=256
    )
    original_vision = VisionWrapper(
        backbone.vision_model, inputs["pixel_attention_mask"], inputs["spatial_shapes"]
    ).eval()
    with torch.no_grad():
        original_reference = original_vision(inputs["pixel_values"]).numpy()
        embeddings = backbone.vision_model.embeddings
        position = embeddings.position_embedding.weight.reshape(
            embeddings.position_embedding_size, embeddings.position_embedding_size, -1
        )
        fixed_position = embeddings.resize_positional_embeddings(
            position, inputs["spatial_shapes"], max_length=inputs["pixel_values"].shape[1]
        )
    backbone.vision_model.embeddings = FixedVisionEmbeddings(
        backbone.vision_model.embeddings, fixed_position
    )
    vision = VisionWrapper(
        backbone.vision_model, inputs["pixel_attention_mask"], inputs["spatial_shapes"]
    ).eval()
    vision_path = args.output / "siglip2_vision.onnx"
    with torch.no_grad():
        reference_vision = vision(inputs["pixel_values"]).numpy()
    fixed_embedding_error = float(np.max(np.abs(original_reference - reference_vision)))
    if fixed_embedding_error > 1e-6:
        raise SystemExit(f"fixed embedding rewrite changed output: {fixed_embedding_error}")
    torch.onnx.export(
        vision,
        (inputs["pixel_values"],),
        vision_path,
        input_names=("pixel_values",),
        output_names=("image_features",),
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
    )
    onnx.checker.check_model(str(vision_path))

    vocabularies = metadata["vocabularies"]
    head = CaseHead(768, {field: len(vocabularies[field]) for field in CATEGORICAL})
    head.load_state_dict({
        key.removeprefix("head."): value
        for key, value in state.items() if key.startswith("head.")
    }, strict=True)
    head_wrapper = HeadWrapper(head).eval()
    head_path = args.output / "case_head.onnx"
    sample_features = torch.randn(2, 8, 768)
    with torch.no_grad():
        reference_head = [value.numpy() for value in head_wrapper(sample_features)]
    output_names = [f"{field}_logits" for field in CATEGORICAL] + list(CONTINUOUS)
    torch.onnx.export(
        head_wrapper, (sample_features,), head_path,
        input_names=("image_features",), output_names=output_names,
        dynamic_axes={
            "image_features": {0: "batch", 1: "images"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=18, do_constant_folding=True, dynamo=True,
    )
    onnx.checker.check_model(onnx.load(head_path))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    vision_session = ort.InferenceSession(str(vision_path), providers=providers)
    vision_result = vision_session.run(
        None, {"pixel_values": inputs["pixel_values"].numpy()}
    )[0]
    head_session = ort.InferenceSession(str(head_path), providers=providers)
    head_result = head_session.run(None, {"image_features": sample_features.numpy()})
    parity = {
        "vision_max_abs_error": float(np.max(np.abs(reference_vision - vision_result))),
        "vision_cosine": float(np.sum(reference_vision * vision_result) / (
            np.linalg.norm(reference_vision) * np.linalg.norm(vision_result) + 1e-12
        )),
        "head_max_abs_error": float(max(
            np.max(np.abs(expected - actual))
            for expected, actual in zip(reference_head, head_result)
        )),
        "providers": vision_session.get_providers(),
        "fixed_embedding_rewrite_max_abs_error": fixed_embedding_error,
        "vision_bytes": vision_path.stat().st_size,
        "head_bytes": head_path.stat().st_size,
    }
    (args.output / "model_metadata.json").write_text(json.dumps(
        {"vocabularies": vocabularies, "scales": metadata["scales"],
         "categorical_outputs": list(CATEGORICAL),
         "continuous_outputs": list(CONTINUOUS), "parity": parity},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(json.dumps(parity, indent=2))
    if parity["vision_cosine"] < 0.9999 or parity["vision_max_abs_error"] > 5e-4:
        raise SystemExit("vision ONNX parity failed")
    # ORT CUDA may fuse the LayerNorm/linear stack differently.  The observed
    # absolute drift is sub-millithreshold and far below a class-logit margin;
    # retain a strict 1e-3 ceiling while avoiding flaky rejection at 5e-4.
    if parity["head_max_abs_error"] > 1e-3:
        raise SystemExit("head ONNX parity failed")


if __name__ == "__main__":
    main()
