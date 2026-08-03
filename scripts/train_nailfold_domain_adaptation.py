from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.windows_inference import preprocess_uvc_image  # noqa: E402


def conservative_augment(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.6)))
    return image


def batch_inputs(images: list[Image.Image], device: torch.device) -> dict[str, torch.Tensor]:
    values = np.concatenate([preprocess_uvc_image(image) for image in images], axis=0)
    batch = len(images)
    mask = np.zeros((batch, 256), dtype=np.int32)
    mask[:, :252] = 1
    shapes = np.tile(np.array([[14, 18]], dtype=np.int64), (batch, 1))
    return {
        "pixel_values": torch.from_numpy(values).to(device),
        "pixel_attention_mask": torch.from_numpy(mask).to(device),
        "spatial_shapes": torch.from_numpy(shapes).to(device),
    }


def image_embedding(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model.get_image_features(**inputs)
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    elif not isinstance(output, torch.Tensor):
        output = output[0]
    return output.float()


class Projector(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, 512),
            nn.GELU(), nn.Linear(512, 256),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.layers(values), dim=-1)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = pd.read_csv(args.files)
    files = files[
        files.role.eq("cap_image")
        & ~files.is_black_placeholder.astype(bool)
    ]
    cases = {
        case_id: [args.image_root / path for path in group.path.astype(str)]
        for case_id, group in files.groupby("exam_case_id")
    }
    cases = {key: value for key, value in cases.items() if value}

    from transformers import AutoModel

    device = torch.device("cuda")
    student = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    for parameter in student.parameters():
        parameter.requires_grad = False
    for layer in student.vision_model.encoder.layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name in ("post_layernorm", "head"):
        module = getattr(student.vision_model, name, None)
        if module:
            for parameter in module.parameters():
                parameter.requires_grad = True
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        first = next(iter(cases.values()))[0]
        with Image.open(first) as source:
            dimension = image_embedding(
                student, batch_inputs([source.convert("RGB")], device)
            ).shape[-1]
    student_projector = Projector(dimension).to(device)
    teacher_projector = copy.deepcopy(student_projector).eval()
    for parameter in teacher_projector.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr},
        {"params": student_projector.parameters(), "lr": args.lr * 20},
    ], weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    case_ids = list(cases)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(case_ids)
        losses = []
        student.train()
        student_projector.train()
        for start in range(0, len(case_ids), args.batch_size):
            group = case_ids[start:start + args.batch_size]
            first_views, second_views = [], []
            for case_id in group:
                paths = cases[case_id]
                selected = random.sample(paths, 2) if len(paths) >= 2 else paths * 2
                with Image.open(selected[0]) as source:
                    first_views.append(conservative_augment(source))
                with Image.open(selected[1]) as source:
                    second_views.append(conservative_augment(source))
            first_inputs = batch_inputs(first_views, device)
            second_inputs = batch_inputs(second_views, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_first = student_projector(
                    image_embedding(student, first_inputs)
                )
                student_second = student_projector(
                    image_embedding(student, second_inputs)
                )
                with torch.no_grad():
                    teacher_first = teacher_projector(
                        image_embedding(teacher, first_inputs)
                    )
                    teacher_second = teacher_projector(
                        image_embedding(teacher, second_inputs)
                    )
                invariance = (
                    2.0 - (student_first * teacher_second).sum(-1)
                    - (student_second * teacher_first).sum(-1)
                ).mean()
                combined = torch.cat((student_first, student_second), dim=0)
                standard_deviation = torch.sqrt(combined.var(0) + 1e-4)
                variance = nn.functional.relu(0.04 - standard_deviation).mean()
                loss = invariance + 2.0 * variance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            progress = (epoch - 1 + (start + len(group)) / len(case_ids)) / args.epochs
            momentum = 1.0 - (1.0 - 0.996) * (math.cos(math.pi * progress) + 1.0) / 2.0
            ema_update(teacher, student, momentum)
            ema_update(teacher_projector, student_projector, momentum)
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    teacher.save_pretrained(args.output, safe_serialization=True)
    (args.output / "domain_adaptation.json").write_text(
        json.dumps({
            "cases": len(cases), "images": len(files), "epochs": args.epochs,
            "augmentation": (
                "same-size horizontal flip, brightness/contrast ±5%, "
                "Gaussian blur radius 0.1–0.6"
            ),
            "objective": "EMA cross-view consistency plus variance floor",
            "history": history,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
