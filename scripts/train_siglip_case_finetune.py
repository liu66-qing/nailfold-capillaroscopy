from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from torch import nn


CATEGORICAL = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus",
    "papilla", "sweat_duct",
)
CONTINUOUS = (
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length",
)
ENGINEERING_TOLERANCE = {
    "afferent_diameter": 2.0,
    "efferent_diameter": 2.0,
    "output_input_ratio": 0.2,
    "apex_diameter": 3.0,
    "loop_length": 30.0,
}
ORDINAL_SEVERITY = {
    "clarity": {"清晰": 0, "模糊": 1, "不清": 2},
    "capillary_count": {">=7": 0, "5--6": 1, "3--4": 2, "1--2": 3, "<1": 4},
    "crossing_ratio": {
        "<=30%": 0, "10--30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 3,
    },
    "malformation_ratio": {
        "<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3,
    },
    "blood_color": {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 3},
    "exudation": {"无": 0, "+": 1, "++": 2, "+++": 3},
    "hemorrhage": {"无": 0, "1--2": 1, "3--4": 2, ">=5": 3},
    "subpapillary_venous_plexus": {
        "不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3,
    },
    "papilla": {"波纹状": 0, "浅波纹状": 1, "平坦": 2},
    "sweat_duct": {"0--2": 0, "3--4": 1, ">=5": 2},
}


def bootstrap_interval(
    truth: list[float], predicted: list[float], metric, seed: int = 20260802,
    samples: int = 1000,
) -> list[float]:
    truth_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    if len(truth_array) < 2:
        return [math.nan, math.nan]
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        indices = generator.integers(0, len(truth_array), len(truth_array))
        estimates.append(float(metric(truth_array[indices], predicted_array[indices])))
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


@dataclass
class Case:
    case_id: str
    split: str
    paths: list[Path]
    categorical: dict[str, int]
    continuous: dict[str, float]
    label_confidence: dict[str, float]


class CaseHead(nn.Module):
    def __init__(self, dim: int, sizes: dict[str, int]) -> None:
        super().__init__()
        hidden = 384
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.attention = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(hidden, 192), nn.Tanh(), nn.Linear(192, 1)
            )
            for field in (*CATEGORICAL, *CONTINUOUS)
        })
        self.field_fuse = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden),
            nn.GELU(), nn.Dropout(0.25),
        )
        self.shared_gate = nn.Sequential(
            nn.Linear(hidden, 192), nn.Tanh(), nn.Linear(192, 4)
        )
        self.shared_fuse = nn.Sequential(
            nn.LayerNorm(hidden * 6), nn.Linear(hidden * 6, hidden),
            nn.GELU(), nn.Dropout(0.25),
        )
        self.combine = nn.ModuleDict({
            field: nn.Sequential(
                nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU()
            )
            for field in (*CATEGORICAL, *CONTINUOUS)
        })
        self.classifiers = nn.ModuleDict({k: nn.Linear(hidden, v) for k, v in sizes.items()})
        self.regressors = nn.ModuleDict({k: nn.Linear(hidden, 1) for k in CONTINUOUS})
        self.attention_regularization = torch.tensor(0.0)

    def forward(self, features: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # features: B,K,D
        encoded = self.project(features)
        mean = encoded.mean(1)
        maximum = encoded.amax(1)
        shared_attention = self.shared_gate(encoded).transpose(1, 2).softmax(dim=-1)
        shared_attended = torch.bmm(shared_attention, encoded).flatten(1)
        shared = self.shared_fuse(
            torch.cat((shared_attended, mean, maximum), dim=1)
        )
        representations = {}
        attentions = []
        for field, gate in self.attention.items():
            logits = gate(encoded).squeeze(-1)
            attention = logits.softmax(dim=-1)
            attended = torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)
            top_count = min(2, encoded.shape[1])
            top_indices = logits.topk(top_count, dim=1).indices
            top = torch.gather(
                encoded, 1, top_indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
            ).mean(1)
            field_representation = self.field_fuse(
                torch.cat((attended, top, mean, maximum), dim=1)
            )
            representations[field] = self.combine[field](
                torch.cat((shared, field_representation), dim=1)
            )
            attentions.append(attention)
        attention_matrix = torch.stack(attentions, dim=1)
        entropy = -(
            attention_matrix.clamp_min(1e-8).log() * attention_matrix
        ).sum(-1).mean() / math.log(max(encoded.shape[1], 2))
        normalized = nn.functional.normalize(attention_matrix, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        fields = similarity.shape[1]
        diversity = (
            (similarity.sum((1, 2)) - fields) / max(fields * (fields - 1), 1)
        ).mean()
        self.attention_regularization = 0.01 * entropy + 0.002 * diversity
        return (
            {k: head(representations[k]) for k, head in self.classifiers.items()},
            {k: head(representations[k]).squeeze(1) for k, head in self.regressors.items()},
        )


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, sizes: dict[str, int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = CaseHead(dim, sizes)

    def forward(self, batch: dict[str, torch.Tensor], cases: int, images: int):
        output = self.backbone.get_image_features(**batch)
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        elif not isinstance(output, torch.Tensor):
            output = output[0]
        output = nn.functional.normalize(output.float(), dim=-1).reshape(cases, images, -1)
        return self.head(output)


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.7:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.88, 1.12))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
    return image


def sample_paths(paths: list[Path], count: int, training: bool) -> list[Path]:
    # `paths` are ranked by objective image quality in `prepare`.  Training
    # samples from a small high-quality pool to retain view diversity; evaluation
    # uses exactly the same top-view policy as Windows inference.
    if len(paths) >= count:
        if training:
            return random.sample(paths, count)
        return paths[:count]
    return (paths * math.ceil(count / len(paths)))[:count]


def rank_usable_paths(paths: list[Path]) -> list[Path]:
    """Match deployment view ranking and remove unusable evidence before MIL."""
    from nailfold_report.windows_inference import quality_score

    ranked = sorted(
        ((quality_score(path), path) for path in paths),
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )
    return [path for score, path in ranked if math.isfinite(score)]


def prepare(
    labels_path: Path, folds_path: Path, files_path: Path, image_root: Path,
    test_fold: int, full_train: bool = False, min_label_confidence: float = 0.05,
    evaluation_roles_path: Path | None = None, locked_evaluation: bool = False,
    train_all_development: bool = False, class_weight_power: float = 0.5,
):
    labels = pd.read_csv(labels_path)
    folds = pd.read_csv(folds_path)
    if "conflict_field_count" in labels:
        labels = labels[labels.conflict_field_count.fillna(0).astype(int).eq(0)]
    frame = labels.merge(folds[["exam_case_id", "fold"]], on="exam_case_id")
    if evaluation_roles_path is not None:
        roles = pd.read_csv(evaluation_roles_path)[
            ["exam_case_id", "evaluation_role", "development_fold"]
        ]
        frame = frame.merge(roles, on="exam_case_id")
        if locked_evaluation:
            if train_all_development:
                frame["split"] = np.where(
                    frame.evaluation_role.eq("locked_test"), "test", "train"
                )
            else:
                frame["split"] = np.where(
                    frame.evaluation_role.eq("locked_test"), "test",
                    np.where(frame.development_fold.eq(test_fold), "val", "train"),
                )
        else:
            frame["split"] = np.where(
                frame.evaluation_role.eq("locked_test"), "excluded",
                np.where(
                    frame.development_fold.eq(test_fold), "test",
                    np.where(
                        frame.development_fold.eq((test_fold + 1) % 5),
                        "val", "train",
                    ),
                ),
            )
    else:
        frame["split"] = np.where(
            frame.fold.eq(test_fold), "test",
            np.where(frame.fold.eq((test_fold + 1) % 5), "val", "train"),
        )
    if full_train:
        frame["split"] = "train"
    vocab = {}
    for field in CATEGORICAL:
        confidence_column = f"{field}__confidence"
        usable = frame[field].notna()
        if confidence_column in frame:
            usable &= frame[confidence_column].fillna(0).ge(min_label_confidence)
        vocab[field] = sorted(frame.loc[usable, field].astype(str).unique())
    train = frame[frame.split.eq("train")]
    scales = {}
    weights = {}
    for field in CONTINUOUS:
        values = pd.to_numeric(train[field], errors="coerce")
        confidence_column = f"{field}__confidence"
        if confidence_column in train:
            # Scaling is a conditioning operation, not a way to retain noisy labels.
            # Estimate it only from reliable values; low-confidence labels still
            # participate in the weighted loss after normalization.
            values = values[train[confidence_column].fillna(0).ge(0.50)]
        scales[field] = (float(values.mean()), max(float(values.std()), 1e-6))
    for field, values in vocab.items():
        counts = train[field].astype("string").value_counts()
        raw = torch.tensor(
            [
                (len(train) / max(float(counts.get(v, 0)), 1.0))
                ** class_weight_power
                for v in values
            ]
        )
        weights[field] = raw / raw.mean()
    files = pd.read_csv(files_path)
    files = files[(files.role == "cap_image") & ~files.is_black_placeholder.astype(bool)]
    paths = {
        case_id: rank_usable_paths(
            [image_root / value for value in group.path.astype(str)]
        )
        for case_id, group in files.groupby("exam_case_id")
    }
    paths = {case_id: values for case_id, values in paths.items() if values}
    cases = []
    for row in frame.itertuples(index=False):
        if row.exam_case_id not in paths:
            continue
        categorical = {}
        continuous = {}
        label_confidence = {}
        for field, values in vocab.items():
            value = getattr(row, field)
            confidence_column = f"{field}__confidence"
            field_confidence = (
                float(getattr(row, confidence_column))
                if hasattr(row, confidence_column)
                and not pd.isna(getattr(row, confidence_column))
                else 1.0
            )
            label_confidence[field] = field_confidence
            categorical[field] = (
                values.index(str(value))
                if not pd.isna(value) and field_confidence >= min_label_confidence
                else -1
            )
        for field, (mean, std) in scales.items():
            value = pd.to_numeric(pd.Series([getattr(row, field)]), errors="coerce").iloc[0]
            confidence_column = f"{field}__confidence"
            field_confidence = (
                float(getattr(row, confidence_column))
                if hasattr(row, confidence_column)
                and not pd.isna(getattr(row, confidence_column))
                else 1.0
            )
            label_confidence[field] = field_confidence
            continuous[field] = (
                float((value - mean) / std)
                if not pd.isna(value) and field_confidence >= min_label_confidence
                else math.nan
            )
        cases.append(Case(
            row.exam_case_id, row.split, paths[row.exam_case_id],
            categorical, continuous, label_confidence,
        ))
    ordinal_severity = {
        field: torch.tensor(
            [ORDINAL_SEVERITY[field].get(value, 0) for value in values],
            dtype=torch.float32,
        )
        for field, values in vocab.items()
    }
    return cases, vocab, scales, weights, ordinal_severity


def make_inputs(cases: list[Case], processor, k: int, training: bool, device):
    images = []
    for case in cases:
        for path in sample_paths(case.paths, k, training):
            with Image.open(path) as source:
                image = source.convert("RGB")
            images.append(augment(image) if training else image)
    inputs = processor(
        images=images, return_tensors="pt", padding="max_length", max_num_patches=256
    )
    return {name: value.to(device) for name, value in inputs.items()}


def loss_function(
    prediction, cases, weights, device, scales=None, attention_regularization=None,
    ordinal_severity=None, ordinal_loss_weight: float = 0.0,
):
    categorical, continuous = prediction
    losses = []
    for field in CATEGORICAL:
        indices = [i for i, case in enumerate(cases) if case.categorical[field] >= 0]
        if indices:
            targets = torch.tensor([cases[i].categorical[field] for i in indices], device=device)
            per_case = nn.functional.cross_entropy(
                categorical[field][indices], targets, weight=weights[field].to(device),
                label_smoothing=0.04, reduction="none",
            )
            confidence = torch.tensor(
                [cases[i].label_confidence[field] for i in indices], device=device
            )
            field_loss = (
                (per_case * confidence).sum() / confidence.sum().clamp_min(1e-6)
            )
            if ordinal_loss_weight > 0 and ordinal_severity is not None:
                severity = ordinal_severity[field].to(device)
                denominator = max(float(severity.max()), 1.0)
                expected = (
                    categorical[field][indices].softmax(-1) * severity
                ).sum(-1) / denominator
                target_severity = severity[targets] / denominator
                ordinal_per_case = nn.functional.smooth_l1_loss(
                    expected, target_severity, beta=0.20, reduction="none"
                )
                field_loss = field_loss + ordinal_loss_weight * (
                    (ordinal_per_case * confidence).sum()
                    / confidence.sum().clamp_min(1e-6)
                )
            losses.append(field_loss)
    for field in CONTINUOUS:
        indices = [i for i, case in enumerate(cases) if not math.isnan(case.continuous[field])]
        if indices:
            targets = torch.tensor([cases[i].continuous[field] for i in indices], device=device)
            per_case = nn.functional.smooth_l1_loss(
                continuous[field][indices], targets, beta=0.5, reduction="none"
            )
            confidence = torch.tensor(
                [cases[i].label_confidence[field] for i in indices], device=device
            )
            losses.append((per_case * confidence).sum() / confidence.sum().clamp_min(1e-6))
    loss = torch.stack(losses).mean()
    if scales is not None:
        raw = {
            field: continuous[field] * scales[field][1] + scales[field][0]
            for field in CONTINUOUS
        }
        derived_ratio = raw["efferent_diameter"] / raw[
            "afferent_diameter"
        ].clamp_min(1.0)
        consistency = nn.functional.smooth_l1_loss(
            raw["output_input_ratio"], derived_ratio.detach(), beta=0.12
        )
        positivity = torch.stack([
            nn.functional.relu(0.5 - raw[field]).mean()
            for field in (
                "afferent_diameter", "efferent_diameter",
                "apex_diameter", "loop_length",
            )
        ]).mean()
        loss = loss + 0.05 * consistency + 0.005 * positivity
    if attention_regularization is not None:
        loss = loss + attention_regularization
    return loss


@torch.inference_mode()
def evaluate(
    model, cases, split, processor, k, batch_size, scales, device,
    minimum_confidence: float = 0.0, include_intervals: bool = False,
):
    model.eval()
    selected = [case for case in cases if case.split == split]
    truth = {field: [] for field in CATEGORICAL + CONTINUOUS}
    predicted = {field: [] for field in CATEGORICAL + CONTINUOUS}
    record_case_ids = {field: [] for field in CATEGORICAL + CONTINUOUS}
    for start in range(0, len(selected), batch_size):
        group = selected[start:start + batch_size]
        inputs = make_inputs(group, processor, k, False, device)
        categorical, continuous = model(inputs, len(group), k)
        for i, case in enumerate(group):
            for field in CATEGORICAL:
                if (
                    case.categorical[field] >= 0
                    and case.label_confidence[field] >= minimum_confidence
                ):
                    truth[field].append(case.categorical[field])
                    predicted[field].append(int(categorical[field][i].argmax()))
                    record_case_ids[field].append(case.case_id)
            for field in CONTINUOUS:
                if (
                    not math.isnan(case.continuous[field])
                    and case.label_confidence[field] >= minimum_confidence
                ):
                    mean, std = scales[field]
                    truth[field].append(case.continuous[field] * std + mean)
                    predicted[field].append(float(continuous[field][i]) * std + mean)
                    record_case_ids[field].append(case.case_id)
    metrics = {}
    for field in CATEGORICAL:
        if truth[field]:
            labels = sorted(set(truth[field]))
            recalls = recall_score(
                truth[field], predicted[field], labels=labels,
                average=None, zero_division=0,
            )
            result = {
                "n": len(truth[field]),
                "balanced_accuracy": float(balanced_accuracy_score(truth[field], predicted[field])),
                "accuracy": float(np.mean(np.asarray(truth[field]) == predicted[field])),
                "macro_f1": float(f1_score(
                    truth[field], predicted[field], average="macro", zero_division=0
                )),
                "per_class_recall": {
                    str(label): float(value) for label, value in zip(labels, recalls)
                },
            }
            if include_intervals:
                result["balanced_accuracy_ci95"] = bootstrap_interval(
                    truth[field], predicted[field], balanced_accuracy_score
                )
                result["records"] = [
                    {
                        "exam_case_id": case_id,
                        "truth": int(truth_value),
                        "prediction": int(prediction),
                    }
                    for case_id, truth_value, prediction in zip(
                        record_case_ids[field], truth[field], predicted[field]
                    )
                ]
            metrics[field] = result
    for field in CONTINUOUS:
        if truth[field]:
            errors = np.abs(np.asarray(truth[field]) - np.asarray(predicted[field]))
            tolerance = ENGINEERING_TOLERANCE[field]
            result = {
                "n": len(truth[field]),
                "mae": float(mean_absolute_error(truth[field], predicted[field])),
                "median_absolute_error": float(np.median(errors)),
                "within_engineering_tolerance": float((errors <= tolerance).mean()),
                "engineering_tolerance": tolerance,
                "normalized_mae": float(mean_absolute_error(truth[field], predicted[field]) / scales[field][1]),
            }
            if include_intervals:
                result["mae_ci95"] = bootstrap_interval(
                    truth[field], predicted[field], mean_absolute_error
                )
                result["records"] = [
                    {
                        "exam_case_id": case_id,
                        "truth": float(truth_value),
                        "prediction": float(prediction),
                    }
                    for case_id, truth_value, prediction in zip(
                        record_case_ids[field], truth[field], predicted[field]
                    )
                ]
            metrics[field] = result
    return metrics


def objective(metrics):
    classification = [metrics[x]["balanced_accuracy"] for x in CATEGORICAL if x in metrics]
    regression = [metrics[x]["normalized_mae"] for x in CONTINUOUS if x in metrics]
    return float(np.mean(classification) - 0.25 * np.mean(regression))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--images-per-case", type=int, default=4)
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--deployment-preprocess", action="store_true")
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--min-label-confidence", type=float, default=0.05)
    parser.add_argument("--eval-min-confidence", type=float, default=0.80)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--evaluation-roles", type=Path)
    parser.add_argument("--locked-evaluation", action="store_true")
    parser.add_argument("--train-development-evaluate-locked", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoModel, AutoProcessor
    device = torch.device("cuda")
    cases, vocab, scales, weights, ordinal_severity = prepare(
        args.labels, args.folds, args.files, args.image_root, args.test_fold,
        args.full_train, args.min_label_confidence,
        args.evaluation_roles, args.locked_evaluation,
        args.train_development_evaluate_locked,
        args.class_weight_power,
    )
    if args.deployment_preprocess:
        from nailfold_report.windows_inference import _target_size, preprocess_uvc_image

        class DeploymentProcessor:
            def __call__(self, images, **kwargs):
                values = np.concatenate(
                    [
                        preprocess_uvc_image(image, maximum=args.max_patches)
                        for image in images
                    ],
                    axis=0,
                )
                batch = len(images)
                mask = np.zeros((batch, args.max_patches), dtype=np.int32)
                shape_values = [
                    tuple(value // 16 for value in _target_size(
                        image.height, image.width, maximum=args.max_patches
                    ))
                    for image in images
                ]
                for row, (height, width) in enumerate(shape_values):
                    mask[row, :height * width] = 1
                shapes = np.asarray(shape_values, dtype=np.int64)
                return {
                    "pixel_values": torch.from_numpy(values),
                    "pixel_attention_mask": torch.from_numpy(mask),
                    "spatial_shapes": torch.from_numpy(shapes),
                }

        processor = DeploymentProcessor()
    else:
        processor = AutoProcessor.from_pretrained(args.model)
    backbone = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    layers = backbone.vision_model.encoder.layers
    for layer in layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name in ("post_layernorm", "head"):
        module = getattr(backbone.vision_model, name, None)
        if module:
            for parameter in module.parameters():
                parameter.requires_grad = True
    backbone.to(device)
    with torch.no_grad():
        sample = make_inputs([cases[0]], processor, 1, False, device)
        sample_output = backbone.get_image_features(**sample)
        if hasattr(sample_output, "pooler_output"):
            sample_output = sample_output.pooler_output
        elif not isinstance(sample_output, torch.Tensor):
            sample_output = sample_output[0]
        dim = sample_output.shape[-1]
    model = Model(backbone, dim, {k: len(v) for k, v in vocab.items()}).to(device)
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
    encoder_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    train = [case for case in cases if case.split == "train"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.train_development_evaluate_locked:
        if not args.locked_evaluation or args.evaluation_roles is None:
            raise ValueError(
                "--train-development-evaluate-locked requires "
                "--locked-evaluation and --evaluation-roles"
            )
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); random.shuffle(train); losses = []
            for start in range(0, len(train), args.batch_size):
                group = train[start:start + args.batch_size]
                inputs = make_inputs(group, processor, args.images_per_case, True, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(inputs, len(group), args.images_per_case)
                    loss = loss_function(
                        prediction, group, weights, device, scales,
                        model.head.attention_regularization,
                        ordinal_severity, args.ordinal_loss_weight,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            record = {"epoch": epoch, "loss": float(np.mean(losses))}
            history.append(record)
            print(json.dumps(record), flush=True)
        torch.save(model.state_dict(), args.output / "locked_candidate.pt")
        test = evaluate(
            model, cases, "test", processor, args.eval_images, args.batch_size,
            scales, device, args.eval_min_confidence, include_intervals=True,
        )
        (args.output / "metrics.json").write_text(json.dumps(
            {
                "mode": "development_train_locked_test",
                "development_cases": len(train),
                "locked_cases": sum(case.split == "test" for case in cases),
                "epochs": args.epochs,
                "test": test,
                "history": history,
                "vocabularies": vocab,
                "scales": scales,
            },
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        return
    if args.full_train:
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); random.shuffle(train); losses = []
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(train), args.batch_size):
                group = train[start:start + args.batch_size]
                inputs = make_inputs(group, processor, args.images_per_case, True, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(inputs, len(group), args.images_per_case)
                    loss = loss_function(
                        prediction, group, weights, device, scales,
                        model.head.attention_regularization,
                        ordinal_severity, args.ordinal_loss_weight,
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            scheduler.step()
            history.append({"epoch": epoch, "loss": float(np.mean(losses))})
            print(json.dumps(history[-1]), flush=True)
        torch.save(model.state_dict(), args.output / "full.pt")
        (args.output / "training.json").write_text(json.dumps(
            {"mode": "full_train", "cases": len(train), "epochs": args.epochs,
             "vocabularies": vocab, "scales": scales, "history": history},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        return
    best = -1e9
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train); losses = []
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(train), args.batch_size):
            group = train[start:start + args.batch_size]
            inputs = make_inputs(group, processor, args.images_per_case, True, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs, len(group), args.images_per_case)
                loss = loss_function(
                    prediction, group, weights, device, scales,
                    model.head.attention_regularization,
                    ordinal_severity, args.ordinal_loss_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss))
        scheduler.step()
        validation = evaluate(
            model, cases, "val", processor, args.eval_images, args.batch_size,
            scales, device, args.eval_min_confidence,
        )
        score = objective(validation)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": score, "val": validation})
        print(json.dumps({"epoch": epoch, "loss": np.mean(losses), "score": score}), flush=True)
        if score > best:
            best = score; patience = 0
            torch.save(model.state_dict(), args.output / "best.pt")
        else:
            patience += 1
            if patience >= 7:
                break
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device))
    test = evaluate(
        model, cases, "test", processor, args.eval_images, args.batch_size,
        scales, device, args.eval_min_confidence, include_intervals=True,
    )
    (args.output / "metrics.json").write_text(json.dumps(
        {"test_fold": args.test_fold, "best_val_objective": best, "test": test,
         "history": history, "vocabularies": vocab, "scales": scales},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
