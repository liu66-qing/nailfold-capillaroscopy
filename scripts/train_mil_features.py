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
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from torch import nn


CATEGORICAL_FIELDS = (
    "clarity",
    "capillary_count",
    "crossing_ratio",
    "malformation_ratio",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
    "sweat_duct",
)
CONTINUOUS_FIELDS = (
    "afferent_diameter",
    "efferent_diameter",
    "output_input_ratio",
    "apex_diameter",
    "loop_length",
)


@dataclass
class Case:
    case_id: str
    split: str
    features: torch.Tensor
    categorical: dict[str, int]
    continuous: dict[str, float]


class MultiHeadMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        categorical_sizes: dict[str, int],
        hidden_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, attention_heads),
        )
        pooled_dim = hidden_dim * (attention_heads + 2)
        self.case_encoder = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.categorical_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, size) for field, size in categorical_sizes.items()}
        )
        self.continuous_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, 1) for field in CONTINUOUS_FIELDS}
        )

    def forward(self, bag: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        encoded = self.encoder(bag)
        attention = self.attention(encoded).transpose(0, 1).softmax(dim=1)
        attentive = attention @ encoded
        pooled = torch.cat(
            [attentive.flatten(), encoded.mean(dim=0), encoded.max(dim=0).values], dim=0
        )
        representation = self.case_encoder(pooled)
        categorical = {
            field: head(representation) for field, head in self.categorical_heads.items()
        }
        continuous = {
            field: head(representation).squeeze(-1)
            for field, head in self.continuous_heads.items()
        }
        return categorical, continuous


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cases(
    matrix: np.ndarray,
    index: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[
    list[Case],
    dict[str, list[str]],
    dict[str, tuple[float, float]],
    dict[str, torch.Tensor],
]:
    frame = labels.merge(splits[["exam_case_id", "split"]], on="exam_case_id")
    frame = frame[frame["conflict_field_count"].fillna(0).astype(int) == 0].copy()
    train = frame[frame["split"] == "train"]

    vocabularies: dict[str, list[str]] = {}
    class_weights: dict[str, torch.Tensor] = {}
    for field in CATEGORICAL_FIELDS:
        vocabulary = sorted(train[field].dropna().astype(str).unique())
        vocabularies[field] = vocabulary
        counts = train[field].astype("string").value_counts()
        weights = [math.sqrt(len(train) / max(float(counts.get(value, 0)), 1.0)) for value in vocabulary]
        weight = torch.tensor(weights, dtype=torch.float32)
        class_weights[field] = weight / weight.mean()

    scales: dict[str, tuple[float, float]] = {}
    for field in CONTINUOUS_FIELDS:
        values = pd.to_numeric(train[field], errors="coerce")
        mean = float(values.mean())
        std = max(float(values.std()), 1e-6)
        scales[field] = (mean, std)

    labels_by_case = frame.set_index("exam_case_id")
    positions_by_case = index.groupby("exam_case_id").indices
    cases: list[Case] = []
    for case_id, positions in positions_by_case.items():
        if case_id not in labels_by_case.index:
            continue
        row = labels_by_case.loc[case_id]
        categorical: dict[str, int] = {}
        for field, vocabulary in vocabularies.items():
            value = None if pd.isna(row[field]) else str(row[field])
            categorical[field] = vocabulary.index(value) if value in vocabulary else -1
        continuous: dict[str, float] = {}
        for field, (mean, std) in scales.items():
            value = pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]
            continuous[field] = float((value - mean) / std) if not pd.isna(value) else math.nan
        cases.append(
            Case(
                case_id=str(case_id),
                split=str(row["split"]),
                features=torch.from_numpy(matrix[np.asarray(positions)].astype(np.float32)),
                categorical=categorical,
                continuous=continuous,
            )
        )
    return cases, vocabularies, scales, class_weights


def case_loss(
    model: MultiHeadMIL,
    case: Case,
    class_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    categorical, continuous = model(case.features.to(device))
    losses: list[torch.Tensor] = []
    for field, target in case.categorical.items():
        if target >= 0:
            losses.append(
                nn.functional.cross_entropy(
                    categorical[field].unsqueeze(0),
                    torch.tensor([target], device=device),
                    weight=class_weights[field].to(device),
                    label_smoothing=0.04,
                )
            )
    for field, target in case.continuous.items():
        if not math.isnan(target):
            losses.append(
                nn.functional.smooth_l1_loss(
                    continuous[field], torch.tensor(target, device=device), beta=0.5
                )
            )
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(
    model: MultiHeadMIL,
    cases: list[Case],
    split: str,
    scales: dict[str, tuple[float, float]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    selected = [case for case in cases if case.split == split]
    categorical_true: dict[str, list[int]] = {field: [] for field in CATEGORICAL_FIELDS}
    categorical_pred: dict[str, list[int]] = {field: [] for field in CATEGORICAL_FIELDS}
    continuous_true: dict[str, list[float]] = {field: [] for field in CONTINUOUS_FIELDS}
    continuous_pred: dict[str, list[float]] = {field: [] for field in CONTINUOUS_FIELDS}
    for case in selected:
        categorical, continuous = model(case.features.to(device))
        for field, target in case.categorical.items():
            if target >= 0:
                categorical_true[field].append(target)
                categorical_pred[field].append(int(categorical[field].argmax()))
        for field, target in case.continuous.items():
            if not math.isnan(target):
                mean, std = scales[field]
                continuous_true[field].append(target * std + mean)
                continuous_pred[field].append(float(continuous[field]) * std + mean)
    result: dict[str, dict[str, float]] = {}
    for field in CATEGORICAL_FIELDS:
        truth, prediction = categorical_true[field], categorical_pred[field]
        if truth:
            result[field] = {
                "n": len(truth),
                "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                "accuracy": float(np.mean(np.asarray(truth) == np.asarray(prediction))),
            }
    for field in CONTINUOUS_FIELDS:
        truth, prediction = continuous_true[field], continuous_pred[field]
        if truth:
            result[field] = {
                "n": len(truth),
                "mae": float(mean_absolute_error(truth, prediction)),
                "normalized_mae": float(mean_absolute_error(truth, prediction) / scales[field][1]),
            }
    return result


def validation_objective(metrics: dict[str, dict[str, float]]) -> float:
    categorical = [
        value["balanced_accuracy"]
        for field, value in metrics.items()
        if field in CATEGORICAL_FIELDS
    ]
    continuous = [
        value["normalized_mae"]
        for field, value in metrics.items()
        if field in CONTINUOUS_FIELDS
    ]
    return float(np.mean(categorical) - 0.25 * np.mean(continuous))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--folds-file", type=Path)
    parser.add_argument("--test-fold", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    args = parser.parse_args()

    matrix = np.load(args.features.with_suffix(".npy"))
    index = pd.read_csv(args.features.with_suffix(".csv"))
    labels = pd.read_csv(args.labels)
    splits = pd.read_csv(args.splits)
    if args.folds_file is not None:
        if args.test_fold is None:
            raise ValueError("--test-fold is required with --folds-file")
        folds = pd.read_csv(args.folds_file)[["exam_case_id", "fold"]]
        fold_count = int(folds["fold"].max()) + 1
        validation_fold = (args.test_fold + 1) % fold_count
        folds["split"] = np.where(
            folds["fold"] == args.test_fold,
            "test",
            np.where(folds["fold"] == validation_fold, "val", "train"),
        )
        splits = folds
    cases, vocabularies, scales, class_weights = build_cases(
        matrix, index, labels, splits
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, object] = {
        "cases": {name: sum(case.split == name for case in cases) for name in ("train", "val", "test")},
        "vocabularies": vocabularies,
        "scales": scales,
        "seeds": {},
    }
    train_cases = [case for case in cases if case.split == "train"]

    for seed in args.seeds:
        set_seed(seed)
        model = MultiHeadMIL(matrix.shape[1], {k: len(v) for k, v in vocabularies.items()}).to(
            device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_objective = -float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            random.shuffle(train_cases)
            optimizer.zero_grad(set_to_none=True)
            for index_case, case in enumerate(train_cases, start=1):
                loss = case_loss(model, case, class_weights, device) / 8
                loss.backward()
                if index_case % 8 == 0 or index_case == len(train_cases):
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            val_metrics = evaluate(model, cases, "val", scales, device)
            objective = validation_objective(val_metrics)
            if objective > best_objective + 1e-4:
                best_objective = objective
                best_epoch = epoch
                stale = 0
                best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
        assert best_state is not None
        model.load_state_dict(best_state)
        seed_result = {
            "best_epoch": best_epoch,
            "validation_objective": best_objective,
            "val": evaluate(model, cases, "val", scales, device),
            "test": evaluate(model, cases, "test", scales, device),
        }
        all_results["seeds"][str(seed)] = seed_result
        torch.save(
            {
                "state_dict": best_state,
                "vocabularies": vocabularies,
                "scales": scales,
                "seed": seed,
            },
            args.output_dir / f"mil_seed_{seed}.pt",
        )
        print(json.dumps({"seed": seed, **seed_result}, ensure_ascii=False), flush=True)

    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
