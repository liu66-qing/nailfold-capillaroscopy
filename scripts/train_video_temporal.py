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
from sklearn.metrics import balanced_accuracy_score
from torch import nn


TARGETS = ("flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "microthrombus")


@dataclass
class Case:
    case_id: str
    split: str
    positions: np.ndarray
    targets: dict[str, int]


class TemporalCaseModel(nn.Module):
    def __init__(self, appearance_dim: int, flow_dim: int, sizes: dict[str, int]) -> None:
        super().__init__()
        hidden = 192
        self.appearance = nn.Sequential(
            nn.LayerNorm(appearance_dim), nn.Linear(appearance_dim, 128), nn.GELU()
        )
        self.flow = nn.Sequential(
            nn.LayerNorm(flow_dim), nn.Linear(flow_dim, 64), nn.GELU()
        )
        self.position = nn.Parameter(torch.randn(1, 32, hidden) * 0.01)
        layer = nn.TransformerEncoderLayer(
            hidden, 4, 384, dropout=0.25, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.frame_attention = nn.Sequential(nn.Linear(hidden, 96), nn.Tanh(), nn.Linear(96, 1))
        self.video_attention = nn.Sequential(nn.Linear(hidden, 96), nn.Tanh(), nn.Linear(96, 1))
        self.heads = nn.ModuleDict({field: nn.Linear(hidden, size) for field, size in sizes.items()})

    def forward(self, appearance: torch.Tensor, flow: torch.Tensor):
        # V,32,D and V,31,F. Zero motion is prepended to align with frame time.
        zero = torch.zeros(flow.shape[0], 1, flow.shape[2], device=flow.device, dtype=flow.dtype)
        motion = torch.cat((zero, flow), dim=1)
        sequence = torch.cat((self.appearance(appearance), self.flow(motion)), dim=-1)
        sequence = self.temporal(sequence + self.position[:, : sequence.shape[1]])
        frame_weights = self.frame_attention(sequence).softmax(dim=1)
        videos = (frame_weights * sequence).sum(dim=1)
        video_weights = self.video_attention(videos).softmax(dim=0)
        case = (video_weights * videos).sum(dim=0)
        return {field: head(case) for field, head in self.heads.items()}


def prepare(labels_path: Path, folds_path: Path, index: pd.DataFrame, test_fold: int):
    labels = pd.read_csv(labels_path)
    labels = labels[labels.conflict_field_count.fillna(0).astype(int).eq(0)]
    folds = pd.read_csv(folds_path)[["exam_case_id", "fold"]]
    frame = labels.merge(folds, on="exam_case_id")
    frame["split"] = np.where(
        frame.fold.eq(test_fold), "test",
        np.where(frame.fold.eq((test_fold + 1) % 5), "val", "train"),
    )
    vocab = {field: sorted(frame[field].dropna().astype(str).unique()) for field in TARGETS}
    train = frame[frame.split.eq("train")]
    weights = {}
    for field, values in vocab.items():
        counts = train[field].astype("string").value_counts()
        raw = torch.tensor([
            math.sqrt(len(train) / max(float(counts.get(value, 0)), 1.0)) for value in values
        ])
        weights[field] = raw / raw.mean()
    positions = index.groupby("exam_case_id").indices
    cases = []
    for row in frame.itertuples(index=False):
        if row.exam_case_id not in positions:
            continue
        targets = {}
        for field, values in vocab.items():
            value = getattr(row, field)
            targets[field] = values.index(str(value)) if not pd.isna(value) else -1
        cases.append(Case(row.exam_case_id, row.split, np.asarray(positions[row.exam_case_id]), targets))
    return cases, vocab, weights


def case_loss(prediction, case, weights, device):
    losses = []
    for field, target in case.targets.items():
        if target >= 0:
            losses.append(nn.functional.cross_entropy(
                prediction[field].unsqueeze(0), torch.tensor([target], device=device),
                weight=weights[field].to(device), label_smoothing=0.04,
            ))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model, cases, split, appearance, flow, device):
    model.eval()
    truth = {field: [] for field in TARGETS}
    predicted = {field: [] for field in TARGETS}
    for case in cases:
        if case.split != split:
            continue
        prediction = model(appearance[case.positions].to(device), flow[case.positions].to(device))
        for field, target in case.targets.items():
            if target >= 0:
                truth[field].append(target)
                predicted[field].append(int(prediction[field].argmax()))
    metrics = {}
    for field in TARGETS:
        if truth[field]:
            _, counts = np.unique(truth[field], return_counts=True)
            metrics[field] = {
                "n": len(truth[field]),
                "classes_present": int(len(counts)),
                "min_class_count": int(counts.min()),
                "balanced_accuracy": float(balanced_accuracy_score(truth[field], predicted[field])),
                "accuracy": float(np.mean(np.asarray(truth[field]) == predicted[field])),
            }
    return metrics


def objective(metrics):
    reliable = [
        value["balanced_accuracy"] for value in metrics.values()
        if value["classes_present"] >= 2 and value["min_class_count"] >= 2
    ]
    return float(np.mean(reliable)) if reliable else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    arrays = np.load(args.features.parent / f"{args.features.name}_sequences.npz")
    appearance = torch.from_numpy(arrays["appearance"].astype(np.float32))
    flow = torch.from_numpy(arrays["flow"].astype(np.float32))
    index = pd.read_csv(args.features.with_suffix(".csv"))
    cases, vocab, weights = prepare(args.labels, args.folds, index, args.test_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCaseModel(
        appearance.shape[-1], flow.shape[-1], {field: len(values) for field, values in vocab.items()}
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    train = [case for case in cases if case.split == "train"]
    args.output.mkdir(parents=True, exist_ok=True)
    best, patience, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train); optimizer.zero_grad(set_to_none=True)
        losses = []
        for index_case, case in enumerate(train, start=1):
            prediction = model(
                appearance[case.positions].to(device), flow[case.positions].to(device)
            )
            loss = case_loss(prediction, case, weights, device) / 8
            loss.backward(); losses.append(float(loss.detach()) * 8)
            if index_case % 8 == 0 or index_case == len(train):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        validation = evaluate(model, cases, "val", appearance, flow, device)
        score = objective(validation)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, patience = score, 0
            torch.save(model.state_dict(), args.output / "best.pt")
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device))
    test = evaluate(model, cases, "test", appearance, flow, device)
    (args.output / "metrics.json").write_text(json.dumps(
        {"test_fold": args.test_fold, "best_val_objective": best, "test": test,
         "history": history, "vocabularies": vocab}, ensure_ascii=False, indent=2
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
