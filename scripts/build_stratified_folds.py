from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


STRATIFY_FIELDS = (
    "archive_id",
    "overall_assessment",
    "clarity",
    "capillary_count",
    "crossing_ratio",
    "malformation_ratio",
    "blood_color",
    "exudation",
    "subpapillary_venous_plexus",
    "papilla",
)


def stable_tie(seed: int, group: str, fold: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{group}:{fold}".encode()).digest()[:8], "big"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    cases = pd.read_csv(args.splits)
    labels = pd.read_csv(args.labels)
    frame = cases.merge(labels, on="exam_case_id", suffixes=("", "_label"))
    if "conflict_field_count" in frame:
        frame = frame[
            frame["conflict_field_count"].fillna(0).astype(int) == 0
        ].copy()

    tokens_by_case: dict[str, list[str]] = {}
    token_totals: Counter[str] = Counter()
    for row in frame.to_dict("records"):
        tokens: list[str] = []
        for field in STRATIFY_FIELDS:
            value = row.get(field)
            if value is not None and not pd.isna(value):
                token = f"{field}={value}"
                tokens.append(token)
                token_totals[token] += 1
        tokens_by_case[str(row["exam_case_id"])] = tokens

    all_tokens = sorted(token_totals)
    token_index = {token: index for index, token in enumerate(all_tokens)}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in frame.to_dict("records"):
        groups[str(row["duplicate_group"])].append(str(row["exam_case_id"]))

    group_vectors: dict[str, np.ndarray] = {}
    rarity: dict[str, float] = {}
    for group, members in groups.items():
        vector = np.zeros(len(all_tokens), dtype=np.float64)
        for member in members:
            for token in tokens_by_case[member]:
                vector[token_index[token]] += 1
        group_vectors[group] = vector
        rarity[group] = sum(
            vector[token_index[token]] / token_totals[token]
            for token in all_tokens
            if vector[token_index[token]] > 0
        )

    ordered = sorted(
        groups,
        key=lambda group: (-rarity[group], -len(groups[group]), group),
    )
    target_vector = np.asarray([token_totals[token] for token in all_tokens]) / args.folds
    target_size = len(frame) / args.folds
    fold_vectors = [np.zeros(len(all_tokens), dtype=np.float64) for _ in range(args.folds)]
    fold_sizes = [0 for _ in range(args.folds)]
    fold_by_group: dict[str, int] = {}

    for group in ordered:
        vector = group_vectors[group]
        size = len(groups[group])
        candidates: list[tuple[float, int, int]] = []
        active = vector > 0
        for fold in range(args.folds):
            token_load = np.mean(
                (fold_vectors[fold][active] + vector[active])
                / (target_vector[active] + 0.5)
            )
            size_load = (fold_sizes[fold] + size) / target_size
            capacity_penalty = max(0.0, size_load - 1.08) * 100
            score = float(0.75 * token_load + 0.25 * size_load + capacity_penalty)
            candidates.append((score, stable_tie(args.seed, group, fold), fold))
        _, _, selected = min(candidates)
        fold_by_group[group] = selected
        fold_vectors[selected] += vector
        fold_sizes[selected] += size

    frame["fold"] = frame["duplicate_group"].astype(str).map(fold_by_group)
    frame["fold_seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    report: dict[str, object] = {
        "cases": len(frame),
        "duplicate_groups": len(groups),
        "fold_sizes": {str(i): fold_sizes[i] for i in range(args.folds)},
        "distributions": {},
    }
    for field in STRATIFY_FIELDS:
        table = pd.crosstab(frame[field], frame["fold"], dropna=False)
        report["distributions"][field] = {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in table.iterrows()
        }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "distributions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
