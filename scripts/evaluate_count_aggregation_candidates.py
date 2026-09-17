"""Strict fold-inductive count post-processing search on development cases.

No model is retrained. Candidate threshold/aggregation selection uses only the
validation fold; the outer test fold is used once for OOF scoring.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from validate_count import count_instances, count_to_category


def normalize(value: object) -> str:
    return {"<1": "0", "0--0": "0"}.get(str(value), str(value))


def aggregate(values: list[int], method: str) -> int:
    x = np.asarray(values, dtype=float)
    if method == "median":
        return int(np.median(x))
    if method == "q75":
        return int(np.ceil(np.quantile(x, 0.75)))
    if method == "max":
        return int(np.max(x))
    if method == "mean":
        return int(np.rint(np.mean(x)))
    raise ValueError(method)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--roles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    roles = pd.read_csv(args.roles)
    roles.exam_case_id = roles.exam_case_id.astype(str)
    dev = roles[roles.evaluation_role.eq("development")].copy()
    folds = dev.set_index("exam_case_id").development_fold.astype(int).to_dict()
    labels = pd.read_csv(args.labels)
    id_col = "exam_case_id" if "exam_case_id" in labels else "case_id"
    truth = labels.set_index(labels[id_col].astype(str)).capillary_count.dropna().map(normalize).to_dict()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    by_case: dict[str, list[dict]] = defaultdict(list)
    for frame in predictions.values():
        case = str(frame["case_id"])
        if case in folds and case in truth:
            by_case[case].append(frame)

    cases = sorted(by_case)
    candidates = [
        {"threshold": float(t), "skeleton": bool(sk), "aggregation": agg}
        for t in np.round(np.arange(0.05, 0.501, 0.025), 3)
        for sk in (False, True)
        for agg in ("median", "q75", "max", "mean")
    ]
    def predict(case: str, candidate: dict) -> str:
        counts = [
            count_instances(
                frame, candidate["threshold"], {"normal", "abnormal"}, candidate["skeleton"]
            )
            for frame in by_case[case]
        ]
        return count_to_category(aggregate(counts, candidate["aggregation"]))

    y = np.asarray([truth[c] for c in cases])
    case_folds = np.asarray([folds[c] for c in cases])
    oof = np.empty(len(cases), dtype=object)
    selected = []
    fold_scores = []
    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        train_idx = np.flatnonzero(case_folds != test_fold)
        val_idx = np.flatnonzero(case_folds == val_fold)
        test_idx = np.flatnonzero(case_folds == test_fold)
        scores = []
        for cand in candidates:
            pred_val = np.asarray([predict(cases[i], cand) for i in val_idx])
            scores.append((balanced_accuracy_score(y[val_idx], pred_val), cand))
        score, cand = max(scores, key=lambda item: (item[0], -item[1]["threshold"], item[1]["aggregation"]))
        pred_test = np.asarray([predict(cases[i], cand) for i in test_idx])
        oof[test_idx] = pred_test
        selected.append({"fold": test_fold, "validation_fold": val_fold, "selected": cand, "validation_ba": float(score), "test_n": int(len(test_idx))})
        fold_scores.append({"fold": test_fold, "test_ba": float(balanced_accuracy_score(y[test_idx], pred_test))})

    labels_order = ["0", "1--2", "3--4", "5--6", ">=7"]
    report = {
        "schema_version": "count-aggregation-candidates-oof/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "cases": len(cases),
        "candidate_count": len(candidates),
        "selection": "validation_fold_only; outer_test_once",
        "selected_per_fold": selected,
        "fold_scores": fold_scores,
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "majority_baseline_balanced_accuracy": float(balanced_accuracy_score(y, np.repeat(pd.Series(y).mode().iloc[0], len(y)))),
        "confusion_matrix": confusion_matrix(y, oof, labels=labels_order).tolist(),
        "true_distribution": pd.Series(y).value_counts().to_dict(),
        "predicted_distribution": pd.Series(oof).value_counts().to_dict(),
        "source_predictions": str(args.predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"exam_case_id": cases, "fold": case_folds, "truth": y, "prediction": oof}).to_csv(args.output.with_suffix(".cases.csv"), index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
