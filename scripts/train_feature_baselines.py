from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONTINUOUS_FIELDS = {
    "afferent_diameter",
    "efferent_diameter",
    "output_input_ratio",
    "apex_diameter",
    "loop_length",
    "flow_speed_um_s",
    "morphology_score",
    "flow_score",
    "periloop_score",
    "total_score",
}
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def numeric_target(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    values = [float(item) for item in NUMBER.findall(str(value).replace(",", ""))]
    if not values:
        return np.nan
    return float(np.mean(values[:2]))


def aggregate_features(matrix: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, group in index.groupby("exam_case_id", sort=True):
        values = matrix[group.index.to_numpy()].astype(np.float32)
        vector = np.concatenate([values.mean(0), values.std(0), values.max(0)])
        rows.append({"exam_case_id": case_id, "features": vector})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    matrix = np.load(args.features.with_suffix(".npy"))
    index = pd.read_csv(args.features.with_suffix(".csv"))
    aggregated = aggregate_features(matrix, index)
    labels = pd.read_csv(args.labels)
    splits = pd.read_csv(args.splits)[["exam_case_id", "split"]]
    frame = aggregated.merge(labels, on="exam_case_id").merge(splits, on="exam_case_id")
    x = np.stack(frame["features"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}

    excluded = {
        "exam_case_id",
        "features",
        "split",
        "report_copy_predictions",
        "conflict_field_count",
        "auto_label_status",
    }
    for field in [column for column in labels.columns if column not in excluded]:
        if field in CONTINUOUS_FIELDS:
            y = frame[field].map(numeric_target).to_numpy()
            valid = ~np.isnan(y)
            train = valid & (frame["split"].to_numpy() == "train")
            test = valid & (frame["split"].to_numpy() == "test")
            if train.sum() < 20 or test.sum() < 5:
                continue
            estimator = make_pipeline(
                SimpleImputer(), StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
            )
            estimator.fit(x[train], y[train])
            prediction = estimator.predict(x[test])
            metrics[field] = {
                "kind": "continuous",
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "mae": float(mean_absolute_error(y[test], prediction)),
            }
        else:
            y = frame[field].astype("string")
            valid = y.notna().to_numpy()
            train = valid & (frame["split"].to_numpy() == "train")
            test = valid & (frame["split"].to_numpy() == "test")
            if train.sum() < 20 or test.sum() < 5 or y[train].nunique() < 2:
                continue
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.2,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            )
            estimator.fit(x[train], y[train])
            prediction = estimator.predict(x[test])
            metrics[field] = {
                "kind": "categorical",
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "classes": int(y[train].nunique()),
                "accuracy": float(accuracy_score(y[test], prediction)),
            }
        joblib.dump(estimator, args.output_dir / f"{field}.joblib")

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
