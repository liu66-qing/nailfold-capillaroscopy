from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

from train_fusion_cv import (
    CATEGORICAL,
    CONTINUOUS,
    aggregate,
    classifier_candidates,
    regressor_candidates,
)


def bootstrap(truth, prediction, metric, seed=20260803, samples=2000):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = generator.integers(0, len(truth), len(truth))
        values.append(float(metric(truth[indices], prediction[indices])))
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--development-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-label-confidence", type=float, default=0.80)
    args = parser.parse_args()

    features = aggregate(args.features)
    vectors = np.stack(features.vector)
    features["_position"] = np.arange(len(features))
    labels = pd.read_csv(args.labels)
    roles = pd.read_csv(args.roles)[["exam_case_id", "evaluation_role"]]
    frame = features.drop(columns="vector").merge(labels, on="exam_case_id").merge(
        roles, on="exam_case_id"
    )
    development_metrics = json.loads(
        args.development_metrics.read_text(encoding="utf-8")
    )["results"]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for field in (*CATEGORICAL, *CONTINUOUS):
        confidence_column = f"{field}__confidence"
        usable = frame[
            frame[field].notna()
            & frame[confidence_column].fillna(0).ge(args.min_label_confidence)
        ].copy()
        train = usable[usable.evaluation_role.eq("development")]
        test = usable[usable.evaluation_role.eq("locked_test")]
        choices = [
            fold["selected"] for fold in development_metrics[field]["folds"]
        ]
        if not choices or len(train) < 10 or not len(test):
            continue
        selected = Counter(choices).most_common(1)[0][0]
        if field in CATEGORICAL:
            candidates = classifier_candidates(20260803, vectors.shape[1])
            model = candidates[selected]
            model.fit(
                vectors[train._position.astype(int)], train[field].astype(str)
            )
            truth = test[field].astype(str).to_numpy()
            prediction = model.predict(vectors[test._position.astype(int)])
            metric = balanced_accuracy_score
            result = {
                "n": len(test), "selected": selected,
                "balanced_accuracy": float(metric(truth, prediction)),
                "balanced_accuracy_ci95": bootstrap(truth, prediction, metric),
            }
        else:
            candidates = regressor_candidates(20260803, vectors.shape[1])
            model = candidates[selected]
            model.fit(
                vectors[train._position.astype(int)],
                pd.to_numeric(train[field]),
            )
            truth = pd.to_numeric(test[field]).to_numpy()
            prediction = model.predict(vectors[test._position.astype(int)])
            metric = mean_absolute_error
            errors = np.abs(truth - prediction)
            result = {
                "n": len(test), "selected": selected,
                "mae": float(metric(truth, prediction)),
                "median_absolute_error": float(np.median(errors)),
                "mae_ci95": bootstrap(truth, prediction, metric),
            }
        result["records"] = [
            {
                "exam_case_id": case_id,
                "truth": str(actual) if field in CATEGORICAL else float(actual),
                "prediction": str(predicted) if field in CATEGORICAL else float(predicted),
            }
            for case_id, actual, predicted in zip(
                test.exam_case_id, truth, prediction
            )
        ]
        joblib.dump(model, args.output / f"{field}_{selected}.joblib")
        results[field] = result
    (args.output / "metrics.json").write_text(
        json.dumps({
            "mode": "development_train_locked_test",
            "warning": "Locked metrics must not be used for further family selection.",
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
