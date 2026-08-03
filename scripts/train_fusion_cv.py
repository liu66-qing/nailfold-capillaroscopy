from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CATEGORICAL = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus",
    "papilla", "sweat_duct",
)
CONTINUOUS = (
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length",
)


def aggregate(prefix: Path) -> pd.DataFrame:
    matrix = np.load(prefix.with_suffix(".npy")).astype(np.float32)
    index = pd.read_csv(prefix.with_suffix(".csv"))
    rows = []
    for case_id, positions in index.groupby("exam_case_id").indices.items():
        values = matrix[np.asarray(positions)]
        vector = np.concatenate((values.mean(0), values.std(0), values.max(0)))
        rows.append((case_id, vector))
    return pd.DataFrame({"exam_case_id": [x[0] for x in rows], "vector": [x[1] for x in rows]})


def classifier_candidates(seed: int, feature_count: int):
    selected = min(256, feature_count)
    return {
        "logistic": make_pipeline(
            SimpleImputer(), SelectKBest(f_classif, k=selected), StandardScaler(),
            LogisticRegression(
                C=0.3, class_weight="balanced", max_iter=5000, random_state=seed
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=240, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=seed,
            ),
        ),
    }


def regressor_candidates(seed: int, feature_count: int):
    selected = min(256, feature_count)
    return {
        "ridge": make_pipeline(
            SimpleImputer(), SelectKBest(f_regression, k=selected), StandardScaler(),
            Ridge(alpha=30.0),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesRegressor(
                n_estimators=240, min_samples_leaf=3, max_features=0.5,
                n_jobs=-1, random_state=seed,
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--evaluation-roles", type=Path)
    parser.add_argument("--min-label-confidence", type=float, default=0.80)
    args = parser.parse_args()
    sources = [aggregate(path) for path in args.features]
    combined = sources[0].rename(columns={"vector": "vector_0"})
    for index, source in enumerate(sources[1:], start=1):
        combined = combined.merge(
            source.rename(columns={"vector": f"vector_{index}"}), on="exam_case_id"
        )
    vector_columns = [column for column in combined if column.startswith("vector_")]
    vectors = np.stack([
        np.concatenate([getattr(row, column) for column in vector_columns])
        for row in combined.itertuples(index=False)
    ])
    combined["_position"] = np.arange(len(combined))
    labels = pd.read_csv(args.labels)
    if "conflict_field_count" in labels:
        labels = labels[labels.conflict_field_count.fillna(0).astype(int).eq(0)]
    folds = pd.read_csv(args.folds)[["exam_case_id", "fold"]]
    frame = combined.drop(columns=vector_columns).merge(labels, on="exam_case_id").merge(
        folds, on="exam_case_id"
    )
    if args.evaluation_roles is not None:
        roles = pd.read_csv(args.evaluation_roles)[["exam_case_id", "evaluation_role"]]
        frame = frame.merge(roles, on="exam_case_id")
        frame = frame[frame.evaluation_role.eq("development")].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for field in CATEGORICAL + CONTINUOUS:
        usable = frame[frame[field].notna()].copy()
        confidence_column = f"{field}__confidence"
        if confidence_column in usable:
            usable = usable[
                usable[confidence_column].fillna(0).ge(args.min_label_confidence)
            ]
        fold_results = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            train = usable[~usable.fold.isin([test_fold, val_fold])]
            val = usable[usable.fold.eq(val_fold)]
            test = usable[usable.fold.eq(test_fold)]
            if field in CATEGORICAL:
                classes = set(train[field].astype(str))
                val = val[val[field].astype(str).isin(classes)]
                test = test[test[field].astype(str).isin(classes)]
                if len(classes) < 2:
                    continue
            if len(train) < 10 or not len(val) or not len(test):
                continue
            x_train = vectors[train._position.astype(int)]
            x_val = vectors[val._position.astype(int)]
            if field in CATEGORICAL:
                candidates = classifier_candidates(args.seed + test_fold, vectors.shape[1])
                y_train, y_val = train[field].astype(str), val[field].astype(str)
                scorer = lambda y, p: balanced_accuracy_score(y, p)
                baseline_value = y_train.value_counts().index[0]
                baseline = float(balanced_accuracy_score(
                    test[field].astype(str), np.repeat(baseline_value, len(test))
                ))
            else:
                candidates = regressor_candidates(args.seed + test_fold, vectors.shape[1])
                y_train = pd.to_numeric(train[field])
                y_val = pd.to_numeric(val[field])
                scorer = lambda y, p: -mean_absolute_error(y, p)
                baseline_value = float(y_train.median())
                baseline = float(mean_absolute_error(
                    pd.to_numeric(test[field]), np.repeat(baseline_value, len(test))
                ))
            validation_scores = {}
            for name, model in candidates.items():
                model.fit(x_train, y_train)
                validation_scores[name] = float(scorer(y_val, model.predict(x_val)))
            selected = max(validation_scores, key=validation_scores.get)
            final_model = candidates[selected]
            development = pd.concat((train, val))
            final_model.fit(
                vectors[development._position.astype(int)],
                development[field].astype(str) if field in CATEGORICAL
                else pd.to_numeric(development[field]),
            )
            prediction = final_model.predict(vectors[test._position.astype(int)])
            if field in CATEGORICAL:
                score = float(balanced_accuracy_score(test[field].astype(str), prediction))
                metric = "balanced_accuracy"
            else:
                score = float(mean_absolute_error(pd.to_numeric(test[field]), prediction))
                metric = "mae"
            joblib.dump(final_model, args.output / f"{field}_fold{test_fold}_{selected}.joblib")
            fold_results.append({
                "fold": test_fold, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "selected": selected, "validation_scores": validation_scores,
                metric: score, "baseline": baseline,
            })
        metric = "balanced_accuracy" if field in CATEGORICAL else "mae"
        results[field] = {
            "metric": metric, "folds": fold_results,
            "mean": float(np.mean([row[metric] for row in fold_results])) if fold_results else None,
            "mean_baseline": float(np.mean([row["baseline"] for row in fold_results]))
            if fold_results else None,
        }
    (args.output / "metrics.json").write_text(json.dumps(
        {"sources": [str(path) for path in args.features], "feature_dim": vectors.shape[1],
         "cases": len(frame), "results": results},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
