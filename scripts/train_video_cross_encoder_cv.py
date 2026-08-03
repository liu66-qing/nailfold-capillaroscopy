from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ("flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "microthrombus")


def load_cases(prefix: Path):
    matrix = np.load(prefix.with_suffix(".npy")).astype(np.float32)
    index = pd.read_csv(prefix.with_suffix(".csv"))
    vectors, case_ids = [], []
    for case_id, positions in index.groupby("exam_case_id").indices.items():
        values = matrix[np.asarray(positions)]
        vectors.append(np.concatenate((values.mean(0), values.std(0), values.max(0))))
        case_ids.append(case_id)
    return np.asarray(vectors), pd.DataFrame(
        {"exam_case_id": case_ids, "_position": np.arange(len(case_ids))}
    )


def candidates(seed):
    return {
        "pca_logistic": make_pipeline(
            SimpleImputer(), StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000, random_state=seed),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(), ExtraTreesClassifier(
                n_estimators=800, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-template", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    if "conflict_field_count" in labels:
        labels = labels[labels.conflict_field_count.fillna(0).astype(int).eq(0)]
    folds = pd.read_csv(args.folds)[["exam_case_id", "fold"]]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {field: [] for field in TARGETS}
    for test_fold in range(5):
        prefix = Path(args.features_template.format(fold=test_fold))
        features, index = load_cases(prefix)
        frame = index.merge(labels, on="exam_case_id").merge(folds, on="exam_case_id")
        val_fold = (test_fold + 1) % 5
        for target in TARGETS:
            usable = frame[frame[target].notna()].copy()
            confidence_column = f"{target}__confidence"
            if confidence_column in usable:
                usable = usable[
                    usable[confidence_column].fillna(0).ge(0.80)
                ]
            train = usable[~usable.fold.isin([test_fold, val_fold])]
            val = usable[usable.fold.eq(val_fold)]
            test = usable[usable.fold.eq(test_fold)]
            vocabulary = set(train[target].astype(str))
            val = val[val[target].astype(str).isin(vocabulary)]
            test = test[test[target].astype(str).isin(vocabulary)]
            if len(train) < 10 or not len(val) or not len(test) or len(vocabulary) < 2:
                continue
            fitted, scores = {}, {}
            for name, model in candidates(args.seed + test_fold).items():
                model.fit(features[train._position.astype(int)], train[target].astype(str))
                fitted[name] = model
                scores[name] = float(balanced_accuracy_score(
                    val[target].astype(str), model.predict(features[val._position.astype(int)])
                ))
            selected = max(scores, key=scores.get)
            model = fitted[selected]
            test_score = float(balanced_accuracy_score(
                test[target].astype(str), model.predict(features[test._position.astype(int)])
            ))
            joblib.dump(model, args.output / f"{target}_fold{test_fold}_{selected}.joblib")
            results[target].append({
                "fold": test_fold, "selected": selected, "validation_scores": scores,
                "test_balanced_accuracy": test_score, "n_test": len(test),
            })
    summary = {
        field: {"folds": rows, "mean_test_balanced_accuracy": float(np.mean(
            [row["test_balanced_accuracy"] for row in rows]
        )) if rows else None}
        for field, rows in results.items()
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
