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
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ("flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "microthrombus")


def case_features(matrix: np.ndarray, index: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    vectors, case_ids = [], []
    for case_id, positions in index.groupby("exam_case_id").indices.items():
        values = matrix[np.asarray(positions)]
        vectors.append(np.concatenate((values.mean(0), values.std(0), values.max(0))))
        case_ids.append(case_id)
    return np.asarray(vectors, dtype=np.float32), case_ids


def candidates(seed: int):
    return {
        "pca_logistic": make_pipeline(
            SimpleImputer(), StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000, random_state=seed),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=800, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--evaluation-roles", type=Path)
    parser.add_argument("--min-label-confidence", type=float, default=0.05)
    parser.add_argument("--eval-min-confidence", type=float, default=0.80)
    args = parser.parse_args()
    matrix = np.load(args.features.with_suffix(".npy")).astype(np.float32)
    index = pd.read_csv(args.features.with_suffix(".csv"))
    features, case_ids = case_features(matrix, index)
    feature_frame = pd.DataFrame({"exam_case_id": case_ids, "_position": range(len(case_ids))})
    labels = pd.read_csv(args.labels)
    folds = pd.read_csv(args.folds)[["exam_case_id", "fold"]]
    frame = feature_frame.merge(labels, on="exam_case_id").merge(folds, on="exam_case_id")
    if "conflict_field_count" in frame:
        frame = frame[frame.conflict_field_count.fillna(0).astype(int).eq(0)]
    if args.evaluation_roles:
        roles = pd.read_csv(args.evaluation_roles)[
            ["exam_case_id", "evaluation_role", "development_fold"]
        ]
        frame = frame.merge(roles, on="exam_case_id")
        frame = frame[frame.evaluation_role.eq("development")].copy()
        frame["fold"] = frame.development_fold.astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for target in TARGETS:
        confidence_column = f"{target}__confidence"
        usable = frame[frame[target].notna()].copy()
        if confidence_column in usable:
            usable = usable[
                usable[confidence_column].fillna(0).ge(args.min_label_confidence)
            ]
        target_results = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            train = usable[~usable.fold.isin([test_fold, val_fold])]
            val = usable[usable.fold.eq(val_fold)]
            test = usable[usable.fold.eq(test_fold)]
            if confidence_column in usable:
                val = val[val[confidence_column].ge(args.eval_min_confidence)]
                test = test[test[confidence_column].ge(args.eval_min_confidence)]
            vocab = sorted(train[target].astype(str).unique())
            train = train[train[target].astype(str).isin(vocab)]
            val = val[val[target].astype(str).isin(vocab)]
            test = test[test[target].astype(str).isin(vocab)]
            if len(train) < 10 or len(val) == 0 or len(test) == 0 or len(vocab) < 2:
                continue
            x_train = features[train._position.astype(int)]
            y_train = train[target].astype(str)
            x_val = features[val._position.astype(int)]
            y_val = val[target].astype(str)
            fitted = {}
            val_scores = {}
            for name, model in candidates(args.seed + test_fold).items():
                model.fit(x_train, y_train)
                fitted[name] = model
                val_scores[name] = float(balanced_accuracy_score(y_val, model.predict(x_val)))
            selected = max(val_scores, key=val_scores.get)
            model = fitted[selected]
            truth = test[target].astype(str).to_numpy()
            prediction = model.predict(features[test._position.astype(int)])
            score = float(balanced_accuracy_score(truth, prediction))
            labels_in_test = sorted(set(truth))
            joblib.dump(model, args.output / f"{target}_fold{test_fold}_{selected}.joblib")
            target_results.append({
                "fold": test_fold, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "selected": selected, "validation_scores": val_scores,
                "test_balanced_accuracy": score,
                "test_macro_f1": float(f1_score(
                    truth, prediction, labels=labels_in_test, average="macro",
                    zero_division=0,
                )),
                "per_class_recall": {
                    label: float(value) for label, value in zip(
                        labels_in_test,
                        recall_score(
                            truth, prediction, labels=labels_in_test,
                            average=None, zero_division=0,
                        ),
                    )
                },
                "records": [
                    {
                        "exam_case_id": case_id,
                        "truth": str(actual),
                        "prediction": str(predicted),
                    }
                    for case_id, actual, predicted in zip(
                        test.exam_case_id, truth, prediction
                    )
                ],
            })
        results[target] = {
            "folds": target_results,
            "mean_test_balanced_accuracy": float(np.mean(
                [row["test_balanced_accuracy"] for row in target_results]
            )) if target_results else None,
        }
    (args.output / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
