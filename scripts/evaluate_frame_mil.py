from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FIELDS = ("clarity", "blood_color", "crossing_ratio", "malformation_ratio", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")


def models(seed: int, dimensions: int):
    k = min(128, dimensions)
    return {
        "logistic": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=k), StandardScaler(), LogisticRegression(C=0.2, class_weight="balanced", max_iter=4000, random_state=seed)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed)),
        "gbt": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=k), HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=2.0, class_weight="balanced", random_state=seed)),
    }


def fit_weighted(model, name: str, x, y, weights):
    parameter = {"logistic": "logisticregression__sample_weight", "extra_trees": "extratreesclassifier__sample_weight", "gbt": "histgradientboostingclassifier__sample_weight"}[name]
    model.fit(x, y, **{parameter: weights})


def case_predictions(probabilities: np.ndarray, frame_cases: np.ndarray, cases: np.ndarray, classes: np.ndarray, method: str, priors: np.ndarray, alpha: float) -> np.ndarray:
    predictions = []
    for case_id in cases:
        values = probabilities[frame_cases == case_id]
        pooled = values.mean(0) if method == "mean" else np.median(values, axis=0)
        pooled = pooled / np.power(np.clip(priors, 1e-4, None), alpha)
        predictions.append(classes[int(np.argmax(pooled))])
    return np.asarray(predictions, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2", type=Path, required=True); parser.add_argument("--dinov2-index", type=Path, required=True)
    parser.add_argument("--hulumed", type=Path, required=True); parser.add_argument("--hulumed-index", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dino = np.asarray(np.load(args.dinov2, mmap_mode="r"), dtype=np.float32); hulu = np.asarray(np.load(args.hulumed, mmap_mode="r"), dtype=np.float32)
    index = pd.read_csv(args.dinov2_index); hulu_index = pd.read_csv(args.hulumed_index)
    if not index.equals(hulu_index): raise ValueError("feature indices do not align")
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"]); roles = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role")
    labels = pd.read_csv(args.labels, usecols=["exam_case_id", *FIELDS])
    cases = roles.merge(labels, on="exam_case_id", validate="one_to_one")
    frame = index.reset_index(names="feature_position").merge(cases, on="exam_case_id", validate="many_to_one")
    route_values = {"dino": dino[frame.feature_position], "hulu": hulu[frame.feature_position], "dual": np.concatenate((dino[frame.feature_position], hulu[frame.feature_position]), axis=1)}
    args.output_dir.mkdir(parents=True, exist_ok=True); reports = {}
    for field in FIELDS:
        valid_cases = cases[cases[field].notna()].copy(); support = valid_cases[field].astype(str).value_counts(); supported = set(support[support >= 5].index); valid_cases = valid_cases[valid_cases[field].astype(str).isin(supported)].copy()
        case_target = dict(zip(valid_cases.exam_case_id, valid_cases[field].astype(str))); usable_frame = frame.exam_case_id.isin(case_target).to_numpy(); frame_cases = frame.loc[usable_frame, "exam_case_id"].to_numpy(); frame_y = np.asarray([case_target[x] for x in frame_cases])
        oof = {}; selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5; train_cases = valid_cases.loc[~valid_cases.development_fold.astype(int).isin([test_fold, val_fold]), "exam_case_id"].to_numpy(); val_cases = valid_cases.loc[valid_cases.development_fold.astype(int).eq(val_fold), "exam_case_id"].to_numpy(); test_cases = valid_cases.loc[valid_cases.development_fold.astype(int).eq(test_fold), "exam_case_id"].to_numpy()
            train = np.isin(frame_cases, train_cases); val = np.isin(frame_cases, val_cases); test = np.isin(frame_cases, test_cases); counts = pd.Series(frame_cases[train]).value_counts(); weights = np.asarray([1.0 / counts[x] for x in frame_cases[train]], dtype=np.float32)
            validation_scores = {}
            for route, all_values in route_values.items():
                x = all_values[usable_frame]
                for name, model in models(20260816 + test_fold, x.shape[1]).items():
                    fit_weighted(model, name, x[train], frame_y[train], weights); probabilities = model.predict_proba(x[val]); classes_array = np.asarray(model.classes_)
                    truth = np.asarray([case_target[x] for x in val_cases]); train_case_targets = np.asarray([case_target[x] for x in train_cases]); priors = np.asarray([(train_case_targets == value).mean() for value in classes_array])
                    for pooling in ("mean", "median"):
                        for alpha in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                            prediction = case_predictions(probabilities, frame_cases[val], val_cases, classes_array, pooling, priors, alpha); validation_scores[f"{route}:{name}:{pooling}:{alpha}"] = float(balanced_accuracy_score(truth, prediction))
            selected = max(validation_scores, key=lambda key: (validation_scores[key], key)); route, name, pooling, alpha_text = selected.split(":"); alpha = float(alpha_text); development_cases = np.concatenate((train_cases, val_cases)); development = np.isin(frame_cases, development_cases); counts = pd.Series(frame_cases[development]).value_counts(); weights = np.asarray([1.0 / counts[x] for x in frame_cases[development]], dtype=np.float32); x = route_values[route][usable_frame]; final = models(20260816 + test_fold, x.shape[1])[name]; fit_weighted(final, name, x[development], frame_y[development], weights); probabilities = final.predict_proba(x[test]); classes_array = np.asarray(final.classes_); development_targets = np.asarray([case_target[x] for x in development_cases]); priors = np.asarray([(development_targets == value).mean() for value in classes_array]); prediction = case_predictions(probabilities, frame_cases[test], test_cases, classes_array, pooling, priors, alpha)
            oof.update(dict(zip(test_cases, prediction))); joblib.dump(final, args.output_dir / f"{field}_fold{test_fold}_{route}_{name}_{pooling}.joblib"); selections.append({"fold": test_fold, "selected": selected, "validation_score": validation_scores[selected]})
        ordered = valid_cases.exam_case_id.to_numpy(); truth = valid_cases[field].astype(str).to_numpy(); prediction = np.asarray([oof[x] for x in ordered]); majority = pd.Series(truth).value_counts().index[0]
        reports[field] = {"cases": len(truth), "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "baseline": float(balanced_accuracy_score(truth, np.repeat(majority, len(truth)))), "selections": selections}
    report = {"schema_version": "frame-mil-development-oof/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "images": len(frame), "cases": frame.exam_case_id.nunique(), "fields": reports}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
