from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score


ORDER = ("papilla", "clarity", "subpapillary_venous_plexus", "blood_color", "exudation", "malformation_ratio", "crossing_ratio", "hemorrhage")


def one_hot(values, classes):
    return np.column_stack([np.asarray(values) == value for value in classes]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seg-features", type=Path, required=True); parser.add_argument("--count-features", type=Path, required=True); parser.add_argument("--geometry", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    seg = pd.read_parquet(args.seg_features).rename(columns={"case_id": "exam_case_id"}); seg_cols = [c for c in seg if c.startswith("feature_")] + ["frame_count"]
    count = pd.read_csv(args.count_features).rename(columns={"case_id": "exam_case_id"}); excluded = {"exam_case_id", "archive", "frame_count", "target", "development_fold", "oof_prediction", "exam_case_id_x", "exam_case_id_y"}; count_cols = [c for c in count if c not in excluded and pd.api.types.is_numeric_dtype(count[c])]
    geometry = pd.read_csv(args.geometry).rename(columns={"case_id": "exam_case_id"}); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"]); roles = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role"); labels = pd.read_csv(args.labels, usecols=["exam_case_id", *ORDER])
    frame = seg[["exam_case_id", *seg_cols]].merge(count[["exam_case_id", *count_cols]], on="exam_case_id", how="left").merge(geometry.drop(columns=["archive"], errors="ignore"), on="exam_case_id", how="left").merge(roles, on="exam_case_id").merge(labels, on="exam_case_id")
    feature_columns = seg_cols + count_cols + ["afferent_px", "efferent_px", "apex_px", "length_px"]; raw = frame[feature_columns].to_numpy(dtype=np.float32); imputer = SimpleImputer(); oof = {field: {} for field in ORDER}; args.output_dir.mkdir(parents=True, exist_ok=True); fold_reports = []
    for fold in range(5):
        train = frame.development_fold.astype(int).ne(fold).to_numpy(); test = ~train; x_train = imputer.fit_transform(raw[train]); x_test = imputer.transform(raw[test]); train_ids = frame.loc[train, "exam_case_id"].to_numpy(); test_ids = frame.loc[test, "exam_case_id"].to_numpy(); models = []; fields = []
        for position, field in enumerate(ORDER):
            train_values = frame.loc[train, field].astype("string"); support = train_values.dropna().astype(str).value_counts(); classes = sorted(support[support >= 5].index); majority = support.index[0]; y_train = train_values.fillna(majority).astype(str); y_train = y_train.where(y_train.isin(classes), majority).to_numpy()
            model = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=20260816 + fold * 20 + position); model.fit(x_train, y_train); prediction = model.predict(x_test); oof[field].update(dict(zip(test_ids, prediction))); models.append(model); fields.append({"field": field, "classes": classes})
            x_train = np.concatenate((x_train, one_hot(y_train, classes)), axis=1); x_test = np.concatenate((x_test, one_hot(prediction, classes)), axis=1)
        joblib.dump({"imputer": imputer, "models": models, "fields": fields}, args.output_dir / f"fold{fold}.joblib"); fold_reports.append({"fold": fold, "train_cases": int(train.sum()), "test_cases": int(test.sum())})
    reports = {}
    for field in ORDER:
        valid = frame[field].notna(); support = frame.loc[valid, field].astype(str).value_counts(); supported = set(support[support >= 5].index); subset = frame[valid & frame[field].astype(str).isin(supported)]; truth = subset[field].astype(str).to_numpy(); prediction = np.asarray([oof[field][case_id] for case_id in subset.exam_case_id]); majority = pd.Series(truth).value_counts().index[0]; reports[field] = {"cases": len(truth), "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "baseline": float(balanced_accuracy_score(truth, np.repeat(majority, len(truth))))}
    report = {"schema_version": "structured-classifier-chain-oof/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "order": list(ORDER), "folds": fold_reports, "fields": reports}; (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
