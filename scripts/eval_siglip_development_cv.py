"""Re-evaluate existing SigLIP2 image features on development folds only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline


CAT_FIELDS = ["clarity", "capillary_count", "crossing_ratio", "malformation_ratio", "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla"]
NUM_FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]


def canonical(field: str, value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    aliases = {
        "crossing_ratio": {"[<30%]": "<=30%", "10--30%": "<=30%"},
        "malformation_ratio": {"[<10%]": "<=10%"},
        "blood_color": {"[淡红色]": "淡红"},
        "exudation": {"[无]": "无"},
        "subpapillary_venous_plexus": {"[不见]": "不见"},
        "papilla": {"[波纹状]": "波纹状"},
    }
    text = aliases.get(field, {}).get(text, text)
    invalid = {"capillary_count": {"<1", "条/mm"}, "blood_color": {"+", "++", "+++", "暗紫"}, "hemorrhage": {"管袢/一指甲襞"}}
    return np.nan if text in invalid.get(field, set()) else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--feature-npy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.feature_csv)
    vectors = np.load(args.feature_npy, mmap_mode="r")
    rows = []
    for case_id, group in index.groupby("exam_case_id", sort=False):
        block = np.asarray(vectors[group.index], dtype=np.float32)
        rows.append((case_id, np.concatenate([block.mean(0), block.std(0), block.max(0)])))
    case_ids = [row[0] for row in rows]
    features = np.stack([row[1] for row in rows])
    feature_frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
    feature_frame.insert(0, "exam_case_id", case_ids)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.evaluation_role.eq("development")].copy()
    frame = manifest[["exam_case_id", "development_fold"] + CAT_FIELDS + NUM_FIELDS].merge(feature_frame, on="exam_case_id", how="inner")
    feature_cols = [column for column in frame if column.startswith("f")]
    result = []
    for field in CAT_FIELDS:
        frame[field] = frame[field].map(lambda value: canonical(field, value))
        for fold in range(5):
            train = frame[frame.development_fold != fold].dropna(subset=[field])
            test = frame[frame.development_fold == fold].dropna(subset=[field])
            model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=400, random_state=20260803, class_weight="balanced", min_samples_leaf=2, max_features="sqrt", n_jobs=-1))
            model.fit(train[feature_cols], train[field])
            pred = model.predict(test[feature_cols])
            labels = sorted(train[field].unique())
            result.append({"method": "siglip2", "field": field, "fold": fold, "n": len(test), "macro_f1": f1_score(test[field], pred, labels=labels, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(test[field], pred), "exact_accuracy": float(np.mean(test[field].to_numpy() == pred))})
    for field in NUM_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        for fold in range(5):
            train = frame[frame.development_fold != fold].dropna(subset=[field])
            test = frame[frame.development_fold == fold].dropna(subset=[field])
            model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=400, random_state=20260803, min_samples_leaf=3, max_features=0.5, n_jobs=-1))
            model.fit(train[feature_cols], train[field])
            pred = model.predict(test[feature_cols])
            result.append({"method": "siglip2", "field": field, "fold": fold, "n": len(test), "mae": mean_absolute_error(test[field], pred), "median_absolute_error": median_absolute_error(test[field], pred)})
    metrics = pd.DataFrame(result)
    metrics.to_csv(args.output / "siglip2_cv_metrics.csv", index=False)
    metrics.groupby(["method", "field"], as_index=False).mean(numeric_only=True).to_csv(args.output / "siglip2_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
