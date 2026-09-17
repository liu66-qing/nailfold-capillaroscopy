"""Leakage-safe exploratory CV for full-development SAM pixel geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline


CAT_FIELDS = ["capillary_count", "crossing_ratio", "malformation_ratio"]
NUM_FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
METHODS = ["traditional", "sam2_tiny", "sam2_base_plus", "medsam"]
FEATURES = ["coverage", "area_px", "component_count_ge_16px", "largest_component_px", "component_area_median_px", "perimeter_px", "width_px_median", "width_px_p90", "skeleton_length_px", "skeleton_endpoints", "skeleton_branchpoints"]


def canonical(field: str, value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    aliases = {
        "crossing_ratio": {"[<30%]": "<=30%", "10--30%": "<=30%"},
        "malformation_ratio": {"[<10%]": "<=10%"},
    }
    text = aliases.get(field, {}).get(text, text)
    if field == "capillary_count" and text in {"<1", "条/mm"}:
        return np.nan
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.evaluation_role.eq("development")].copy()
    rows = []
    for method in METHODS:
        geom = pd.read_csv(args.root / "mask_geometry_candidates" / f"{method}_geometry_candidates.csv")
        frame = manifest[["exam_case_id", "development_fold"] + CAT_FIELDS + NUM_FIELDS].merge(geom[["exam_case_id"] + FEATURES], on="exam_case_id", how="inner")
        X = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
        for field in CAT_FIELDS:
            frame[field] = frame[field].map(lambda value: canonical(field, value))
            for fold in sorted(frame.development_fold.unique()):
                train = frame[frame.development_fold != fold].dropna(subset=[field])
                test = frame[frame.development_fold == fold].dropna(subset=[field])
                if train[field].nunique() < 2 or test.empty:
                    continue
                model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=300, random_state=20260803, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))
                model.fit(train[FEATURES], train[field])
                pred = model.predict(test[FEATURES])
                labels = sorted(train[field].unique())
                rows.append({"method": method, "field": field, "fold": int(fold), "n": len(test), "macro_f1": f1_score(test[field], pred, labels=labels, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(test[field], pred), "exact_accuracy": float(np.mean(test[field].to_numpy() == pred))})
        for field in NUM_FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
            for fold in sorted(frame.development_fold.unique()):
                train = frame[frame.development_fold != fold].dropna(subset=[field])
                test = frame[frame.development_fold == fold].dropna(subset=[field])
                if len(train) < 20 or test.empty:
                    continue
                model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=300, random_state=20260803, min_samples_leaf=3, n_jobs=-1))
                model.fit(train[FEATURES], train[field])
                pred = model.predict(test[FEATURES])
                rows.append({"method": method, "field": field, "fold": int(fold), "n": len(test), "mae": mean_absolute_error(test[field], pred), "median_absolute_error": median_absolute_error(test[field], pred)})
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "geometry_cv_metrics.csv", index=False)
    summary = result.groupby(["method", "field"], as_index=False).mean(numeric_only=True)
    summary.to_csv(args.output / "geometry_cv_summary.csv", index=False)
    (args.output / "geometry_cv_metadata.json").write_text(json.dumps({"methods": METHODS, "features": FEATURES, "categorical_fields": CAT_FIELDS, "numeric_fields": NUM_FIELDS, "locked_test_used": False, "note": "Exploratory predictive CV on pixel-domain candidates; not segmentation accuracy or calibrated medical measurement."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
