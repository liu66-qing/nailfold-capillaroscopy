"""Five-fold feature fusion for SigLIP2 and SAM/traditional geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline

from eval_siglip_development_cv import CAT_FIELDS, NUM_FIELDS, canonical


GEOM_FEATURES = ["coverage", "area_px", "component_count_ge_16px", "largest_component_px", "component_area_median_px", "perimeter_px", "width_px_median", "width_px_p90", "skeleton_length_px", "skeleton_endpoints", "skeleton_branchpoints"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--feature-npy", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
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
    sig = pd.DataFrame(np.stack([row[1] for row in rows]), columns=[f"s{i}" for i in range(len(rows[0][1]))])
    sig.insert(0, "exam_case_id", [row[0] for row in rows])
    sig_cols = [column for column in sig if column.startswith("s")]
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.evaluation_role.eq("development")].copy()
    results = []
    for method in ["traditional", "sam2_base_plus"]:
        geom = pd.read_csv(args.geometry_root / "mask_geometry_candidates" / f"{method}_geometry_candidates.csv")
        geom = geom[["exam_case_id"] + GEOM_FEATURES].rename(columns={column: f"g_{column}" for column in GEOM_FEATURES})
        frame = manifest[["exam_case_id", "development_fold"] + CAT_FIELDS + NUM_FIELDS].merge(sig, on="exam_case_id").merge(geom, on="exam_case_id")
        feature_cols = sig_cols + [f"g_{column}" for column in GEOM_FEATURES]
        name = f"siglip2_plus_{method}"
        for field in CAT_FIELDS:
            frame[field] = frame[field].map(lambda value: canonical(field, value))
            for fold in range(5):
                train = frame[frame.development_fold != fold].dropna(subset=[field])
                test = frame[frame.development_fold == fold].dropna(subset=[field])
                model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=400, random_state=20260803, class_weight="balanced", min_samples_leaf=2, max_features="sqrt", n_jobs=-1))
                model.fit(train[feature_cols], train[field]); pred = model.predict(test[feature_cols]); labels = sorted(train[field].unique())
                results.append({"method": name, "field": field, "fold": fold, "n": len(test), "macro_f1": f1_score(test[field], pred, labels=labels, average="macro", zero_division=0), "balanced_accuracy": balanced_accuracy_score(test[field], pred), "exact_accuracy": float(np.mean(test[field].to_numpy() == pred))})
        for field in NUM_FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
            for fold in range(5):
                train = frame[frame.development_fold != fold].dropna(subset=[field]); test = frame[frame.development_fold == fold].dropna(subset=[field])
                model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=400, random_state=20260803, min_samples_leaf=3, max_features=0.5, n_jobs=-1))
                model.fit(train[feature_cols], train[field]); pred = model.predict(test[feature_cols])
                results.append({"method": name, "field": field, "fold": fold, "n": len(test), "mae": mean_absolute_error(test[field], pred), "median_absolute_error": median_absolute_error(test[field], pred)})
    metrics = pd.DataFrame(results)
    metrics.to_csv(args.output / "fusion_cv_metrics.csv", index=False)
    metrics.groupby(["method", "field"], as_index=False).mean(numeric_only=True).to_csv(args.output / "fusion_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
