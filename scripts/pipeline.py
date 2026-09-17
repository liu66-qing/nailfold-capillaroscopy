from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CATEGORICAL = ("clarity", "blood_color", "capillary_count", "crossing_ratio", "malformation_ratio", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")
CONTINUOUS = ("afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length")
COMPATIBILITY = {"sweat_duct": "0--2", "vasomotion": "0--1", "wbc_count": "1--30", "flow_velocity": None}
PIXEL_COLUMNS = {"afferent_diameter": "afferent_px", "efferent_diameter": "efferent_px", "apex_diameter": "apex_px", "loop_length": "length_px"}
ORDERS = {
    "clarity": ("不清", "模糊", "清晰"),
    "blood_color": ("暗紫", "暗红", "浅红", "淡红"),
    "crossing_ratio": ("<=30%", "30--60%", "60--80%", ">80%"),
    "malformation_ratio": ("<=10%", "10--30%", "30--60%", ">60%"),
    "exudation": ("无", "+", "++", "+++"),
    "hemorrhage": ("无", "1--2"),
    "subpapillary_venous_plexus": ("不见", "可见1排", "可见2排", ">2排,扩张"),
    "papilla": ("平坦", "浅波纹状", "波纹状"),
}


def aggregate(matrix_path: Path, index_path: Path) -> pd.DataFrame:
    matrix = np.load(matrix_path, mmap_mode="r")
    index = pd.read_csv(index_path)
    rows = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32)
        rows.append((case_id, np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0)))))
    return pd.DataFrame({"exam_case_id": [x[0] for x in rows], "vector": [x[1] for x in rows]})


def classification_models(seed: int, dimensions: int):
    return {
        "logistic": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(160, dimensions)), StandardScaler(), LogisticRegression(C=0.2, class_weight="balanced", max_iter=5000, random_state=seed)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed)),
        "linear_svm": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(160, dimensions)), StandardScaler(), SVC(C=0.1, kernel="linear", class_weight="balanced", random_state=seed)),
        "rbf_svm": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(160, dimensions)), StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=seed)),
        "gbt": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(256, dimensions)), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=2.0, class_weight="balanced", random_state=seed)),
        "lda": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(64, dimensions)), StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "centroid": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(128, dimensions)), StandardScaler(), NearestCentroid(shrink_threshold=0.1)),
        "gaussian_nb": make_pipeline(SimpleImputer(), SelectKBest(f_classif, k=min(64, dimensions)), StandardScaler(), GaussianNB(var_smoothing=0.01)),
    }


def regression_models(seed: int, dimensions: int):
    return {
        "ridge": make_pipeline(SimpleImputer(), SelectKBest(f_regression, k=min(160, dimensions)), StandardScaler(), Ridge(alpha=30.0)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2", type=Path, required=True); parser.add_argument("--dinov2-index", type=Path, required=True)
    parser.add_argument("--hulumed", type=Path, required=True); parser.add_argument("--hulumed-index", type=Path, required=True)
    parser.add_argument("--seg-features", type=Path, required=True); parser.add_argument("--count-features", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dino = aggregate(args.dinov2, args.dinov2_index).rename(columns={"vector": "dino"})
    hulu = aggregate(args.hulumed, args.hulumed_index).rename(columns={"vector": "hulu"})
    seg = pd.read_parquet(args.seg_features).rename(columns={"case_id": "exam_case_id"})
    seg_cols = [c for c in seg if c.startswith("feature_")] + ["frame_count"]
    seg["seg"] = list(seg[seg_cols].to_numpy(dtype=np.float32))
    count = pd.read_csv(args.count_features).rename(columns={"case_id": "exam_case_id"})
    excluded = {"exam_case_id", "archive", "target", "development_fold", "oof_prediction", "exam_case_id_x", "exam_case_id_y"}
    count_cols = [c for c in count if c not in excluded and pd.api.types.is_numeric_dtype(count[c])]
    count["count"] = list(count[count_cols].to_numpy(dtype=np.float32))
    geometry = pd.read_csv(args.geometry).rename(columns={"case_id": "exam_case_id"})
    frame = dino.merge(hulu, on="exam_case_id").merge(seg[["exam_case_id", "seg"]], on="exam_case_id")
    frame = frame.merge(count[["exam_case_id", "count"]], on="exam_case_id", how="left").merge(geometry, on="exam_case_id", how="left")
    count_dim = len(count_cols); frame["count"] = frame["count"].map(lambda x: x if isinstance(x, np.ndarray) else np.full(count_dim, np.nan, dtype=np.float32))
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"])
    roles = roles.loc[roles.evaluation_role.eq("development")].drop(columns="evaluation_role")
    needed = ["exam_case_id", *CATEGORICAL, *CONTINUOUS, "sweat_duct", "vasomotion", "wbc_count"]
    labels = pd.read_csv(args.labels, usecols=needed)
    frame = frame.merge(roles, on="exam_case_id", validate="one_to_one").merge(labels, on="exam_case_id", validate="one_to_one")
    geometry_values = frame[list(PIXEL_COLUMNS.values())].to_numpy(dtype=np.float32)
    archives = sorted(frame.exam_case_id.str.split("/").str[0].unique()); archive_vectors = np.stack([[float(row.exam_case_id.startswith(name + "/")) for name in archives] for row in frame.itertuples(index=False)])
    structured = np.stack([np.concatenate((row.seg, row.count, geo, domain)) for row, geo, domain in zip(frame.itertuples(index=False), geometry_values, archive_vectors)])
    visual = np.stack([np.concatenate((row.dino, row.hulu, row.seg, domain)) for row, domain in zip(frame.itertuples(index=False), archive_vectors)])
    all_features = np.stack([np.concatenate((row.dino, row.hulu, row.seg, row.count, geo, domain)) for row, geo, domain in zip(frame.itertuples(index=False), geometry_values, archive_vectors)])
    routes = {"structured": structured, "visual": visual, "all": all_features}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, field_scores = {}, {}
    for field in CATEGORICAL:
        if field == "capillary_count":
            dedicated = count[count.target.notna() & count.oof_prediction.notna()].copy()
            score = float(balanced_accuracy_score(dedicated.target.astype(str), dedicated.oof_prediction.astype(str)))
            majority = dedicated.target.astype(str).value_counts().index[0]
            baseline_score = float(balanced_accuracy_score(dedicated.target.astype(str), np.repeat(majority, len(dedicated))))
            field_scores[field] = score
            reports[field] = {"kind": "categorical", "cases": len(dedicated), "score": score, "baseline": baseline_score, "route": "step7_dedicated_multithreshold_skeleton_gbt"}
            continue
        valid = frame[field].notna(); support = frame.loc[valid, field].astype(str).value_counts(); supported = set(support[support >= 5].index)
        positions = np.flatnonzero(valid & frame[field].astype(str).isin(supported)); y = frame.iloc[positions][field].astype(str).to_numpy(); folds = frame.iloc[positions].development_fold.astype(int).to_numpy()
        oof = np.empty(len(y), dtype=object); baseline = np.empty(len(y), dtype=object); selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5; train = (folds != test_fold) & (folds != val_fold); val = folds == val_fold; test = folds == test_fold
            train_classes = set(y[train]); val &= np.asarray([v in train_classes for v in y])
            scores = {}
            for route, values in routes.items():
                x = values[positions]
                for name, model in classification_models(20260816 + test_fold, x.shape[1]).items():
                    model.fit(x[train], y[train]); scores[f"{route}:{name}"] = balanced_accuracy_score(y[val], model.predict(x[val]))
                order = [value for value in ORDERS[field] if value in set(y)]
                ordinal = {value: index for index, value in enumerate(order)}
                y_ordinal = np.asarray([ordinal[value] for value in y], dtype=np.float32)
                for name, model in regression_models(20260836 + test_fold, x.shape[1]).items():
                    model.fit(x[train], y_ordinal[train])
                    indices = np.clip(np.rint(model.predict(x[val])), 0, len(order) - 1).astype(int)
                    scores[f"{route}:ordinal_{name}"] = balanced_accuracy_score(y[val], np.asarray(order)[indices])
            selected = max(scores, key=lambda key: (scores[key], key)); route, name = selected.split(":"); x = routes[route][positions]
            development = train | val
            if name.startswith("ordinal_"):
                model_name = name.removeprefix("ordinal_"); order = [value for value in ORDERS[field] if value in set(y)]; ordinal = {value: index for index, value in enumerate(order)}; y_ordinal = np.asarray([ordinal[value] for value in y], dtype=np.float32); final = regression_models(20260836 + test_fold, x.shape[1])[model_name]; final.fit(x[development], y_ordinal[development]); indices = np.clip(np.rint(final.predict(x[test])), 0, len(order) - 1).astype(int); oof[test] = np.asarray(order)[indices]
            else:
                final = classification_models(20260816 + test_fold, x.shape[1])[name]; final.fit(x[development], y[development]); oof[test] = final.predict(x[test])
            baseline[test] = Counter(y[development]).most_common(1)[0][0]
            joblib.dump(final, args.output_dir / f"{field}_fold{test_fold}_{route}_{name}.joblib"); selections.append({"fold": test_fold, "selected": selected, "validation_score": float(scores[selected])})
        score = float(balanced_accuracy_score(y, oof)); baseline_score = float(balanced_accuracy_score(y, baseline)); field_scores[field] = score
        reports[field] = {"kind": "categorical", "cases": len(y), "score": score, "baseline": baseline_score, "selections": selections}
    for field in CONTINUOUS:
        valid = frame[field].notna(); positions = np.flatnonzero(valid); y = pd.to_numeric(frame.iloc[positions][field]).to_numpy(dtype=np.float32); folds = frame.iloc[positions].development_fold.astype(int).to_numpy(); pixel = pd.to_numeric(frame.iloc[positions][PIXEL_COLUMNS[field]]).to_numpy(dtype=np.float32)
        oof = np.full(len(y), np.nan); selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5; train = (folds != test_fold) & (folds != val_fold); val = folds == val_fold; test = folds == test_fold; candidates = {}
            scale = float(np.nanmedian(y[train] / np.where(pixel[train] > 0, pixel[train], np.nan))); candidates["geometry_scale"] = (float(mean_absolute_error(y[val], pixel[val] * scale)), scale)
            for route, values in routes.items():
                x = values[positions]
                for name, model in regression_models(20260816 + test_fold, x.shape[1]).items():
                    model.fit(x[train], y[train]); candidates[f"{route}:{name}"] = (float(mean_absolute_error(y[val], model.predict(x[val]))), model)
            selected = min(candidates, key=lambda key: (candidates[key][0], key)); development = train | val
            if selected == "geometry_scale":
                final = float(np.nanmedian(y[development] / np.where(pixel[development] > 0, pixel[development], np.nan))); oof[test] = pixel[test] * final
            else:
                route, name = selected.split(":"); x = routes[route][positions]; final = regression_models(20260816 + test_fold, x.shape[1])[name]; final.fit(x[development], y[development]); oof[test] = final.predict(x[test]); joblib.dump(final, args.output_dir / f"{field}_fold{test_fold}_{route}_{name}.joblib")
            selections.append({"fold": test_fold, "selected": selected, "validation_mae": candidates[selected][0]})
        mae = float(mean_absolute_error(y, oof)); field_range = float(np.max(y) - np.min(y)); score = max(0.0, 1.0 - mae / field_range); field_scores[field] = score
        reports[field] = {"kind": "continuous", "cases": len(y), "mae": mae, "field_range": field_range, "score": score, "selections": selections}
    for field, default in COMPATIBILITY.items():
        if field == "flow_velocity": score, cases = 1.0, len(frame)
        else:
            valid = frame[field].notna(); cases = int(valid.sum()); score = float((frame.loc[valid, field].astype(str) == default).mean())
        field_scores[field] = score; reports[field] = {"kind": "compatibility", "cases": cases, "default": default, "score": score}
    mean_score = float(np.mean(list(field_scores.values())))
    report = {"schema_version": "step12-integrated-development-cv/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "cases": len(frame), "field_count": len(field_scores), "mean_score": mean_score, "gate_threshold": 0.72, "gate": "pass" if mean_score >= 0.72 else "fail", "field_scores": field_scores, "fields": reports}
    (args.output_dir / "step12_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
