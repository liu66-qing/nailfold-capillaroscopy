from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FIELDS = ("clarity", "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus", "papilla")


def aggregate(matrix_path: Path, index_path: Path) -> pd.DataFrame:
    matrix = np.load(matrix_path, mmap_mode="r")
    index = pd.read_csv(index_path)
    if len(matrix) != len(index):
        raise ValueError(f"row mismatch for {matrix_path}")
    rows = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32)
        rows.append((case_id, np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0)))))
    return pd.DataFrame({"exam_case_id": [x[0] for x in rows], "vector": [x[1] for x in rows]})


def candidates(seed: int, dimensions: int):
    return {
        "logistic": make_pipeline(
            SimpleImputer(), SelectKBest(f_classif, k=min(128, dimensions)), StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=4000, random_state=seed),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(), ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=seed,
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--dinov2-index", type=Path, required=True)
    parser.add_argument("--hulumed", type=Path, required=True)
    parser.add_argument("--hulumed-index", type=Path, required=True)
    parser.add_argument("--seg-features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dino = aggregate(args.dinov2, args.dinov2_index).rename(columns={"vector": "dino"})
    hulu = aggregate(args.hulumed, args.hulumed_index).rename(columns={"vector": "hulu"})
    seg = pd.read_parquet(args.seg_features).rename(columns={"case_id": "exam_case_id"})
    seg_columns = [column for column in seg if column.startswith("feature_")] + ["frame_count"]
    seg["seg"] = [row for row in seg[seg_columns].to_numpy(dtype=np.float32)]
    combined = dino.merge(hulu, on="exam_case_id", validate="one_to_one").merge(
        seg[["exam_case_id", "seg"]], on="exam_case_id", validate="one_to_one"
    )
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"])
    roles = roles.loc[roles.evaluation_role.eq("development")].drop(columns="evaluation_role")
    labels = pd.read_csv(args.labels, usecols=["exam_case_id", *FIELDS, *[f"{field}__confidence" for field in FIELDS]])
    frame = combined.merge(roles, on="exam_case_id", validate="one_to_one").merge(labels, on="exam_case_id", validate="one_to_one")
    routes = {
        "dino": lambda row: row.dino,
        "hulu": lambda row: row.hulu,
        "dual": lambda row: np.concatenate((row.dino, row.hulu)),
        "dual_seg": lambda row: np.concatenate((row.dino, row.hulu, row.seg)),
    }
    route_vectors = {name: np.stack([builder(row) for row in frame.itertuples(index=False)]) for name, builder in routes.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    improved = 0
    for field in FIELDS:
        valid = frame[field].notna() & frame[f"{field}__confidence"].fillna(0).ge(0.8)
        support = frame.loc[valid, field].astype(str).value_counts()
        supported = set(support[support >= 5].index)
        positions = np.flatnonzero(valid & frame[field].astype(str).isin(supported))
        y = frame.iloc[positions][field].astype(str).to_numpy()
        folds = frame.iloc[positions].development_fold.astype(int).to_numpy()
        oof = np.empty(len(positions), dtype=object)
        baseline = np.empty(len(positions), dtype=object)
        selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5
            train = (folds != test_fold) & (folds != val_fold)
            val = folds == val_fold
            test = folds == test_fold
            train_classes = set(y[train])
            val &= np.asarray([value in train_classes for value in y])
            if not train.any() or not val.any() or not test.any():
                continue
            scores = {}
            fitted = {}
            for route_name, all_vectors in route_vectors.items():
                x = all_vectors[positions]
                for model_name, model in candidates(20260816 + test_fold, x.shape[1]).items():
                    model.fit(x[train], y[train])
                    key = f"{route_name}:{model_name}"
                    scores[key] = float(balanced_accuracy_score(y[val], model.predict(x[val])))
                    fitted[key] = model
            selected = max(scores, key=lambda key: (scores[key], key))
            route_name, model_name = selected.split(":")
            x = route_vectors[route_name][positions]
            final = candidates(20260816 + test_fold, x.shape[1])[model_name]
            development = train | val
            final.fit(x[development], y[development])
            oof[test] = final.predict(x[test])
            majority = Counter(y[development]).most_common(1)[0][0]
            baseline[test] = majority
            joblib.dump(final, args.output_dir / f"{field}_fold{test_fold}_{route_name}_{model_name}.joblib")
            selections.append({"fold": test_fold, "selected": selected, "validation_balanced_accuracy": scores[selected], "test_n": int(test.sum())})
        score = float(balanced_accuracy_score(y, oof))
        baseline_score = float(balanced_accuracy_score(y, baseline))
        improved += score > baseline_score
        reports[field] = {"cases": len(y), "classes": sorted(set(y)), "balanced_accuracy": score, "baseline": baseline_score, "improved": score > baseline_score, "selections": selections}
    report = {
        "schema_version": "step10-dual-feature-oof/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "cases": len(frame),
        "fields_improved": int(improved),
        "fields_total": len(FIELDS),
        "gate": "pass" if improved >= 3 else "fail",
        "fields": reports,
    }
    (args.output_dir / "step10_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
