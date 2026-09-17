"""Development-only CPU OOF diagnostic for weak-field level merging.

This is an exploratory diagnostic. It never reads labels or features from the
locked test role and does not modify any v1 artifact. Features are the existing
case-level aggregation of ``geometry_deploy_v2`` image features; no GPU model
or frame-after-aggregation method is called MIL here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline


# The six weak fields used by the existing Step-10 route.
FIELDS = (
    "clarity",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
)

LEVELS = ("raw", "three_level", "two_level")


def canonical(field: str, value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    aliases = {
        "blood_color": {"[淡红色]": "淡红"},
        "exudation": {"[无]": "无"},
        "subpapillary_venous_plexus": {"[不见]": "不见"},
        "papilla": {"[波纹状]": "波纹状"},
    }
    text = aliases.get(field, {}).get(text, text)
    invalid = {
        "blood_color": {"+", "++", "+++", "暗紫"},
        "hemorrhage": {"管袢/一指甲襞"},
    }
    return None if text in invalid.get(field, set()) else text


def merge_level(field: str, value: str, level: str) -> str:
    if level == "raw":
        return value
    if field == "clarity":
        if level == "three_level":
            return value
        # Binary diagnostic: merge borderline blur with the poor-quality class.
        return {"不清": "不清/模糊", "模糊": "不清/模糊", "清晰": "清晰"}.get(value, value)
    if field == "blood_color":
        if level == "three_level":
            return {"暗红": "暗", "暗紫": "暗", "浅红": "浅红", "淡红": "淡红"}.get(value, value)
        return "暗" if value in {"暗红", "暗紫"} else "浅/淡"
    if field == "exudation":
        if level == "three_level":
            return "无" if value == "无" else ("轻度" if value == "+" else "中重度")
        return "无" if value == "无" else "有"
    if field == "hemorrhage":
        # Source schema has only absence vs 1--2; no artificial third class.
        return "无" if value == "无" else "有"
    if field == "subpapillary_venous_plexus":
        if level == "three_level":
            if value == "不见":
                return "不见"
            if value in {"可见1排", "可见2排"}:
                return "可见1--2排"
            return ">2排,扩张"
        return "不见" if value == "不见" else "可见"
    if field == "papilla":
        if level == "three_level":
            return value
        return "平坦" if value == "平坦" else "波纹"

    raise ValueError(f"unsupported field: {field}")


def case_features(matrix_path: Path, index_path: Path, allowed_case_ids: set[str] | None = None) -> pd.DataFrame:
    matrix = np.load(matrix_path, mmap_mode="r")
    index = pd.read_csv(index_path)
    rows: list[tuple[str, np.ndarray]] = []
    for case_id, positions in index.groupby("exam_case_id", sort=True).indices.items():
        if allowed_case_ids is not None and str(case_id) not in allowed_case_ids:
            continue
        values = np.asarray(matrix[np.asarray(positions)], dtype=np.float32)
        # Same image-feature summary convention used by the existing geometry CV.
        rows.append((str(case_id), np.concatenate((values.mean(0), np.median(values, axis=0), values.std(0)))))
    return pd.DataFrame({"exam_case_id": [r[0] for r in rows], "vector": [r[1] for r in rows]})


def evaluate_field(
    field: str,
    level: str,
    frame: pd.DataFrame,
    vectors: np.ndarray,
    output_rows: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    usable = frame[frame[field].notna()].copy()
    usable["target"] = usable[field].map(lambda x: merge_level(field, x, level))
    support = usable.target.value_counts()
    # Do not report a class that is absent from the training folds.
    usable = usable[usable.target.isin(set(support[support >= 5].index))].copy()
    if usable.empty:
        return {"cases": 0, "balanced_accuracy": None, "class_recall": {}, "confusion_matrix": {}}

    case_positions = dict(zip(frame.exam_case_id, frame.index))
    y = usable.target.to_numpy(dtype=object)
    folds = usable.development_fold.astype(int).to_numpy()
    ids = usable.exam_case_id.to_numpy()
    oof = np.full(len(usable), None, dtype=object)
    classes = sorted(set(y))
    for test_fold in range(5):
        train = folds != test_fold
        test = folds == test_fold
        if not test.any() or len(set(y[train])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed + test_fold,
                n_jobs=-1,
            ),
        )
        x_train = np.stack([frame.loc[case_positions[c], "vector"] for c in ids[train]])
        x_test = np.stack([frame.loc[case_positions[c], "vector"] for c in ids[test]])
        model.fit(x_train, y[train])
        oof[test] = model.predict(x_test)

    valid = np.asarray([value is not None for value in oof])
    truth = y[valid]
    pred = oof[valid]
    labels = sorted(set(truth) | set(pred))
    matrix = confusion_matrix(truth, pred, labels=labels)
    recalls = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
    for case_id, actual, prediction, fold in zip(ids[valid], truth, pred, folds[valid]):
        output_rows.append({
            "exam_case_id": case_id,
            "field": field,
            "level": level,
            "development_fold": int(fold),
            "truth": str(actual),
            "prediction": str(prediction),
            "correct": bool(actual == prediction),
        })
    return {
        "cases": int(len(truth)),
        "class_support": {label: int(np.sum(truth == label)) for label in labels},
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "class_recall": {label: float(score) for label, score in zip(labels, recalls)},
        "confusion_labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True, help="Existing geometry_deploy_v2.npy")
    parser.add_argument("--feature-index", type=Path, required=True, help="Existing geometry_deploy_v2.csv")
    parser.add_argument("--roles-labels", type=Path, required=True, help="locked_evaluation_v1.csv; development rows only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    role_labels = pd.read_csv(args.roles_labels)
    locked_seen = int((role_labels.evaluation_role == "locked_test").sum())
    development = role_labels[role_labels.evaluation_role == "development"].copy()
    features = case_features(args.features, args.feature_index, set(development.exam_case_id.astype(str)))
    frame = features.merge(development[["exam_case_id", "development_fold", *FIELDS]], on="exam_case_id", how="inner", validate="one_to_one")
    for field in FIELDS:
        frame[field] = frame[field].map(lambda value, f=field: canonical(f, value))
    vectors = np.stack(frame.vector.to_numpy())
    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    for field in FIELDS:
        results[field] = {}
        for level in LEVELS:
            results[field][level] = evaluate_field(field, level, frame, vectors, rows, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = {
        "schema_version": "level-merging-development-oof/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "locked_cases_in_source_manifest": locked_seen,
        "development_cases_in_source_manifest": int((role_labels.evaluation_role == "development").sum()),
        "cases_with_features_and_labels": int(len(frame)),
        "feature_source": str(args.features),
        "feature_aggregation": "per-case mean/median/std of existing image features",
        "gpu_used": False,
        "v1_modified": False,
        "fields": results,
        "level_definitions": {
            "raw": "canonical source categories",
            "three_level": "field-specific clinically coarse grouping defined in merge_level()",
            "two_level": "field-specific binary grouping defined in merge_level()",
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(frame), "locked_cases_seen": 0, "fields": list(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
