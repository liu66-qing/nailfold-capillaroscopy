"""OCR-consistency proxy and confidence-weighted CPU OOF diagnostic.

The ``*_status`` fields are the existing RapidOCR/Qwen agreement audit proxy.
They are not a physician annotation study and are explicitly not Cohen's
kappa/doctor agreement. Only development-role cases are fitted/evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline

from diagnose_level_merging import FIELDS, case_features, canonical


def agreement_group(status: object) -> str:
    text = "" if pd.isna(status) else str(status)
    if text.startswith("high_consensus"):
        return "agreement"
    if "disagreement" in text:
        return "disagreement"
    if "missing" in text or "qwen_only" in text or "blank" in text:
        return "missing"
    return "missing"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("exam_case_id"):
            rows.append(row)
    return rows


def build_ocr_groups(rapid_path: Path, qwen_path: Path, case_ids: set[str]) -> dict[tuple[str, str], str]:
    rapid_rows, qwen_rows = load_jsonl(rapid_path), load_jsonl(qwen_path)
    rapid = {str(row.get("report_path")): row for row in rapid_rows if row.get("report_path")}
    qwen = {str(row.get("report_path")): row for row in qwen_rows if row.get("report_path")}
    groups: dict[tuple[str, str], str] = {}
    for case_id in sorted(case_ids):
        for field in FIELDS:
            statuses: list[str] = []
            for report_path in sorted(set(rapid) & set(qwen)):
                if str(rapid[report_path].get("exam_case_id")) != case_id:
                    continue
                lv = canonical(field, rapid[report_path].get("fields", {}).get(field))
                rv = canonical(field, qwen[report_path].get("fields", {}).get(field))
                if lv is None and rv is None:
                    statuses.append("missing")
                elif lv is None or rv is None:
                    statuses.append("missing")
                elif lv == rv:
                    statuses.append("agreement")
                else:
                    statuses.append("disagreement")
            if not statuses or all(value == "missing" for value in statuses):
                group = "missing"
            elif "disagreement" in statuses:
                group = "disagreement"
            else:
                group = "agreement"
            groups[(case_id, field)] = group
    return groups


def fit_oof(frame: pd.DataFrame, field: str, weighted: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    usable = frame[frame[field].notna()].copy()
    y = usable[field].astype(str).to_numpy(dtype=object)
    folds = usable.development_fold.astype(int).to_numpy()
    ids = usable.exam_case_id.to_numpy()
    groups = usable[f"{field}__ocr_group"].to_numpy()
    predictions = np.full(len(usable), None, dtype=object)
    id_to_vector = dict(zip(frame.exam_case_id, frame.vector))
    for test_fold in range(5):
        train = folds != test_fold
        test = folds == test_fold
        if not test.any() or len(set(y[train])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=220, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", random_state=seed + test_fold, n_jobs=1,
            ),
        )
        x_train = np.stack([id_to_vector[c] for c in ids[train]])
        x_test = np.stack([id_to_vector[c] for c in ids[test]])
        weights = None
        if weighted:
            weights = np.asarray([
                {"agreement": 1.5, "disagreement": 0.5, "missing": 1.0}[g]
                for g in groups[train]
            ], dtype=np.float32)
        model.fit(x_train, y[train], **{"extratreesclassifier__sample_weight": weights} if weights is not None else {})
        predictions[test] = model.predict(x_test)
    return y, predictions


def summarize(y: np.ndarray, predictions: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    valid = np.asarray([x is not None for x in predictions])
    result: dict[str, object] = {"cases": int(valid.sum()), "balanced_accuracy": float(balanced_accuracy_score(y[valid], predictions[valid]))}
    by_group: dict[str, object] = {}
    for group in ("agreement", "disagreement", "missing"):
        selected = valid & (groups == group)
        if not selected.any():
            by_group[group] = {"cases": 0, "balanced_accuracy": None, "class_recall": {}}
            continue
        labels = sorted(set(y[selected]) | set(predictions[selected]))
        recalls = recall_score(y[selected], predictions[selected], labels=labels, average=None, zero_division=0)
        by_group[group] = {
            "cases": int(selected.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[selected], predictions[selected])),
            "class_support": {label: int(np.sum(y[selected] == label)) for label in labels},
            "class_recall": {label: float(score) for label, score in zip(labels, recalls)},
        }
    result["ocr_proxy_groups"] = by_group
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--roles-labels", type=Path, required=True)
    parser.add_argument("--level-oof", type=Path, required=True)
    parser.add_argument("--rapidocr", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    labels = pd.read_csv(args.roles_labels)
    locked_seen = int((labels.evaluation_role == "locked_test").sum())
    development = labels[labels.evaluation_role == "development"].copy()
    feature_frame = case_features(args.features, args.feature_index, set(development.exam_case_id.astype(str)))
    needed = ["exam_case_id", "development_fold", *FIELDS, *[f"{f}__status" for f in FIELDS]]
    frame = feature_frame.merge(development[needed], on="exam_case_id", how="inner", validate="one_to_one")
    ocr_groups = build_ocr_groups(args.rapidocr, args.qwen, set(frame.exam_case_id))
    for field in FIELDS:
        frame[field] = frame[field].map(lambda value, f=field: canonical(f, value))
        frame[f"{field}__ocr_group"] = [ocr_groups[(str(case_id), field)] for case_id in frame.exam_case_id]

    reports: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for field in FIELDS:
        usable = frame[frame[field].notna()].copy()
        raw_y, raw_pred = fit_oof(frame, field, weighted=False, seed=args.seed)
        weighted_y, weighted_pred = fit_oof(frame, field, weighted=True, seed=args.seed + 1000)
        groups = usable[f"{field}__ocr_group"].to_numpy()
        baseline = summarize(raw_y, raw_pred, groups)
        weighted = summarize(weighted_y, weighted_pred, groups)
        reports[field] = {"unweighted_oof": baseline, "agreement_weighted_oof": weighted, "weight_policy": {"agreement": 1.5, "disagreement": 0.5, "missing": 1.0}}
        for case_id, group, truth, p0, p1 in zip(usable.exam_case_id, groups, raw_y, raw_pred, weighted_pred):
            rows.append({"exam_case_id": case_id, "field": field, "ocr_proxy_group": group, "truth": str(truth), "unweighted_prediction": None if p0 is None else str(p0), "weighted_prediction": None if p1 is None else str(p1)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "oof_group_predictions.csv", index=False)
    report = {
        "schema_version": "ocr-confidence-development-oof/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "locked_cases_in_source_manifest": locked_seen,
        "development_cases_in_source_manifest": int((labels.evaluation_role == "development").sum()),
        "cases_with_features_and_labels": int(len(frame)),
        "gpu_used": False,
        "v1_modified": False,
        "ocr_agreement_definition": "existing RapidOCR/Qwen consensus status proxy; not physician kappa",
        "level_oof_reference": str(args.level_oof),
        "fields": reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(frame), "locked_cases_seen": 0, "fields": list(reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
