"""Post-process completed Stage 2 predictions without rerunning the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment
    metrics = pd.read_csv(experiment / "all_metrics.csv")
    metrics["fold"] = metrics["fold"].astype(str)
    metrics.loc[metrics["fold"] == "mean", "fold"] = "pooled"
    metrics.to_csv(experiment / "all_metrics_corrected.csv", index=False)

    fold_rows = metrics[metrics["fold"].isin([str(value) for value in range(5)])].copy()
    numeric = ["n", "class_count", "min_class_support", "macro_f1", "balanced_accuracy", "exact_accuracy"]
    cv = fold_rows.groupby(["method", "field"], as_index=False)[numeric].mean()
    cv = cv.rename(columns={column: f"cv_mean_{column}" for column in numeric})
    cv.to_csv(experiment / "five_fold_mean_metrics.csv", index=False)

    methods = fold_rows.pivot(index=["field", "fold"], columns="method", values=["macro_f1", "balanced_accuracy"]).reset_index()
    methods.columns = ["_".join(item).strip("_") if isinstance(item, tuple) else item for item in methods.columns]
    rows = []
    for field, part in methods.groupby("field"):
        macro_delta = part["macro_f1_few_shot"] - part["macro_f1_zero_shot"]
        balanced_delta = part["balanced_accuracy_few_shot"] - part["balanced_accuracy_zero_shot"]
        zero = cv[(cv.method == "zero_shot") & (cv.field == field)].iloc[0]
        few = cv[(cv.method == "few_shot") & (cv.field == field)].iloc[0]
        rows.append({
            "field": field,
            "macro_f1_cv_mean_delta": float(few.cv_mean_macro_f1 - zero.cv_mean_macro_f1),
            "balanced_accuracy_cv_mean_delta": float(few.cv_mean_balanced_accuracy - zero.cv_mean_balanced_accuracy),
            "macro_f1_positive_folds": int((macro_delta > 0).sum()),
            "macro_f1_negative_folds": int((macro_delta < 0).sum()),
            "balanced_accuracy_positive_folds": int((balanced_delta > 0).sum()),
            "balanced_accuracy_negative_folds": int((balanced_delta < 0).sum()),
            "min_pooled_class_support": int(metrics[(metrics.method == "few_shot") & (metrics.field == field) & (metrics.fold == "pooled")].iloc[0].min_class_support),
            "pooled_class_count": int(metrics[(metrics.method == "few_shot") & (metrics.field == field) & (metrics.fold == "pooled")].iloc[0].class_count),
        })
    comparison = pd.DataFrame(rows).sort_values("field")
    comparison["directionally_consistent_both"] = (
        (comparison.macro_f1_cv_mean_delta > 0)
        & (comparison.balanced_accuracy_cv_mean_delta > 0)
        & (comparison.macro_f1_positive_folds >= 3)
        & (comparison.balanced_accuracy_positive_folds >= 3)
    )
    comparison["interpretation_limited"] = (comparison.pooled_class_count < 2) | (comparison.min_pooled_class_support < 10)
    comparison.to_csv(experiment / "few_shot_vs_zero_shot_five_fold.csv", index=False)

    metadata = json.loads((experiment / "run_metadata.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(metadata["folds"])
    folds = folds[folds.evaluation_role == "development"]
    case_fold = dict(zip(folds.exam_case_id, folds.development_fold))
    demo_manifest = json.loads((experiment / "few_shot_demo_manifest.json").read_text(encoding="utf-8"))
    demo_violations = [
        item for item in demo_manifest
        if int(case_fold[item["exam_case_id"]]) == int(item["fold"])
    ]
    demo_audit = {
        "demonstrations": len(demo_manifest),
        "demonstrations_within_held_out_fold": len(demo_violations),
        "violations": demo_violations,
    }
    (experiment / "few_shot_leakage_audit.json").write_text(json.dumps(demo_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = []
    mismatches = []
    distributions = []
    for method in ("zero_shot", "few_shot"):
        prediction_counts: dict[str, dict[str, int]] = {}
        for line in (experiment / f"{method}_predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["json_valid"] or row["unsupported_output_fields"]:
                failures.append({
                    "method": method,
                    "exam_case_id": row["exam_case_id"],
                    "fold": row["fold"],
                    "json_valid": row["json_valid"],
                    "unsupported_output_fields": row["unsupported_output_fields"],
                    "raw": row["raw"],
                })
            for field, target in row["target"].items():
                prediction = row["prediction"].get(field)
                if prediction is not None:
                    prediction_counts.setdefault(field, {})[prediction] = prediction_counts.setdefault(field, {}).get(prediction, 0) + 1
                if target is not None and target != prediction:
                    mismatches.append({
                        "method": method,
                        "field": field,
                        "exam_case_id": row["exam_case_id"],
                        "fold": row["fold"],
                        "target": target,
                        "prediction": prediction,
                    })
        for field, counts in prediction_counts.items():
            distributions.extend({"method": method, "field": field, "prediction": value, "count": count} for value, count in sorted(counts.items()))
    (experiment / "failure_samples.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (experiment / "field_mismatch_samples.json").write_text(json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(distributions).to_csv(experiment / "prediction_distributions.csv", index=False)
    print(json.dumps({"failures": len(failures), "fields": len(comparison), "demo_violations": len(demo_violations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
