"""Aggregate five-fold QLoRA results and enforce routing gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    args = parser.parse_args()

    baseline = pd.read_csv(args.stage2 / "all_metrics.csv")
    baseline = baseline[baseline.fold.astype(str).isin(["0", "1", "2", "3", "4"])].copy()
    baseline["fold"] = baseline["fold"].astype(int)
    qlora = []
    distributions = []
    for fold in range(5):
        metrics = pd.read_csv(args.stage3 / f"fold{fold}_epoch1_eval" / "metrics.csv")
        metrics["method"] = "qlora"
        qlora.append(metrics)
        dist = pd.read_csv(args.stage3 / f"fold{fold}_epoch1_eval" / "prediction_distributions.csv")
        dist["fold"] = fold
        distributions.append(dist)
    qlora_frame = pd.concat(qlora, ignore_index=True)
    qlora_frame.to_csv(args.stage3 / "five_fold_metrics.csv", index=False)
    pd.concat(distributions, ignore_index=True).to_csv(args.stage3 / "five_fold_prediction_distributions.csv", index=False)

    rows = []
    for field, group in qlora_frame.groupby("field"):
        record = {"field": field}
        for metric in ("macro_f1", "balanced_accuracy"):
            q = group.set_index("fold")[metric]
            record[f"qlora_{metric}_mean"] = q.mean()
            for method in ("zero_shot", "few_shot"):
                b = baseline[(baseline.method == method) & (baseline.field == field)].set_index("fold")[metric]
                delta = q - b
                record[f"{method}_{metric}_mean"] = b.mean()
                record[f"delta_vs_{method}_{metric}_mean"] = delta.mean()
                record[f"positive_folds_vs_{method}_{metric}"] = int((delta > 0).sum())
        record["min_class_support_across_folds"] = int(group.min_class_support.min())
        record["class_count_max"] = int(group.class_count.max())
        record["robust_vs_zero"] = bool(
            record["delta_vs_zero_shot_macro_f1_mean"] > 0
            and record["delta_vs_zero_shot_balanced_accuracy_mean"] > 0
            and record["positive_folds_vs_zero_shot_macro_f1"] >= 3
            and record["positive_folds_vs_zero_shot_balanced_accuracy"] >= 3
        )
        record["robust_vs_few"] = bool(
            record["delta_vs_few_shot_macro_f1_mean"] > 0
            and record["delta_vs_few_shot_balanced_accuracy_mean"] > 0
            and record["positive_folds_vs_few_shot_macro_f1"] >= 3
            and record["positive_folds_vs_few_shot_balanced_accuracy"] >= 3
        )
        record["strong_conclusion_allowed"] = bool(record["min_class_support_across_folds"] >= 5)
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values("field")
    summary.to_csv(args.stage3 / "field_gate_summary.csv", index=False)
    payload = {
        "gate": "positive unweighted mean delta and >=3/5 positive folds for both macro-F1 and balanced accuracy",
        "locked_test_used": False,
        "fields": summary.to_dict(orient="records"),
    }
    (args.stage3 / "field_gate_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_columns = [
        "field", "qlora_macro_f1_mean", "qlora_balanced_accuracy_mean",
        "delta_vs_zero_shot_macro_f1_mean", "delta_vs_zero_shot_balanced_accuracy_mean",
        "positive_folds_vs_zero_shot_macro_f1", "positive_folds_vs_zero_shot_balanced_accuracy",
        "robust_vs_zero", "strong_conclusion_allowed",
    ]
    header = "| " + " | ".join(report_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(report_columns)) + " |"
    table_lines = [header, separator]
    for _, row in summary[report_columns].iterrows():
        values = []
        for column in report_columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        "# Stage 3 Qwen3-VL QLoRA five-fold summary",
        "",
        "The 47-case locked set was excluded. Results are unweighted means across development folds.",
        "A robust gate requires positive mean change and at least 3/5 positive folds for both macro-F1 and balanced accuracy.",
        "Fields with fewer than five minority examples in any fold remain exploratory.",
        "",
        *table_lines,
        "",
    ]
    (args.stage3 / "STAGE3_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
