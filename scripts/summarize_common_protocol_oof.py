"""Summarize a same-fold development OOF comparison.

The v1 and multimodal runs must already have been evaluated on the exact same
development manifest. This script only reads metrics and never touches locked
cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "clarity",
    "capillary_count",
    "crossing_ratio",
    "malformation_ratio",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-metrics", type=Path, required=True)
    parser.add_argument("--multimodal-root", type=Path, required=True)
    parser.add_argument("--manifest-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if audit["alignment"]["strict_same_case_set"]:
        raise ValueError("audit unexpectedly reports identical full sets; expected role-specific subsets")

    rows: list[dict[str, object]] = []
    v1 = json.loads(args.v1_metrics.read_text(encoding="utf-8"))
    for field in FIELDS:
        item = v1["results"][field]
        rows.append(
            {
                "system": "our_v1",
                "method": "structured_oof",
                "field": field,
                "balanced_accuracy": float(item["mean"]),
                "n_cases": int(sum(f["n_test"] for f in item["folds"])),
                "source": str(args.v1_metrics),
            }
        )

    for model_dir in sorted(p for p in args.multimodal_root.iterdir() if p.is_dir()):
        metrics_path = model_dir / "all_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        pooled = metrics[metrics["fold"].astype(str).eq("pooled")]
        for _, item in pooled[pooled.field.isin(FIELDS)].iterrows():
            rows.append(
                {
                    "system": model_dir.name,
                    "method": str(item["method"]),
                    "field": str(item["field"]),
                    "balanced_accuracy": float(item["balanced_accuracy"]),
                    "n_cases": int(item["n"]),
                    "source": str(metrics_path),
                }
            )

    result = pd.DataFrame(rows)
    result["mean_ba"] = result.groupby(["system", "method"])["balanced_accuracy"].transform("mean")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "common_protocol_oof_metrics.csv", index=False)
    summary = (
        result.groupby(["system", "method"], as_index=False)
        .agg(mean_ba=("balanced_accuracy", "mean"), fields=("field", "nunique"), min_n=("n_cases", "min"))
        .sort_values("mean_ba", ascending=False)
    )
    summary.to_csv(args.output_dir / "common_protocol_oof_summary.csv", index=False)
    report = {
        "schema_version": "common-protocol-oof-summary/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "metric": "balanced_accuracy",
        "fields": FIELDS,
        "rows": result.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary.to_markdown(args.output_dir / "report.md", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
