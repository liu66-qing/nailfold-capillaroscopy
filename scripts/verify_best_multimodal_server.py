"""Verify the best completed multimodal run from server-side raw metrics.

This is an audit/recalculation step: it does not open images or locked cases,
and it does not retrain or alter any model checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    runs = []
    for run_dir in sorted(p for p in args.outputs_root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "all_metrics.csv"
        metadata_path = run_dir / "run_metadata.json"
        if not metrics_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("locked_cases_excluded", -1)) != 35:
            raise ValueError(f"unexpected locked exclusion in {run_dir}: {metadata.get('locked_cases_excluded')}")
        if int(metadata.get("development_cases", -1)) != 198:
            raise ValueError(f"unexpected development count in {run_dir}: {metadata.get('development_cases')}")
        metrics = pd.read_csv(metrics_path)
        pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
        if pooled.empty:
            raise ValueError(f"no pooled rows in {metrics_path}")
        for method, part in pooled.groupby("method", sort=True):
            rows.append({
                "model": run_dir.name,
                "method": method,
                "fields": int(len(part)),
                "mean_exact_accuracy": float(part["exact_accuracy"].mean()),
                "mean_balanced_accuracy": float(part["balanced_accuracy"].mean()),
                "mean_macro_f1": float(part["macro_f1"].mean()),
                "json_valid_rate": float(metadata.get(f"{method}_json_valid_rate", float("nan"))),
                "mean_latency_seconds": float(metadata.get(f"{method}_mean_latency_seconds", float("nan"))),
                "locked_cases_excluded": int(metadata["locked_cases_excluded"]),
                "development_cases": int(metadata["development_cases"]),
                "model_sha256": metadata.get("model_sha256"),
            })
        runs.append({
            "run": run_dir.name,
            "model": metadata.get("model"),
            "model_sha256": metadata.get("model_sha256"),
            "development_cases": metadata.get("development_cases"),
            "locked_cases_excluded": metadata.get("locked_cases_excluded"),
        })
    if not rows:
        raise RuntimeError("no completed multimodal runs found")
    table = pd.DataFrame(rows).sort_values(
        ["mean_exact_accuracy", "mean_balanced_accuracy", "mean_macro_f1"], ascending=False
    ).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    report = {
        "schema_version": "best-multimodal-server-verification/1.0",
        "selection_metric": "mean pooled Exact Accuracy across fields",
        "selection_tiebreakers": ["mean pooled Balanced Accuracy", "mean pooled Macro-F1"],
        "best": best,
        "all_candidates": table.to_dict(orient="records"),
        "run_audits": runs,
        "locked_cases_opened": 0,
        "note": "Recomputed from server-side all_metrics.csv; no image inference, training, or checkpoint modification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "candidates": len(table), "locked_cases_opened": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
