from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORICAL = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus",
    "papilla", "sweat_duct",
)
CONTINUOUS = (
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/static_candidate_summary.json"))
    args = parser.parse_args()
    families = {}
    baseline_source = None
    for name in ("geometry", "geometry_v2", "siglip", "combined", "combined_v2"):
        path = args.artifacts / f"fusion_{name}_metrics.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        baseline_source = baseline_source or document
        families[name] = {
            field: {
                "mean": details["mean"],
                "std": float(np.std([
                    fold[details["metric"]] for fold in details["folds"]
                ])),
                "metric": details["metric"],
                "fold_values": [fold[details["metric"]] for fold in details["folds"]],
                "baseline": details["mean_baseline"],
            }
            for field, details in document["results"].items()
        }
    families["constant_baseline"] = {
        field: {
            "mean": details["mean_baseline"],
            "std": float(np.std([fold["baseline"] for fold in details["folds"]])),
            "metric": details["metric"],
            "fold_values": [fold["baseline"] for fold in details["folds"]],
            "baseline": details["mean_baseline"],
        }
        for field, details in baseline_source["results"].items()
    }
    fine_tune = [
        json.loads((args.artifacts / f"siglip_ft_fold{fold}_metrics.json").read_text(
            encoding="utf-8"
        ))["test"] for fold in range(5)
    ]
    families["siglip_case_finetune"] = {}
    for field in CATEGORICAL + CONTINUOUS:
        metric = "balanced_accuracy" if field in CATEGORICAL else "mae"
        values = [fold[field][metric] for fold in fine_tune]
        families["siglip_case_finetune"][field] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "metric": metric, "fold_values": values, "baseline": None,
        }
    selected = {}
    for field in CATEGORICAL + CONTINUOUS:
        candidates = {name: values[field] for name, values in families.items()}
        reverse = field in CATEGORICAL
        winner = sorted(
            candidates, key=lambda name: candidates[name]["mean"], reverse=reverse
        )[0]
        selected[field] = {"family": winner, **candidates[winner]}
    for field in ("hemorrhage", "sweat_duct"):
        selected[field] = {
            "family": "constant_baseline", **families["constant_baseline"][field]
        }
    # These labels have too few positive examples for a meaningful five-fold claim.
    reliability = {
        "hemorrhage": {
            "status": "insufficient_minority_support", "minority_cases": 13,
            "deployment_policy": "emit confidence and indeterminate when below threshold",
        },
        "sweat_duct": {
            "status": "insufficient_minority_support", "minority_cases": 2,
            "deployment_policy": "do not claim learned rare-class sensitivity",
        },
    }
    args.output.write_text(json.dumps(
        {"exploratory_selection": selected, "families": families,
         "reliability_exceptions": reliability,
         "warning": "Family selection used the same five folds for comparison; lock the "
                    "architecture before a final untouched/prospective validation set."},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
