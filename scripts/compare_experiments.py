"""Compare all experiments and select best method per field.
Run after all experiments complete. Produces final Total Score MAE.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

SCORE_RULES_PATH = "/root/nailfold/artifacts/labels/score_rules_v3.json"
LABELS_PATH = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
EXP_DIR = Path("/root/nailfold/artifacts/experiments")

FIELDS = [
    "clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla",
    "capillary_count", "crossing_ratio", "malformation_ratio",
    "flow_state", "rbc_aggregation", "microthrombus", "hemorrhage",
]

def load_exp_results(exp_name):
    """Load per-field metrics from an experiment."""
    rpath = EXP_DIR / exp_name / f"{exp_name.split('/')[-1]}_results.json"
    # Try alternate names
    candidates = list((EXP_DIR / exp_name).glob("*_results.json"))
    if not candidates:
        return None
    results = json.load(open(candidates[0]))
    return results

def main():
    labels = pd.read_csv(LABELS_PATH, dtype={"exam_case_id": str})
    dev = labels[labels.evaluation_role == "development"].set_index("exam_case_id")
    score_rules = json.load(open(SCORE_RULES_PATH))

    print("=" * 70)
    print("EXPERIMENT COMPARISON - Per-field sMAE and BA")
    print("=" * 70)

    experiments = {}
    for exp_dir in sorted(EXP_DIR.glob("exp*")):
        jsons = list(exp_dir.glob("*_results.json"))
        if jsons:
            name = exp_dir.name
            results = json.load(open(jsons[0]))
            experiments[name] = results
            print(f"\nLoaded: {name}")
            if "per_field" in results:
                for f in FIELDS:
                    pf = results["per_field"].get(f, {})
                    if pf:
                        print(f"  {f:30s} BA={pf.get('ba',0):.3f}  sMAE={pf.get('smae',0):.3f}")
            if "total_score_mae" in results:
                print(f"  Total Score MAE: {results['total_score_mae']:.3f}")
            if "overall_assessment_acc" in results:
                print(f"  Overall Assessment Acc: {results['overall_assessment_acc']:.3f}")

    # Select best method per field
    print("\n" + "=" * 70)
    print("BEST METHOD PER FIELD (by sMAE)")
    print("=" * 70)
    best_per_field = {}
    for f in FIELDS:
        best_smae = float("inf")
        best_exp = "mode_baseline"
        for exp_name, results in experiments.items():
            pf = results.get("per_field", {}).get(f, {})
            if pf and pf.get("smae", float("inf")) < best_smae:
                best_smae = pf["smae"]
                best_exp = exp_name
        best_per_field[f] = {"experiment": best_exp, "smae": best_smae}
        print(f"  {f:30s} -> {best_exp:30s} sMAE={best_smae:.3f}")

    # Save comparison
    comparison = {
        "experiments": {k: {
            "total_score_mae": v.get("total_score_mae"),
            "overall_assessment_acc": v.get("overall_assessment_acc"),
        } for k, v in experiments.items()},
        "best_per_field": best_per_field,
    }
    out_path = EXP_DIR / "comparison_results.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(f"\nComparison saved to {out_path}")

if __name__ == "__main__":
    main()
