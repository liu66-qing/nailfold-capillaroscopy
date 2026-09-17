"""Build conservative per-field routing decisions from development-only CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--qlora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(args.siglip), pd.read_csv(args.geometry), pd.read_csv(args.fusion), pd.read_csv(args.qlora)]
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    metrics.to_csv(args.output / "all_candidate_fold_metrics.csv", index=False)
    support = pd.read_csv(args.qlora.parent / "field_gate_summary.csv").set_index("field")
    decisions = []
    for field, group in metrics.groupby("field"):
        categorical = group.macro_f1.notna().any()
        if categorical:
            means = group.groupby("method")[["macro_f1", "balanced_accuracy"]].mean()
            means["composite"] = (means.macro_f1 + means.balanced_accuracy) / 2
            winner = means.composite.idxmax()
            runner = means.drop(index=winner).composite.idxmax()
            w = group[group.method.eq(winner)].set_index("fold")
            r = group[group.method.eq(runner)].set_index("fold")
            delta_f1 = w.macro_f1 - r.macro_f1
            delta_ba = w.balanced_accuracy - r.balanced_accuracy
            min_support = int(support.loc[field, "min_class_support_across_folds"]) if field in support.index else 0
            gate = bool(delta_f1.mean() > 0 and delta_ba.mean() > 0 and (delta_f1 > 0).sum() >= 3 and (delta_ba > 0).sum() >= 3 and min_support >= 5)
            decisions.append({"field": field, "kind": "categorical", "winner": winner, "runner_up": runner, "winner_macro_f1": means.loc[winner, "macro_f1"], "winner_balanced_accuracy": means.loc[winner, "balanced_accuracy"], "delta_macro_f1_vs_runner": delta_f1.mean(), "delta_balanced_accuracy_vs_runner": delta_ba.mean(), "positive_folds_macro_f1": int((delta_f1 > 0).sum()), "positive_folds_balanced_accuracy": int((delta_ba > 0).sum()), "min_class_support_across_folds": min_support, "freeze_route": gate, "decision": winner if gate else "no_frozen_winner"})
        else:
            means = group.groupby("method")[["mae", "median_absolute_error"]].mean()
            means["composite"] = means.mae.rank() + means.median_absolute_error.rank()
            winner = means.composite.idxmin()
            runner = means.drop(index=winner).composite.idxmin()
            w = group[group.method.eq(winner)].set_index("fold")
            r = group[group.method.eq(runner)].set_index("fold")
            delta_mae = r.mae - w.mae
            delta_med = r.median_absolute_error - w.median_absolute_error
            gate = bool(delta_mae.mean() > 0 and delta_med.mean() > 0 and (delta_mae > 0).sum() >= 3 and (delta_med > 0).sum() >= 3)
            decisions.append({"field": field, "kind": "numeric", "winner": winner, "runner_up": runner, "winner_mae": means.loc[winner, "mae"], "winner_median_absolute_error": means.loc[winner, "median_absolute_error"], "mae_improvement_vs_runner": delta_mae.mean(), "median_error_improvement_vs_runner": delta_med.mean(), "positive_folds_mae": int((delta_mae > 0).sum()), "positive_folds_median_error": int((delta_med > 0).sum()), "freeze_route": gate, "decision": winner if gate else "no_frozen_winner"})
    table = pd.DataFrame(decisions).sort_values("field")
    table.to_csv(args.output / "field_routing_decisions.csv", index=False)
    (args.output / "field_routing_decisions.json").write_text(json.dumps({"locked_test_used": False, "gate": "classification: both macro-F1 and balanced accuracy improve with >=3/5 positive folds and min support >=5; numeric: both MAE and median error improve with >=3/5 positive folds", "decisions": table.where(table.notna(), None).to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Stage 4 field routing", "", "The 47-case locked set was not used. `no_frozen_winner` keeps the existing production route unchanged.", "", table.to_csv(index=False)]
    (args.output / "STAGE4_ROUTING.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
