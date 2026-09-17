from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


ORDERS = {
    "clarity": ("不清", "模糊", "清晰"),
    "crossing_ratio": ("<=30%", "30--60%", "60--80%", ">80%"),
    "malformation_ratio": ("<=10%", "10--30%", "30--60%", ">60%"),
    "hemorrhage": ("无", "1--2"),
}


def fit(values, targets, order):
    mapping = {name: index for index, name in enumerate(order)}; y = np.asarray([mapping[x] for x in targets])
    best = (-1.0, None, False)
    for reverse in (False, True):
        transformed = -values if reverse else values; finite = transformed[np.isfinite(transformed)]; candidates = np.unique(np.quantile(finite, np.linspace(0.03, 0.97, 17)))
        for thresholds in itertools.combinations(candidates, len(order) - 1):
            prediction = np.digitize(transformed, thresholds)
            score = balanced_accuracy_score(y, prediction)
            if score > best[0]: best = (score, thresholds, reverse)
    return best


def predict(values, thresholds, reverse, order):
    indices = np.digitize(-values if reverse else values, thresholds)
    return np.asarray(order)[indices]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--features", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    features = pd.read_csv(args.features).rename(columns={"case_id": "exam_case_id"}).drop(columns=["development_fold"], errors="ignore"); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role", "development_fold"]); roles = roles[roles.evaluation_role.eq("development")].drop(columns="evaluation_role"); labels = pd.read_csv(args.labels, usecols=["exam_case_id", *ORDERS]); frame = features.merge(roles, on="exam_case_id").merge(labels, on="exam_case_id")
    epsilon = 1e-3; signals = {}
    for statistic in ("mean", "median", "max", "q75"):
        normal = frame[f"normal_count_{statistic}"].to_numpy(float); abnormal = frame[f"abnormal_count_{statistic}"].to_numpy(float)
        signals[f"abnormal_ratio_{statistic}"] = abnormal / (normal + abnormal + epsilon)
        signals[f"abnormal_count_{statistic}"] = abnormal
        signals[f"normal_count_{statistic}"] = normal
        signals[f"hemo_count_{statistic}"] = frame[f"hemo_count_{statistic}"].to_numpy(float)
        signals[f"blur_count_{statistic}"] = frame[f"blur_count_{statistic}"].to_numpy(float)
    allowed = {"clarity": [k for k in signals if "blur" in k], "crossing_ratio": [k for k in signals if "abnormal" in k], "malformation_ratio": [k for k in signals if "abnormal" in k], "hemorrhage": [k for k in signals if "hemo" in k]}
    reports = {}
    for field, order_all in ORDERS.items():
        valid = frame[field].notna(); support = frame.loc[valid, field].astype(str).value_counts(); order = tuple(x for x in order_all if support.get(x, 0) >= 5); positions = np.flatnonzero(valid & frame[field].astype(str).isin(order)); y = frame.iloc[positions][field].astype(str).to_numpy(); folds = frame.iloc[positions].development_fold.astype(int).to_numpy(); oof = np.empty(len(y), dtype=object); selections = []
        for test_fold in range(5):
            val_fold = (test_fold + 1) % 5; train = (folds != test_fold) & (folds != val_fold); val = folds == val_fold; test = folds == test_fold; candidates = {}
            for name in allowed[field]:
                values = signals[name][positions]; trained = fit(values[train], y[train], order); candidates[name] = (balanced_accuracy_score(y[val], predict(values[val], trained[1], trained[2], order)), trained)
            selected = max(candidates, key=lambda k: candidates[k][0]); values = signals[selected][positions]; final = fit(values[train | val], y[train | val], order); oof[test] = predict(values[test], final[1], final[2], order); selections.append({"fold": test_fold, "signal": selected, "validation_score": candidates[selected][0], "thresholds": list(final[1]), "reverse": final[2]})
        majority = pd.Series(y).value_counts().index[0]; reports[field] = {"cases": len(y), "balanced_accuracy": float(balanced_accuracy_score(y, oof)), "baseline": float(balanced_accuracy_score(y, np.repeat(majority, len(y)))), "selections": selections}
    report = {"schema_version": "semantic-instance-rules-oof/1.0", "evaluation_role": "development_oof", "locked_cases_seen": 0, "fields": reports}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
