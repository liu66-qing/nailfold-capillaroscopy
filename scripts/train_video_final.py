from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_video_cross_encoder_cv import candidates, load_cases, TARGETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cv-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    features, index = load_cases(args.features)
    labels = pd.read_csv(args.labels)
    if "conflict_field_count" in labels:
        labels = labels[labels.conflict_field_count.fillna(0).astype(int).eq(0)]
    frame = index.merge(labels, on="exam_case_id")
    cv = json.loads(args.cv_metrics.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for target in TARGETS:
        usable = frame[frame[target].notna()].copy()
        confidence_column = f"{target}__confidence"
        if confidence_column in usable:
            usable = usable[
                usable[confidence_column].fillna(0).ge(0.80)
            ]
        vocabulary = sorted(usable[target].astype(str).unique())
        fitted = []
        for name, model in candidates(args.seed).items():
            model.fit(
                features[usable._position.astype(int)], usable[target].astype(str)
            )
            joblib.dump(model, args.output / f"{target}_{name}.joblib")
            fitted.append(name)
        counts = usable[target].astype(str).value_counts().to_dict()
        minority = min(counts.values())
        metadata[target] = {
            "classes": vocabulary, "class_counts": counts, "models": fitted,
            "cv_balanced_accuracy": cv[target]["mean_test_balanced_accuracy"],
            "reliable_for_claim": bool(minority >= 30),
            "policy": "ensemble_mean_probability" if minority >= 30 else
                      "emit_low_support_flag_and_confidence",
        }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
