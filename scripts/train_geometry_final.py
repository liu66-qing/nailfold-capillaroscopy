from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_fusion_cv import (
    CATEGORICAL,
    CONTINUOUS,
    aggregate,
    classifier_candidates,
    regressor_candidates,
)


ROUTES = {
    "blood_color": "geometry",
    "exudation": "geometry",
    "sweat_duct": "geometry",
    "afferent_diameter": "geometry",
    "efferent_diameter": "geometry",
    "apex_diameter": "geometry",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--development-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-label-confidence", type=float, default=0.80)
    args = parser.parse_args()

    features = aggregate(args.features)
    vectors = np.stack(features.vector)
    features["_position"] = np.arange(len(features))
    labels = pd.read_csv(args.labels)
    frame = features.drop(columns="vector").merge(labels, on="exam_case_id")
    development = json.loads(
        args.development_metrics.read_text(encoding="utf-8")
    )["results"]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": "geometry-v2-field-route-20260803",
        "routes": ROUTES,
        "models": {},
        "feature_aggregation": ["mean", "std", "max"],
        "warning": "Absolute geometry remains device/magnification dependent.",
    }
    for field in ROUTES:
        confidence = f"{field}__confidence"
        usable = frame[
            frame[field].notna()
            & frame[confidence].fillna(0).ge(args.min_label_confidence)
        ]
        choices = [
            fold["selected"] for fold in development[field]["folds"]
        ]
        selected = Counter(choices).most_common(1)[0][0]
        candidates = (
            classifier_candidates(20260803, vectors.shape[1])
            if field in CATEGORICAL
            else regressor_candidates(20260803, vectors.shape[1])
        )
        model = candidates[selected]
        target = (
            usable[field].astype(str)
            if field in CATEGORICAL else pd.to_numeric(usable[field])
        )
        model.fit(vectors[usable._position.astype(int)], target)
        filename = f"{field}_{selected}.joblib"
        joblib.dump(model, args.output / filename)
        metadata["models"][field] = {
            "file": filename, "family": selected, "cases": len(usable),
            "classes": (
                [str(value) for value in model.classes_]
                if field in CATEGORICAL else None
            ),
        }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
