from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.data.report_layout import REPORT_FIELDS  # noqa: E402
from nailfold_report.labels.parsing import canonical_field_value  # noqa: E402


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
NUMERIC_FIELDS = {
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length", "flow_speed_um_s",
}
SCORE_GROUPS = {
    "morphology_score": [
        "clarity", "capillary_count", "afferent_diameter",
        "efferent_diameter", "output_input_ratio", "apex_diameter",
        "loop_length", "crossing_ratio", "malformation_ratio",
    ],
    "flow_score": [
        "flow_state", "flow_speed_um_s", "vasomotion", "rbc_aggregation",
        "wbc_count", "microthrombus", "blood_color",
    ],
    "periloop_score": [
        "exudation", "hemorrhage", "subpapillary_venous_plexus",
        "papilla", "sweat_duct",
    ],
}


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def export_tree(model: DecisionTreeRegressor, node: int = 0) -> dict[str, Any]:
    tree = model.tree_
    if tree.children_left[node] == tree.children_right[node]:
        return {"value": round(float(tree.value[node, 0, 0]), 6)}
    return {
        "threshold": round(float(tree.threshold[node]), 6),
        "left": export_tree(model, int(tree.children_left[node])),
        "right": export_tree(model, int(tree.children_right[node])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapidocr", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    label_rows = {}
    if args.labels is not None:
        label_frame = pd.read_csv(args.labels)
        label_rows = {
            str(row["exam_case_id"]): row for row in label_frame.to_dict("records")
        }

    by_case_field: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for line in args.rapidocr.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        tokens = row.get("audit", {}).get("tokens", [])
        for field in REPORT_FIELDS:
            label_row = label_rows.get(str(row["exam_case_id"]))
            if label_row is not None:
                confidence = label_row.get(f"{field.name}__confidence", 1.0)
                if pd.isna(confidence) or float(confidence) < 0.80:
                    continue
                raw_value = label_row.get(field.name)
            else:
                raw_value = row["fields"].get(field.name)
            value = canonical_field_value(field.name, raw_value)
            if value is None:
                continue
            target_y = field.center_y * 2
            candidates = []
            for token in tokens:
                if (
                    1100 <= token["x"] <= 1225
                    and abs(token["y"] - target_y) <= 15
                    and NUMBER.fullmatch(str(token["text"]).strip())
                ):
                    candidates.append((
                        float(token["confidence"]),
                        float(token["text"]),
                    ))
            if candidates:
                candidates = [
                    item for item in candidates if 0.0 <= item[1] <= 6.0
                ]
            if candidates:
                _, score = max(candidates)
                by_case_field[(str(row["exam_case_id"]), field.name)].append(
                    (value, score)
                )

    observations: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (_, field), values in by_case_field.items():
        pair_counts = Counter(values)
        pair, _ = pair_counts.most_common(1)[0]
        observations[field].append(pair)

    rules: dict[str, Any] = {}
    for field in (item.name for item in REPORT_FIELDS):
        pairs = observations.get(field, [])
        if field in NUMERIC_FIELDS:
            numeric = [
                (as_float(value), score) for value, score in pairs
                if as_float(value) is not None
            ]
            if len(numeric) >= 10:
                x = np.asarray([value for value, _ in numeric], dtype=np.float64)[:, None]
                y = np.asarray([score for _, score in numeric], dtype=np.float64)
                model = DecisionTreeRegressor(
                    max_depth=4, min_samples_leaf=4, random_state=20260802
                ).fit(x, y)
                rules[field] = {
                    "type": "numeric_tree", "tree": export_tree(model),
                    "cases": len(numeric),
                    "training_mae": float(np.mean(np.abs(model.predict(x) - y))),
                }
            else:
                rules[field] = {"type": "missing", "cases": len(numeric)}
        else:
            grouped: dict[str, list[float]] = defaultdict(list)
            for value, score in pairs:
                grouped[value].append(score)
            mapping = {
                value: round(float(np.median(scores)), 6)
                for value, scores in grouped.items()
            }
            dispersion = {
                value: round(float(np.max(scores) - np.min(scores)), 6)
                for value, scores in grouped.items()
            }
            rules[field] = {
                "type": "categorical_lookup", "mapping": mapping,
                "range_by_value": dispersion,
                "cases": len(pairs),
            }

    payload = {
        "version": "report-score-rules/1.0",
        "source": "RapidOCR fixed-layout score column; case-deduplicated",
        "groups": SCORE_GROUPS,
        "field_rules": rules,
        "assessment_rule": {
            "type": "ordered_thresholds",
            "thresholds": [1.0, 2.0, 4.0, 8.0],
            "labels": ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"],
            "provenance": (
                "Thresholds reconstructed from the ordered score/assessment "
                "distribution; boundary-noise cases remain in the OCR labels."
            ),
        },
        "policy": (
            "Scores are deterministic functions of predicted fields. "
            "Rules with missing values make the corresponding component unavailable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        field: {"type": rule["type"], "cases": rule["cases"]}
        for field, rule in rules.items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
