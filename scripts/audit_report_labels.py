from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = NUMBER.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.labels)
    bounds = {
        "afferent_diameter": (1, 1000),
        "efferent_diameter": (1, 1000),
        "output_input_ratio": (0.05, 20),
        "apex_diameter": (1, 2000),
        "loop_length": (1, 5000),
        "flow_speed_um_s": (0, 10000),
        "morphology_score": (0, 30),
        "flow_score": (0, 30),
        "periloop_score": (0, 30),
        "total_score": (0, 90),
    }
    issues: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        case_id = row["exam_case_id"]
        for field, (lower, upper) in bounds.items():
            parsed = number(row.get(field))
            if parsed is not None and not lower <= parsed <= upper:
                issues.append(
                    {
                        "exam_case_id": case_id,
                        "field": field,
                        "value": row.get(field),
                        "reason": f"outside broad physical/report bounds [{lower}, {upper}]",
                    }
                )
        afferent = number(row.get("afferent_diameter"))
        efferent = number(row.get("efferent_diameter"))
        ratio = number(row.get("output_input_ratio"))
        if afferent and efferent and ratio:
            calculated = efferent / afferent
            if abs(calculated - ratio) > max(0.25, 0.25 * ratio):
                issues.append(
                    {
                        "exam_case_id": case_id,
                        "field": "output_input_ratio",
                        "value": row.get("output_input_ratio"),
                        "reason": f"inconsistent with efferent/afferent={calculated:.3f}",
                    }
                )

    field_counts = {
        field: int(frame[field].notna().sum())
        for field in frame.columns
        if field not in {
            "exam_case_id",
            "report_copy_predictions",
            "conflict_field_count",
            "auto_label_status",
        }
    }
    result = {
        "cases": len(frame),
        "field_non_null_counts": field_counts,
        "conflict_cases": int((frame["conflict_field_count"] > 0).sum()),
        "issue_count": len(issues),
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "issues"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
