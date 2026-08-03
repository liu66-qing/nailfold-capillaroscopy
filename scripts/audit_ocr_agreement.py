from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nailfold_report.labels.parsing import normalize_raw_value
from nailfold_report.labels.schema import ALL_REPORT_FIELDS


NUMERIC_FIELDS = {
    "afferent_diameter",
    "efferent_diameter",
    "output_input_ratio",
    "apex_diameter",
    "loop_length",
    "morphology_score",
    "flow_score",
    "periloop_score",
    "total_score",
}
UNIT_ONLY = {
    "um",
    "um/s",
    "\u6761/mm",
    "\u6b21/min",
    "\u4e2a/15s",
    "\u4e2a/min",
    "\u4e2a/\u4e00\u6307\u7532\u895e",
    "\u7ba1\u88a2/\u4e00\u6307\u7532\u895e",
}
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def load_by_path(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            rows[str(row["report_path"])] = row
    return rows


def canonical(field: str, value: Any) -> str | None:
    text = normalize_raw_value(value)
    if text is None:
        return None
    text = text.replace("\u2265", ">=").replace("\u2264", "<=")
    text = text.replace("\uff1e", ">").replace("\uff1c", "<")
    text = re.sub(r"(?<=\d)-(?=\d)", "--", text)
    if text in UNIT_ONLY or (
        field == "flow_speed_um_s" and not NUMBER.search(text)
    ):
        return None
    aliases = {
        ("capillary_count", ">7"): ">=7",
        ("capillary_count", ">=7\u6761/mm"): ">=7",
        ("crossing_ratio", "<30%"): "<=30%",
        ("malformation_ratio", "<10%"): "<=10%",
        ("blood_color", "\u6de1\u7ea2\u8272"): "\u6de1\u7ea2",
        ("microthrombus", "\u672a\u89c1"): "\u65e0",
    }
    return aliases.get((field, text), text)


def agrees(field: str, left: str | None, right: str | None) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    if field in NUMERIC_FIELDS:
        left_match, right_match = NUMBER.search(left), NUMBER.search(right)
        if left_match and right_match:
            left_number, right_number = float(left_match.group()), float(right_match.group())
            tolerance = 0.06 if field.endswith("score") or field == "output_input_ratio" else 0.51
            return abs(left_number - right_number) <= tolerance
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapidocr", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    args = parser.parse_args()

    rapid = load_by_path(args.rapidocr)
    qwen = load_by_path(args.qwen)
    common = sorted(set(rapid) & set(qwen))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    disagreement_rows: list[dict[str, Any]] = []
    for report_path in common:
        rapid_fields = rapid[report_path]["fields"]
        qwen_fields = qwen[report_path]["fields"]
        for field in ALL_REPORT_FIELDS:
            left = canonical(field, rapid_fields.get(field))
            right = canonical(field, qwen_fields.get(field))
            if left is None and right is None:
                status = "both_blank"
            elif left is None:
                status = "qwen_on_rapid_blank"
            elif right is None:
                status = "rapid_only"
            elif agrees(field, left, right):
                status = "agreement"
            else:
                status = "disagreement"
            counts[field][status] += 1
            if status in {"qwen_on_rapid_blank", "disagreement"}:
                disagreement_rows.append(
                    {
                        "report_path": report_path,
                        "exam_case_id": rapid[report_path]["exam_case_id"],
                        "field": field,
                        "rapidocr": left,
                        "qwen": right,
                        "status": status,
                        "rapidocr_confidence": rapid[report_path]
                        .get("audit", {})
                        .get("confidence", {})
                        .get(field),
                    }
                )

    report = {
        "rapidocr_rows": len(rapid),
        "qwen_rows": len(qwen),
        "common_rows": len(common),
        "field_status_counts": {field: dict(counter) for field, counter in counts.items()},
        "policy": {
            "blank_gate": "RapidOCR/fixed-layout blank always remains null",
            "training_label": "RapidOCR cleaned consensus remains primary",
            "qwen_role": "independent disagreement audit; no automatic blank filling",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.disagreements.open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = [
            "report_path",
            "exam_case_id",
            "field",
            "rapidocr",
            "qwen",
            "status",
            "rapidocr_confidence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreement_rows)
    print(json.dumps({k: v for k, v in report.items() if k != "field_status_counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
