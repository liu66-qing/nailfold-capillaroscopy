from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exudation_from_pixels(data_root: Path, report_path: str) -> str | None:
    with Image.open(data_root / report_path) as source:
        gray = np.asarray(source.convert("L"))
    binary = gray[326:343, 220:335] < 160
    active = np.where(binary.sum(axis=0) > 0)[0]
    if len(active) == 0:
        return None
    width = int(active[-1] - active[0] + 1)
    if 5 <= width <= 8:
        return "+"
    if 12 <= width <= 16:
        return "++"
    if 19 <= width <= 23:
        return "+++"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corrections: dict[str, int] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        fields = row["fields"]
        audit = row.setdefault("cleaning", [])

        deterministic = {
            ("output_input_ratio", "8'0"): "0.8",
            ("crossing_ratio", "%08<"): ">80%",
            ("malformation_ratio", "%09<"): ">60%",
            ("exudation", "#"): "++",
        }
        for (field, bad), replacement in deterministic.items():
            if fields.get(field) == bad:
                fields[field] = replacement
                audit.append(
                    {"field": field, "from": bad, "to": replacement, "rule": "known_ocr_glyph"}
                )
                corrections[field] = corrections.get(field, 0) + 1

        if args.data_root is not None and fields.get("exudation") not in {None, "\u65e0"}:
            pixel_value = exudation_from_pixels(args.data_root, row["report_path"])
            if pixel_value is not None and pixel_value != fields["exudation"]:
                old = fields["exudation"]
                fields["exudation"] = pixel_value
                audit.append(
                    {
                        "field": "exudation",
                        "from": old,
                        "to": pixel_value,
                        "rule": "fixed_row_dark_pixel_width",
                    }
                )
                corrections["exudation_pixels"] = (
                    corrections.get("exudation_pixels", 0) + 1
                )

        score_fields = ("morphology_score", "flow_score", "periloop_score")
        scores = {field: as_float(fields.get(field)) for field in score_fields}
        total = as_float(fields.get("total_score"))
        if total is not None and all(value is not None for value in scores.values()):
            score_sum = sum(value for value in scores.values() if value is not None)
            if abs(score_sum - total) > 0.06:
                confidence = row.get("audit", {}).get("confidence", {})
                weakest = min(
                    score_fields,
                    key=lambda field: (
                        confidence.get(field)
                        if confidence.get(field) is not None
                        else -1.0
                    ),
                )
                other_sum = sum(
                    value
                    for field, value in scores.items()
                    if field != weakest and value is not None
                )
                replacement_value = round(total - other_sum, 1)
                if 0 <= replacement_value <= 30:
                    old = fields[weakest]
                    replacement = f"{replacement_value:.1f}"
                    fields[weakest] = replacement
                    audit.append(
                        {
                            "field": weakest,
                            "from": old,
                            "to": replacement,
                            "rule": "score_additivity_lowest_confidence",
                        }
                    )
                    corrections[weakest] = corrections.get(weakest, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "corrections": corrections}, ensure_ascii=False))


if __name__ == "__main__":
    main()
