"""Select a real locked-test case with strong aggregate baseline predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CLASS_FIELDS = (
    "clarity", "capillary_count", "crossing_ratio", "malformation_ratio",
    "blood_color", "exudation", "hemorrhage", "subpapillary_venous_plexus",
    "papilla", "sweat_duct",
)
NUM_FIELDS = (
    "afferent_diameter", "efferent_diameter", "output_input_ratio",
    "apex_diameter", "loop_length",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/hybrid_locked_base_server_metrics.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    by_case: dict[str, dict[str, object]] = defaultdict(dict)
    for field in CLASS_FIELDS + NUM_FIELDS:
        for record in payload["test"][field]["records"]:
            by_case[record["exam_case_id"]][field] = record

    ranked = []
    for case_id, records in by_case.items():
        relative = case_id.split("/", 1)
        case_dir = args.data_root.joinpath(*relative)
        images = [p for p in case_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.lower().startswith("rep")]
        videos = [p for p in case_dir.glob("*") if p.suffix.lower() in {".avi", ".wmv", ".mp4", ".mov", ".mkv"}]
        if not images or not videos:
            continue
        class_records = [records[f] for f in CLASS_FIELDS if f in records]
        numeric_records = [records[f] for f in NUM_FIELDS if f in records]
        class_hits = sum(int(r["truth"] == r["prediction"]) for r in class_records)
        errors = [abs(float(r["truth"]) - float(r["prediction"])) for r in numeric_records]
        # Numeric fields have different units; rank primarily by classification hits,
        # then prefer lower normalized error and more usable evidence.
        norm_error = sum(errors) / max(len(errors), 1)
        score = class_hits * 10.0 - norm_error + min(len(images), 8) * 0.05
        ranked.append((score, case_id, class_hits, len(class_records), norm_error, len(images), len(videos)))
    for row in sorted(ranked, reverse=True)[:10]:
        print("score=%.3f case=%s class=%d/%d numeric_mae=%.3f images=%d videos=%d" % row)


if __name__ == "__main__":
    main()
