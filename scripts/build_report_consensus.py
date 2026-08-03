from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.labels.parsing import consensus_for_case, group_ocr_rows  # noqa: E402
from nailfold_report.labels.schema import ALL_REPORT_FIELDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.ocr_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = group_ocr_rows(rows)
    consensus_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {}

    for exam_case_id, payloads in sorted(grouped.items()):
        consensus, case_audit = consensus_for_case(payloads)
        conflicts = sum(item["conflict"] for item in case_audit.values())
        consensus_rows.append(
            {
                "exam_case_id": exam_case_id,
                **consensus,
                "report_copy_predictions": len(payloads),
                "conflict_field_count": conflicts,
                "auto_label_status": "conflict" if conflicts else "consensus",
            }
        )
        audit[exam_case_id] = case_audit

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exam_case_id",
        *ALL_REPORT_FIELDS,
        "report_copy_predictions",
        "conflict_field_count",
        "auto_label_status",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consensus_rows)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(consensus_rows)} case labels")


if __name__ == "__main__":
    main()
