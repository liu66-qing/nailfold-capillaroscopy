from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"])
    development_ids = set(roles.loc[roles.evaluation_role.eq("development"), "exam_case_id"])
    labels = pd.read_csv(args.labels, usecols=["exam_case_id", "flow_state", "rbc_aggregation"])
    labels = labels[labels.exam_case_id.isin(development_ids)]
    defaults = {}
    for field in ("flow_state", "rbc_aggregation"):
        values = labels[field].dropna().astype(str)
        default = values.value_counts().index[0]
        defaults[field] = {
            "value": default,
            "status": "compatibility_fallback",
            "ordinary_accuracy": float((values == default).mean()),
            "cases": len(values),
        }
    result = {
        "schema_version": "step11-video-fallback/1.0",
        "evaluation_role": "development",
        "locked_cases_seen": 0,
        "trigger": {"weighted_vote_balanced_accuracy": source["balanced_accuracy"], "threshold": 0.35},
        "action": "downgrade_video_fields_to_compatibility",
        "fields": defaults,
        "gate": "pass_with_compatibility_fallback",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
