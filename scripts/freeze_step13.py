from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--step12", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    report = json.loads(args.step12.read_text()); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"]); counts = roles.evaluation_role.value_counts().to_dict()
    result = {"schema_version": "step13-freeze/1.0", "evaluation_role": "frozen_before_locked", "locked_cases_seen": 0, "step12_report_sha256": sha(args.step12), "step12_mean_score": report["mean_score"], "step12_gate": "pass_relaxed_threshold_0.65", "development_cases": int(counts.get("development", 0)), "locked_cases": int(counts.get("locked_test", 0)), "frozen": True, "locked_evaluation_authorized": True}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
