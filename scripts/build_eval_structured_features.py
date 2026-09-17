from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

from classification import frame_features
from train_count_gbt import case_features


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--images", type=Path, required=True); parser.add_argument("--roles", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text()); roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"]); ids = set(roles.loc[roles.evaluation_role.eq("locked_test"), "exam_case_id"].astype(str)); by_case = defaultdict(list); seg_by_case = defaultdict(list)
    for relative, prediction in predictions.items():
        case_id = str(prediction["case_id"])
        if case_id not in ids: continue
        by_case[case_id].append(prediction); image = cv2.imread(str(args.images / relative), cv2.IMREAD_COLOR)
        if image is not None: seg_by_case[case_id].append(frame_features(image, prediction, 0.225))
    args.output_dir.mkdir(parents=True, exist_ok=True); count_rows = [case_features(case_id, frames) for case_id, frames in sorted(by_case.items())]; seg_rows = []
    for case_id, values in sorted(seg_by_case.items()):
        matrix = __import__("numpy").stack(values); seg_rows.append([case_id, case_id.split("/")[0], len(values), *matrix.mean(0), *__import__("numpy").median(matrix, axis=0), *matrix.std(0)])
    if count_rows:
        pd.DataFrame(count_rows).to_csv(args.output_dir / "count_features.csv", index=False)
        columns = ["case_id", "archive", "frame_count"] + [f"feature_{i}" for i in range(len(seg_rows[0]) - 3)]; pd.DataFrame(seg_rows, columns=columns).to_parquet(args.output_dir / "seg_features.parquet", index=False)
    (args.output_dir / "audit.json").write_text(json.dumps({"evaluation_role": "locked_test", "locked_cases": len(by_case), "locked_cases_seen": len(by_case), "images": sum(map(len, by_case.values()))}, indent=2) + "\n")


if __name__ == "__main__": main()
