from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--dinov2-index", type=Path)
    parser.add_argument("--hulumed", type=Path, required=True)
    parser.add_argument("--hulumed-index", type=Path)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dino_index = pd.read_csv(args.dinov2_index or args.dinov2.with_suffix(".csv"))
    hulu_index = pd.read_csv(args.hulumed_index or args.hulumed.with_suffix(".csv"))
    keys = ["exam_case_id", "image_path"]
    if dino_index.duplicated(keys).any() or hulu_index.duplicated(keys).any():
        raise ValueError("duplicate image keys")
    if not dino_index[keys].equals(hulu_index[keys]):
        raise ValueError("DINOv2 and Hulu-Med image indices are not exactly aligned")

    roles = pd.read_csv(args.roles, usecols=["exam_case_id", "evaluation_role"])
    audit = dino_index[["exam_case_id"]].drop_duplicates().merge(
        roles, on="exam_case_id", validate="one_to_one"
    )
    role_counts = audit.evaluation_role.value_counts().to_dict()
    if set(role_counts) != {"development"}:
        raise ValueError(f"non-development cases present: {role_counts}")

    dino = np.load(args.dinov2.with_suffix(".npy"), mmap_mode="r")
    hulu = np.load(args.hulumed.with_suffix(".npy"), mmap_mode="r")
    if len(dino) != len(dino_index) or len(hulu) != len(hulu_index):
        raise ValueError("feature row count does not match index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dinov2=np.asarray(dino, dtype=np.float16),
        hulumed=np.asarray(hulu, dtype=np.float16),
        exam_case_id=dino_index.exam_case_id.to_numpy(dtype=str),
        image_path=dino_index.image_path.to_numpy(dtype=str),
    )
    report = {
        "schema_version": "dual-visual-features/1.0",
        "evaluation_role": "development",
        "locked_cases_seen": 0,
        "cases": int(dino_index.exam_case_id.nunique()),
        "images": len(dino_index),
        "dinov2_shape": list(dino.shape),
        "hulumed_shape": list(hulu.shape),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
