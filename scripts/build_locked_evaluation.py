from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def stable_value(seed: int, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    frame = pd.read_csv(args.folds)
    frame["archive"] = frame.exam_case_id.str.split("/").str[0]
    groups = (
        frame.groupby("duplicate_group", as_index=False)
        .agg(size=("exam_case_id", "size"), archive=("archive", "first"))
    )
    selected: set[str] = set()
    quotas = {}
    for archive, archive_cases in frame.groupby("archive"):
        quota = round(len(archive_cases) * args.fraction)
        quotas[archive] = quota
        candidates = groups[groups.archive.eq(archive)].copy()
        candidates["order"] = candidates.duplicate_group.astype(str).map(
            lambda value: stable_value(args.seed, value)
        )
        candidates = candidates.sort_values("order")
        count = 0
        for row in candidates.itertuples(index=False):
            if count >= quota:
                break
            selected.add(str(row.duplicate_group))
            count += int(row.size)

    frame["evaluation_role"] = frame.duplicate_group.astype(str).map(
        lambda value: "locked_test" if value in selected else "development"
    )
    frame["development_fold"] = frame["fold"].where(
        frame.evaluation_role.eq("development")
    )
    frame["lock_seed"] = args.seed
    frame.to_csv(args.output, index=False)
    locked_ids = sorted(frame.loc[
        frame.evaluation_role.eq("locked_test"), "exam_case_id"
    ].astype(str))
    lock_hash = hashlib.sha256("\n".join(locked_ids).encode()).hexdigest()
    report = {
        "policy": (
            "Algorithmic lock for multisource/field-MIL development from v2 onward. "
            "Older exploratory models saw all historical cases, so this is not a "
            "prospective clinical lock."
        ),
        "seed": args.seed,
        "requested_fraction": args.fraction,
        "cases": len(frame),
        "locked_cases": len(locked_ids),
        "development_cases": int(frame.evaluation_role.eq("development").sum()),
        "locked_case_id_sha256": lock_hash,
        "archive_counts": pd.crosstab(
            frame.archive, frame.evaluation_role
        ).to_dict(orient="index"),
        "duplicate_group_leakage": int(
            frame.groupby("duplicate_group").evaluation_role.nunique().gt(1).sum()
        ),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
