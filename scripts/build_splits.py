from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases[
        (cases["cap_count_usable"] > 0) & cases["report_label_available"].astype(bool)
    ].copy()
    case_ids = cases["exam_case_id"].astype(str).tolist()
    dsu = DisjointSet(case_ids)

    files = pd.read_csv(args.files)
    files = files[
        files["exam_case_id"].isin(case_ids)
        & files["role"].isin(["cap_image", "report_image"])
    ]
    for _, group in files.groupby(["role", "sha256"]):
        members = sorted(set(group["exam_case_id"].astype(str)))
        for member in members[1:]:
            dsu.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        components[dsu.find(case_id)].append(case_id)

    split_by_case: dict[str, str] = {}
    targets = {"train": 0.70 * len(case_ids), "val": 0.15 * len(case_ids), "test": 0.15 * len(case_ids)}
    assigned = {name: 0 for name in targets}
    ordered_components = sorted(
        components.values(),
        key=lambda members: stable_fraction("|".join(sorted(members)), args.seed),
    )
    for members in ordered_components:
        split = min(targets, key=lambda name: assigned[name] / targets[name])
        for member in members:
            split_by_case[member] = split
        assigned[split] += len(members)

    cases["split"] = cases["exam_case_id"].map(split_by_case)
    cases["duplicate_group"] = cases["exam_case_id"].map(dsu.find)
    cases["split_seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output, index=False)
    print(cases["split"].value_counts().sort_index().to_dict())
    print(f"duplicate groups: {cases['duplicate_group'].nunique()} / cases: {len(cases)}")


if __name__ == "__main__":
    main()
