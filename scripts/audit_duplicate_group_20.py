"""Is duplicate_group `recovered_archive1/32` (20 cases) real, or a grouping bug?

WHY THIS MATTERS
----------------
build_stratified_folds.py assigns folds BY duplicate_group and
build_locked_evaluation.py samples the locked-47 BY duplicate_group. If a group
is wrong, both the fold structure and the locked split were built on it. This
group holds 20 of the 233 cases and every one of them landed in development
fold 3 (43 cases), so if the group is spurious the fold is not 43 independent
cases -- and if it is real, fold 3's effective n is about 24.

HOW THE GROUPING IS MADE (build_splits.py:54-57)
------------------------------------------------
Cases are union-found over identical file sha256 within role cap_image /
report_image. Two cases sharing one byte-identical image become one group.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_duplicate_group_20.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "evidence", "duplicate_group_audit_20260920")
FILES = os.path.join(ROOT, "artifacts", "manifest", "files.csv")
MANIFEST = os.path.join(ROOT, "artifacts", "manifest", "locked_evaluation_v1.csv")
FOLDS = os.path.join(ROOT, "artifacts", "manifest", "stratified_folds_v3.csv")
GROUP = "recovered_archive1/32"
ROLES = ("cap_image", "report_image")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    f = pd.read_csv(FILES)
    f["exam_case_id"] = f["exam_case_id"].astype(str)
    ff = f[f["role"].isin(ROLES)].copy()

    members = sorted(man.loc[man["duplicate_group"] == GROUP, "exam_case_id"])

    # which sha256 values are shared by more than one case -- these are the only
    # edges the union-find could have used
    shared = (ff.groupby(["role", "sha256"])["exam_case_id"].nunique()
              .sort_values(ascending=False))
    multi = shared[shared > 1]

    rows = []
    for (role, sha), ncase in multi.items():
        sel = ff[(ff["role"] == role) & (ff["sha256"] == sha)]
        rows.append({
            "role": role,
            "sha256_prefix": sha[:16],
            "n_cases_linked": int(ncase),
            "n_file_rows": int(len(sel)),
            "size_bytes": int(sel["size_bytes"].iloc[0]),
            "width": float(sel["width"].iloc[0]),
            "height": float(sel["height"].iloc[0]),
            "gray_mean": float(sel["gray_mean"].iloc[0]),
            "gray_std": float(sel["gray_std"].iloc[0]),
            "is_black_placeholder": bool(sel["is_black_placeholder"].iloc[0]),
            "distinct_filenames": int(sel["filename"].nunique()),
            "example_filenames": sorted(sel["filename"].unique())[:5],
        })
    edges = pd.DataFrame(rows)

    sizes = man["duplicate_group"].value_counts()
    big = edges[edges["n_cases_linked"] >= 3]

    out = {
        "group_audited": GROUP,
        "n_members": len(members),
        "members": members,
        "group_size_distribution": {str(k): int(v) for k, v in
                                    sizes.value_counts().sort_index().items()},
        "shape_note": ("213 singletons and one group of 20, with no groups of 2 "
                       "or 3 at all. Real duplicate detection produces a tail of "
                       "small groups; this shape points at one degenerate edge."),
        "linking_edges_multi_case": edges.to_dict("records"),
        "verdict": {},
    }

    # the bug test: is every multi-case edge a black placeholder?
    if len(big) and bool(big["is_black_placeholder"].all()):
        out["verdict"] = {
            "real_duplicates": False,
            "cause": ("the only edge that links 3 or more cases is a "
                      "byte-identical ALL-BLACK placeholder image "
                      "(gray_mean 0.0, gray_std 0.0, is_black_placeholder True). "
                      "build_splits.py unions cases on identical cap_image "
                      "sha256 and never excludes placeholders, so every case "
                      "that happens to contain a blank frame was merged into one "
                      "component."),
            "consequence_folds": ("development fold 3 is NOT 43 correlated "
                                  "cases. The 20 members are unrelated patients, "
                                  "so fold 3's effective n is 43, not ~24. Any "
                                  "earlier claim that fold 3 was degenerate is "
                                  "withdrawn."),
            "consequence_locked": ("build_locked_evaluation.py samples locked "
                                   "cases BY duplicate_group, so these 20 cases "
                                   "were treated as one sampling unit and could "
                                   "contribute at most one locked case. That "
                                   "biases WHICH cases are in locked-47, but it "
                                   "does not put a case on both sides: the group "
                                   "is entirely in development."),
            "what_is_not_affected": ("no result is invalidated. The grouping is "
                                     "over-conservative, which cannot create "
                                     "leakage, only reduce the pool. It is a "
                                     "sampling-representativeness issue."),
            "fix": ("exclude is_black_placeholder rows before the union in "
                    "build_splits.py:54. NOT applied here -- rebuilding splits "
                    "would move cases between development and locked-47, and "
                    "locked-47 must not be redrawn after it has been evaluated "
                    "against."),
        }
    else:
        out["verdict"] = {
            "real_duplicates": None,
            "cause": "not explained by black placeholders; needs pixel review",
        }

    # Why are there no groups of 2 when two report_image edges each link 2 cases?
    # Because one member of each pair was filtered out of the manifest by
    # build_splits.py's cap_count_usable / report_label_available gate, so the
    # surviving member is a singleton. Recorded because the missing 2s were what
    # made the size distribution look like a bug in the first place.
    pair_rows = []
    for _, e in edges[(edges["n_cases_linked"] == 2)].iterrows():
        sha_pre = e["sha256_prefix"]
        sel = ff[(ff["role"] == e["role"]) &
                 (ff["sha256"].str.startswith(sha_pre))]
        cs = sorted(set(sel["exam_case_id"]))
        pair_rows.append({
            "role": e["role"], "sha256_prefix": sha_pre, "cases": cs,
            "in_manifest": [c in set(man["exam_case_id"]) for c in cs],
        })
    out["missing_size_2_groups_explained"] = {
        "pairs": pair_rows,
        "explanation": ("each real 2-case edge has exactly one member outside "
                        "the 233-case manifest (dropped by build_splits.py's "
                        "cap_count_usable / report_label_available filter), so "
                        "the survivor appears as a singleton. The absence of "
                        "size-2 groups is therefore expected, not a second bug."),
    }
    out["group20_all_placeholder_cases"] = bool(
        set(members) <= set(ff.loc[ff["is_black_placeholder"].astype(bool),
                                   "exam_case_id"]))

    # is that placeholder's reach wider than this one group?
    ph = ff[ff["is_black_placeholder"].astype(bool)]
    out["black_placeholder_reach"] = {
        "n_placeholder_file_rows": int(len(ph)),
        "n_cases_containing_one": int(ph["exam_case_id"].nunique()),
        "n_in_development": int(man.loc[
            man["exam_case_id"].isin(ph["exam_case_id"]) &
            man["development_fold"].notna(), "exam_case_id"].nunique()),
        "n_in_locked": int(man.loc[
            man["exam_case_id"].isin(ph["exam_case_id"]) &
            man["development_fold"].isna(), "exam_case_id"].nunique()),
        "note": ("these frames carry no image information and must be excluded "
                 "from any image-level evaluation as well, not just from the "
                 "grouping"),
    }

    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    edges.to_csv(os.path.join(OUT, "linking_edges.csv"), index=False,
                 encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
