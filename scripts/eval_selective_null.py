#!/usr/bin/env python
"""Permutation null for the selective-prediction result: is it multiplicity?

The coverage sweep runs 13 fields x 6 coverage points = 78 hypothesis tests at
CI-lower-bound > 0. At that many looks, some field will pass somewhere by chance.
malformation_ratio passed at coverage 0.90/0.80/0.70 -- three CONSECUTIVE points
with a stable minority share. This script asks how often that pattern arises when
the label carries no information at all.

Null construction: permute the case labels (breaking any link to the features),
rerun the identical OOF fit and the identical coverage sweep, and record the
longest run of consecutive passing coverage points. Repeat N_PERM times. The
p-value is the fraction of permutations whose longest run is >= the observed run.

This is the same discipline used to refute the earlier pooling result: a shuffled
null, not an appeal to plausibility. If p is not small, the selective route is
multiplicity and must be reported as refuted.

locked-47: never loaded, asserted.
"""
import argparse
import json

import numpy as np
import pandas as pd

from eval_selective_coverage import (
    COVERAGES, FIELDS, SEED, binarise, coverage_curve, oof_image_level,
)

N_PERM = 200


def longest_run(curve):
    """Longest run of consecutive coverage points whose delta CI excludes 0."""
    best = cur = 0
    for r in curve:
        if r.get("usable"):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def n_passing(curve):
    return sum(1 for r in curve if r.get("usable"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom",
                    default="artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--fields", default="malformation_ratio,loop_length,capillary_count")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])
    dev = man[man.evaluation_role == "development"].set_index("exam_case_id")
    fold_case = dev["development_fold"].dropna().astype(int)

    Gi = pd.read_csv(args.geom)
    Gi["exam_case_id"] = Gi["exam_case_id"].astype(str)
    assert not (set(Gi.exam_case_id) & locked), "locked case in image geometry"
    idx = pd.read_csv(args.index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    F = np.load(args.dino).astype(np.float32)
    assert len(idx) == len(F), f"index {len(idx)} != features {len(F)}"
    dmap = pd.DataFrame(F)
    dmap["image_path"] = idx["image_path"].astype(str)
    dmap["exam_case_id"] = idx["exam_case_id"]

    Gi["image_path"] = Gi["image_path"].astype(str)
    mrg = Gi.merge(dmap, on=["exam_case_id", "image_path"], how="inner",
                   suffixes=("", "_d"))
    gcols = [c for c in Gi.columns
             if c not in ("exam_case_id", "image_path", "fold")
             and pd.api.types.is_numeric_dtype(Gi[c])]
    XC = np.hstack([mrg[gcols].to_numpy(dtype=np.float32),
                    mrg[list(range(F.shape[1]))].to_numpy(dtype=np.float32)])
    img_case = pd.Index(mrg["exam_case_id"])
    print(f"aligned images: {len(mrg)}  dims={XC.shape[1]}", flush=True)

    out = {"config": {"n_perm": args.n_perm, "seed": SEED,
                      "coverages": COVERAGES,
                      "statistic": "longest run of consecutive usable coverages",
                      "locked_cases_seen": 0,
                      "geom_source": args.geom},
           "fields": {}}

    for field in args.fields.split(","):
        cfg = FIELDS[field]
        y = binarise(dev[field], cfg).dropna()
        y = y[y.index.isin(img_case)]
        prob = oof_image_level(XC, img_case, y, fold_case)
        obs = coverage_curve(y.reindex(prob.index), prob, SEED)
        obs_run, obs_np = longest_run(obs), n_passing(obs)

        rng = np.random.default_rng(SEED)
        runs, nps = [], []
        for b in range(args.n_perm):
            ysh = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
            p = oof_image_level(XC, img_case, ysh, fold_case)
            c = coverage_curve(ysh.reindex(p.index), p, SEED + b)
            runs.append(longest_run(c))
            nps.append(n_passing(c))
            if (b + 1) % 50 == 0:
                print(f"  [{field}] {b + 1}/{args.n_perm}", flush=True)
        runs, nps = np.array(runs), np.array(nps)
        p_run = float((runs >= obs_run).mean()) if obs_run > 0 else 1.0
        p_np = float((nps >= obs_np).mean()) if obs_np > 0 else 1.0
        out["fields"][field] = {
            "observed_longest_run": obs_run,
            "observed_n_passing": obs_np,
            "null_longest_run_mean": round(float(runs.mean()), 3),
            "null_longest_run_p95": int(np.percentile(runs, 95)),
            "null_longest_run_max": int(runs.max()),
            "p_longest_run": round(p_run, 4),
            "null_n_passing_mean": round(float(nps.mean()), 3),
            "p_n_passing": round(p_np, 4),
            "minority_share_by_coverage": [
                r.get("abnormal_share") for r in obs],
        }
        print(f"{field:20s} obs_run={obs_run} p={p_run:.4f} | "
              f"null mean={runs.mean():.2f} p95={np.percentile(runs, 95):.0f} "
              f"max={runs.max()}", flush=True)

    out["limitations"] = [
        "development set only; NOT product capability",
        "the null permutes labels only; it does not create new cases, so it "
        "tests information content, not generalisation to a new cohort",
        f"{args.n_perm} permutations bound the smallest resolvable p at "
        f"{1.0 / args.n_perm:.4f}",
        "a small p means the pattern is not multiplicity; it does NOT mean the "
        "coverage point is calibrated for shipping, which needs held-out data",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
