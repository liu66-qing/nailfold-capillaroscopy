#!/usr/bin/env python
"""Labels-free comparability check for the DELIVERED configuration's features.

Why this exists separately from check_locked_feature_shift.py: that script checks
the geometry+DINOv2 concat store. The delivered configuration uses the 5-pooling
spatial store (artifacts/features_spatial/native), which is a different pipeline.
Its comparability was only ever printed to a console during extraction, so it was
not auditable. This writes it down.

The failure mode being guarded against: if locked features are produced even
slightly differently from development features, the classifier scores badly and
the result gets misread as "the model does not generalise" when the real cause is
that it was shown different-looking inputs. That mistake is unrecoverable once the
locked budget is spent, so the check has to happen before, and it must use NO
labels -- it looks only at feature distributions.

Verdict is USABLE only if every pooling passes all of:
  - same dimensionality as development
  - zero NaN / Inf
  - mean L2 norm ratio (locked/dev) within [0.95, 1.05]
  - no more than 2% of dimensions whose locked mean falls outside dev p1-p99
and the encoder weights hash matches the delivered run exactly.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
EXPECT_WEIGHTS_SHA = ("0b8b82f85de91b424aded121c7e1dcc2b7bc6d0a"
                      "deea651bf73a13307fad8c73")
NORM_RATIO_LO, NORM_RATIO_HI = 0.95, 1.05
MAX_OUT_OF_RANGE_FRAC = 0.02


def load(d, p):
    return np.load(os.path.join(d, "features_%s.npy" % p)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="artifacts/features_spatial/native")
    ap.add_argument("--locked", default="artifacts/features_spatial/native_locked")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out",
                    default="artifacts/features_spatial/native_locked/shift_check.json")
    a = ap.parse_args()

    roles = pd.read_csv(a.manifest)
    roles["exam_case_id"] = roles["exam_case_id"].astype(str)
    dev_ids = set(roles.loc[roles.evaluation_role == "development", "exam_case_id"])
    lk_ids = set(roles.loc[roles.evaluation_role == "locked_test", "exam_case_id"])

    idx_d = pd.read_csv(os.path.join(a.dev, "index.csv"))
    idx_l = pd.read_csv(os.path.join(a.locked, "index.csv"))
    idx_d["exam_case_id"] = idx_d["exam_case_id"].astype(str)
    idx_l["exam_case_id"] = idx_l["exam_case_id"].astype(str)

    leak_dev = sorted(set(idx_d.exam_case_id) & lk_ids)
    leak_lk = sorted(set(idx_l.exam_case_id) & dev_ids)
    assert not leak_dev, f"locked cases inside development store: {leak_dev[:5]}"
    assert not leak_lk, f"development cases inside locked store: {leak_lk[:5]}"

    md = json.load(open(os.path.join(a.dev, "metadata.json")))
    ml = json.load(open(os.path.join(a.locked, "metadata.json")))

    out = {
        "purpose": "prove locked spatial features are comparable to development, "
                   "using no labels, before spending the locked budget",
        "dev_store": a.dev, "locked_store": a.locked,
        "dev_images": int(len(idx_d)), "dev_cases": int(idx_d.exam_case_id.nunique()),
        "locked_images": int(len(idx_l)),
        "locked_cases": int(idx_l.exam_case_id.nunique()),
        "locked_cases_in_dev_store": 0, "dev_cases_in_locked_store": 0,
        "poolings": {},
    }

    for k in ("preset", "resize", "topk", "weights_sha256", "model"):
        vd, vl = md.get(k), ml.get(k)
        out.setdefault("config_match", {})[k] = {
            "dev": vd, "locked": vl, "same": vd == vl}

    sha_d, sha_l = md.get("weights_sha256"), ml.get("weights_sha256")
    out["encoder_hash_matches_delivered"] = bool(
        sha_d == EXPECT_WEIGHTS_SHA and sha_l == EXPECT_WEIGHTS_SHA)

    verdict_ok = out["encoder_hash_matches_delivered"]
    for p in POOLINGS:
        Xd, Xl = load(a.dev, p), load(a.locked, p)
        assert len(Xd) == len(idx_d), f"{p}: dev rows != index"
        assert len(Xl) == len(idx_l), f"{p}: locked rows != index"
        dim_ok = Xd.shape[1] == Xl.shape[1]
        nan_l = int(np.isnan(Xl).sum() + np.isinf(Xl).sum())
        nan_d = int(np.isnan(Xd).sum() + np.isinf(Xd).sum())
        nd = float(np.linalg.norm(Xd, axis=1).mean())
        nl = float(np.linalg.norm(Xl, axis=1).mean())
        ratio = nl / nd if nd else float("nan")
        lo = np.percentile(Xd, 1, axis=0)
        hi = np.percentile(Xd, 99, axis=0)
        lm = Xl.mean(axis=0)
        oor = int(((lm < lo) | (lm > hi)).sum())
        oor_frac = oor / Xd.shape[1]
        ok = bool(dim_ok and nan_l == 0 and nan_d == 0
                  and NORM_RATIO_LO <= ratio <= NORM_RATIO_HI
                  and oor_frac <= MAX_OUT_OF_RANGE_FRAC)
        verdict_ok = verdict_ok and ok
        out["poolings"][p] = {
            "dev_dim": int(Xd.shape[1]), "locked_dim": int(Xl.shape[1]),
            "dim_match": bool(dim_ok),
            "nan_inf_dev": nan_d, "nan_inf_locked": nan_l,
            "mean_l2_dev": round(nd, 4), "mean_l2_locked": round(nl, 4),
            "norm_ratio": round(ratio, 4),
            "dims_out_of_dev_p1_p99": oor,
            "dims_out_of_dev_p1_p99_frac": round(oor_frac, 4),
            "pass": ok,
        }
        print(f"{p:10s} dim {Xd.shape[1]}/{Xl.shape[1]} norm_ratio={ratio:.4f} "
              f"oor={oor}/{Xd.shape[1]} ({oor_frac:.3%}) nan={nan_l} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)

    out["verdict"] = "USABLE" if verdict_ok else "NOT_COMPARABLE"
    out["limitations"] = [
        "this proves the FEATURES are comparable; it says nothing about whether "
        "the classifier generalises",
        "distribution overlap is not distribution identity: a real cohort "
        "difference in prevalence or image quality will still move results",
        "no labels were read by this script",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nverdict:", out["verdict"], "->", a.out)


if __name__ == "__main__":
    main()
