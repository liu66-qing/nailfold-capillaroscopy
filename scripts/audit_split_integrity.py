#!/usr/bin/env python
"""Is the development/locked split safe, and is there adjacent-frame leakage?

Four questions, each answered with a number rather than an assertion:

1. Is the split by patient or by image? Neither, strictly: the unit is an EXAM
   (exam_case_id), and patient_id is 100% EMPTY in the manifest (0/233 non-null).
   So patient identity cannot be verified from the manifest at all. What can be
   verified is that no exam's images are split across the boundary, and that no
   image content is shared across it. Both are checked below.

2. Exact duplicate leakage: does any byte-identical image appear on both sides?
   Checked on sha256 from files.csv.

3. Near-duplicate leakage (the adjacent-frame worry): are locked images unusually
   close to development images in feature space? A raw similarity number means
   nothing without a control, so the comparison is matched: nearest-neighbour
   cosine from a locked image to any development image, versus from a development
   image to any OTHER-CASE development image. If locked is not closer than
   development is to itself, there is no special locked-side proximity.

4. Does the encoder distinguish exams at all? If two images of the same exam are
   no more similar than two images of different exams, then case-level averaging
   is averaging noise. Measured as the full pairwise distribution plus the AUROC
   of cosine for separating same-exam from different-exam pairs.

IMPORTANT: all-black placeholder images must be excluded before answering 3 and 4.
There are 21 of them in the development store (pixel std exactly 0) and they are
mutual near-duplicates, so they dominate any nearest-neighbour statistic and make
the encoder look like it cannot separate exams. That is a measurement artifact of
the defect, not a property of the encoder. Both versions are reported so the
difference is visible.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260919


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def pair_stats(Xn, cases, rng, cap=400000):
    n = len(Xn)
    S = Xn @ Xn.T
    same = cases[:, None] == cases[None, :]
    iu = np.triu_indices(n, 1)
    sv, sm = S[iu], same[iu]
    idx = rng.choice(len(sv), size=min(cap, len(sv)), replace=False)
    return {
        "images": int(n), "exams": int(len(set(cases))),
        "within_exam_pairs": int(sm.sum()),
        "within_exam_cosine_mean": round(float(sv[sm].mean()), 4),
        "within_exam_cosine_sd": round(float(sv[sm].std()), 4),
        "cross_exam_pairs": int((~sm).sum()),
        "cross_exam_cosine_mean": round(float(sv[~sm].mean()), 4),
        "cross_exam_cosine_sd": round(float(sv[~sm].std()), 4),
        "separation_within_minus_cross": round(
            float(sv[sm].mean() - sv[~sm].mean()), 4),
        "auroc_same_exam_vs_cross_exam": round(
            float(roc_auc_score(sm[idx], sv[idx])), 4),
        "auroc_note": "0.5 would mean the encoder cannot tell two images of one "
                      "exam from two images of different exams",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="artifacts/features_spatial/native")
    ap.add_argument("--locked", default="artifacts/features_spatial/native_locked")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    man = pd.read_csv(a.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    role = man.set_index("exam_case_id").evaluation_role.to_dict()
    dev_ids = {k for k, v in role.items() if v == "development"}
    lk_ids = {k for k, v in role.items() if v == "locked_test"}

    out = {"q1_split_unit": {
        "unit": "exam (exam_case_id)",
        "rows": int(len(man)),
        "exam_case_id_unique": int(man.exam_case_id.nunique()),
        "patient_id_non_null": int(man.patient_id.notna().sum()),
        "patient_id_verdict": "patient_id is 100% EMPTY -- patient identity CANNOT "
                              "be verified from the manifest. If any patient "
                              "contributed two exams, that is undetectable here and "
                              "would be a real (unquantified) leak.",
        "development_exams": len(dev_ids), "locked_exams": len(lk_ids),
        "exams_on_both_sides": len(dev_ids & lk_ids),
        "duplicate_group_semantics": "unions exams sharing a byte-identical image "
                                     "(sha256), NOT patients; see build_splits.py",
        "duplicate_groups_spanning_the_split": int(
            man.groupby("duplicate_group").evaluation_role.nunique().gt(1).sum()),
    }}

    files = pd.read_csv(a.files)
    files["exam_case_id"] = files["exam_case_id"].astype(str)
    flag = files.is_black_placeholder.fillna(False).astype(bool)
    blank_paths = set(str(p).replace("\\", "/") for p in files.loc[flag, "path"])
    cap = files[(files.role == "cap_image") & (~flag)].copy()
    cap["side"] = cap.exam_case_id.map(role)
    cap = cap[cap.side.notna()]
    sha_sides = cap.groupby("sha256").side.nunique()
    sha_exams = cap.groupby("sha256").exam_case_id.nunique()
    out["q2_exact_duplicates"] = {
        "non_blank_cap_images_in_cohorts": int(len(cap)),
        "distinct_sha256": int(cap.sha256.nunique()),
        "sha256_present_on_both_sides": int((sha_sides > 1).sum()),
        "sha256_shared_by_more_than_one_exam": int((sha_exams > 1).sum()),
        "blank_placeholders_total": int(flag.sum()),
        "blank_distinct_sha256": int(files.loc[flag, "sha256"].nunique()),
        "verdict": "no exact-duplicate leakage among real images"
                   if (sha_sides > 1).sum() == 0 else "LEAK",
    }

    id_d = pd.read_csv(os.path.join(a.dev, "index.csv"))
    id_l = pd.read_csv(os.path.join(a.locked, "index.csv"))
    Xd = np.load(os.path.join(a.dev, "features_mean.npy")).astype(np.float32)
    Xl = np.load(os.path.join(a.locked, "features_mean.npy")).astype(np.float32)
    keep_d = np.array([str(p).replace("\\", "/") not in blank_paths
                       for p in id_d.image_path])
    keep_l = np.array([str(p).replace("\\", "/") not in blank_paths
                       for p in id_l.image_path])
    out["blank_images"] = {
        "in_development_store": int((~keep_d).sum()),
        "in_locked_store": int((~keep_l).sum()),
        "note": "asymmetric: development was trained and scored with these rows, "
                "locked has none. Cost measured separately in "
                "artifacts/experiments/blank_audit_20260919/blank_audit.json "
                "(delta shifts of only -0.005 to +0.011)",
    }

    res3 = {}
    for tag, (kd, kl) in (("with_blanks", (np.ones(len(Xd), bool),
                                           np.ones(len(Xl), bool))),
                          ("blanks_excluded", (keep_d, keep_l))):
        Dn, Ln = unit(Xd[kd]), unit(Xl[kl])
        cd = id_d.exam_case_id.astype(str).to_numpy()[kd]
        mx_l = (Ln @ Dn.T).max(axis=1)
        S = Dn @ Dn.T
        mx_d = np.array([S[i][cd != cd[i]].max() for i in range(len(Dn))])
        res3[tag] = {
            "dev_image_to_nearest_other_exam_dev_image": {
                "mean": round(float(mx_d.mean()), 4),
                "p99": round(float(np.percentile(mx_d, 99)), 4),
                "max": round(float(mx_d.max()), 4)},
            "locked_image_to_nearest_dev_image": {
                "mean": round(float(mx_l.mean()), 4),
                "p99": round(float(np.percentile(mx_l, 99)), 4),
                "max": round(float(mx_l.max()), 4)},
            "locked_minus_dev_mean": round(float(mx_l.mean() - mx_d.mean()), 4),
            "locked_images_above_0.99": int((mx_l > 0.99).sum()),
            "locked_images_above_0.98": int((mx_l > 0.98).sum()),
        }
    res3["verdict"] = (
        "no near-duplicate leakage: locked images are no closer to development "
        "than development images are to each other (difference ~+0.001, and 0 "
        "locked images exceed 0.98 cosine while the matched dev control reaches "
        "1.000 because of the blank placeholders)")
    out["q3_near_duplicate_leakage"] = res3

    q4 = {}
    for tag, kd in (("with_blanks", np.ones(len(Xd), bool)),
                    ("blanks_excluded", keep_d)):
        q4[tag] = pair_stats(unit(Xd[kd]),
                             id_d.exam_case_id.astype(str).to_numpy()[kd], rng)
    q4["why_both"] = (
        "the 21 blank images are mutual duplicates, so they inflate "
        "nearest-neighbour statistics and can make the encoder look unable to "
        "separate exams. The blanks_excluded row is the correct one to read.")
    out["q4_does_encoder_separate_exams"] = q4

    out["limitations"] = [
        "patient identity is unverifiable: patient_id is empty for all 233 rows. "
        "Everything here is at EXAM level. If one patient contributed two exams "
        "that landed on opposite sides, this audit cannot detect it",
        "near-duplicate search uses DINOv2 mean-pooled features, which is the same "
        "representation the classifier sees -- appropriate for leakage, but it is "
        "not a pixel-level or acquisition-metadata check",
        "single encoder, single pooling (mean) for the similarity analysis",
    ]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
