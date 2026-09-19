#!/usr/bin/env python
"""Leave-one-archive-out: is the signal in the 4 passing fields archive-specific?

This is the ONE genuinely untested proposal in the external review. Group 3
stratified locked PREDICTIONS by archive, but training has always pooled all
three archives, so a field could look fine while its advantage lives in one
archive only. Here each archive is held out ENTIRELY -- no case from it is in
the training fold -- and the field is scored on it alone.

Three directions per field: train a2+a3 -> test a1; a1+a3 -> a2; a1+a2 -> a3.
Configuration is the fixed delivered one, unchanged, so LOAO numbers are
directly comparable to the pooled OOF numbers in
artifacts/evidence/allfields_20260919/.

The baseline is the majority answer OF THE TRAINING ARCHIVES -- which is what a
deployed model would carry -- not the majority of the test archive. Both are
recorded; the training-archive one is the honest ruler because the test
archive's own prevalence is unknown at deployment time.

Also runs a grey-statistics arm on the same splits. If grey statistics transfer
as well as DINOv2 does, the field is a colour/exposure shortcut rather than a
microvascular one, and the archive is the confound.

DEVELOPMENT ONLY. locked-47 is never read. This does NOT create external
validation: all three archives are the same site and the same acquisition
source. It tests archive shift, nothing more.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]

FIELDS = {
    "clarity": {0: ["清晰"], 1: ["不清", "模糊"]},
    "subpapillary_venous_plexus": {0: ["不见"],
                                   1: ["可见1排", "可见2排", ">2排,扩张"]},
    "exudation": {0: ["无"], 1: ["+", "++", "+++"]},
    "blood_color": {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]},
    "malformation_ratio": {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]},
    "microthrombus": {0: ["无"], 1: ["1--2", ">2"]},
    "capillary_count": {0: [">=7"], 1: ["5--6", "3--4", "<1"]},
}
ARCHIVES = ["recovered_archive1", "recovered_archive2", "recovered_archive3"]


def boot_delta(yt, yp, maj, rng):
    n = len(yt)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = (yp[i] == yt[i]).mean() - (yt[i] == maj).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def fit_arm(Xtr, ytr, Xts):
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def grey_table(files_csv, ids):
    f = pd.read_csv(files_csv)
    f["exam_case_id"] = f.exam_case_id.astype(str)
    cols = [c for c in f.columns
            if any(k in c for k in ("mean", "std", "p10", "p50", "p90",
                                    "min", "max", "median"))
            and pd.api.types.is_numeric_dtype(f[c])]
    g = f[f.exam_case_id.isin(ids)].groupby("exam_case_id")[cols].mean()
    return g.fillna(g.median()), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="artifacts/features_spatial/native")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--files-csv", default="artifacts/manifest/files.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    m = m.set_index("exam_case_id")
    dev = m[m.evaluation_role == "development"]
    lk = set(m.index[m.evaluation_role == "locked_test"])
    arch = dev.archive_id.to_dict()

    ix = pd.read_csv(os.path.join(a.store, "index.csv"))
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    assert not set(ix.exam_case_id) & lk, "locked case present"
    feats = {p: np.load(os.path.join(a.store, f"features_{p}.npy"))
                 .astype(np.float32) for p in POOLINGS}

    grey, grey_cols = grey_table(a.files_csv, set(dev.index))
    print("grey features:", len(grey_cols), "cases:", len(grey))

    out = {}
    for field, bm in FIELDS.items():
        y = {}
        for c in dev.index:
            s = str(dev.at[c, field]).strip()
            for lab, vals in bm.items():
                if s in vals:
                    y[c] = float(lab)
        y_case = pd.Series(y, dtype=float)
        rec = {"n_labelled": int(len(y_case)), "directions": {}}
        print(f"\n=== {field}  n={len(y_case)}")

        for held in ARCHIVES:
            te_ids = [c for c in y_case.index if arch.get(c) == held]
            tr_ids = [c for c in y_case.index if arch.get(c) != held]
            if len(te_ids) < 20 or len(set(y_case[tr_ids])) < 2:
                rec["directions"][held] = {"status": "too_small",
                                           "n_test": len(te_ids)}
                continue
            ytr_c, yts_c = y_case[tr_ids], y_case[te_ids]
            trm = ix.exam_case_id.isin(tr_ids).to_numpy()
            tem = ix.exam_case_id.isin(te_ids).to_numpy()
            ytr_img = ytr_c.reindex(ix.exam_case_id[trm]).to_numpy()

            # --- DINOv2 arm, the delivered configuration, unchanged
            ps = [fit_arm(feats[p][trm], ytr_img, feats[p][tem])
                  for p in POOLINGS]
            s = pd.Series(np.mean(ps, axis=0),
                          index=ix.exam_case_id[tem].to_numpy()
                          ).groupby(level=0).mean()
            order = [c for c in te_ids if c in s.index]
            prob = s.reindex(order).to_numpy()
            yts = yts_c.reindex(order).to_numpy()
            yp = (prob >= 0.5).astype(float)

            maj_tr = float(ytr_c.mode().iloc[0])
            maj_te = float(yts_c.mode().iloc[0])
            acc = float((yp == yts).mean())
            b_tr = float((yts == maj_tr).mean())
            b_te = float((yts == maj_te).mean())
            lo, hi = boot_delta(yts, yp, maj_tr, np.random.default_rng(SEED))
            two = len(set(yts)) == 2

            # --- grey-statistics arm, same split
            gtr = grey.reindex(tr_ids).dropna()
            gts = grey.reindex(order).dropna()
            gacc = None
            if len(gtr) > 30 and len(gts) > 10:
                gp = fit_arm(gtr.to_numpy(), y_case.reindex(gtr.index).to_numpy(),
                             gts.to_numpy())
                gy = y_case.reindex(gts.index).to_numpy()
                gacc = round(float(((gp >= 0.5).astype(float) == gy).mean()), 4)

            d = {
                "n_train": len(tr_ids), "n_test": len(order),
                "accuracy": round(acc, 4),
                "baseline_train_archive_majority": round(b_tr, 4),
                "delta_vs_train_majority": round(acc - b_tr, 4),
                "ci95": [lo, hi], "passes": bool(lo > 0),
                "baseline_test_archive_own_majority": round(b_te, 4),
                "delta_vs_test_own_majority": round(acc - b_te, 4),
                "train_abnormal_share": round(float(ytr_c.mean()), 4),
                "test_abnormal_share": round(float(yts_c.mean()), 4),
                "prevalence_shift": round(float(yts_c.mean() - ytr_c.mean()), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(
                    yts.astype(int), yp.astype(int))), 4) if two else None,
                "auroc": round(float(roc_auc_score(yts, prob)), 4) if two else None,
                "collapsed_to_one_class": bool(len(set(yp)) == 1),
                "grey_stats_accuracy_same_split": gacc,
                "grey_matches_or_beats_dinov2": (None if gacc is None
                                                else bool(gacc >= acc)),
            }
            rec["directions"][held] = d
            print(f"  hold {held[-8:]}: n={d['n_test']:3d} acc={acc:.4f} "
                  f"base(tr)={b_tr:.4f} d={acc-b_tr:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
                  f"{'PASS' if d['passes'] else 'fail'} "
                  f"AUROC={d['auroc']} grey={gacc} shift={d['prevalence_shift']:+.3f}")

        ok = [v for v in rec["directions"].values() if v.get("passes")]
        rec["n_directions_passing"] = len(ok)
        rec["passes_all_three_directions"] = bool(len(ok) == 3)
        aur = [v["auroc"] for v in rec["directions"].values()
               if v.get("auroc") is not None]
        rec["auroc_min_across_archives"] = min(aur) if aur else None
        rec["auroc_max_across_archives"] = max(aur) if aur else None
        out[field] = rec

    res = {
        "what_this_tests": "archive shift: each archive held out of TRAINING "
                           "entirely, then scored on it alone",
        "what_this_is_not": "external validation. All three archives are the "
                            "same site and the same acquisition source; there "
                            "is no second site anywhere in this project.",
        "config": "fixed delivered configuration, unchanged; baseline is the "
                  "TRAINING archives' majority answer",
        "archive_case_counts_development": {k: int(v) for k, v in
                                            dev.archive_id.value_counts().items()},
        "locked_cases_seen": 0,
        "fields": out,
        "summary": {f: {"n_directions_passing": v["n_directions_passing"],
                        "all_three": v["passes_all_three_directions"],
                        "auroc_range": [v["auroc_min_across_archives"],
                                        v["auroc_max_across_archives"]]}
                    for f, v in out.items()},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\n--- fields passing all three directions:",
          [f for f, v in out.items() if v["passes_all_three_directions"]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
