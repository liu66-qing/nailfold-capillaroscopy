#!/usr/bin/env python
"""How much did 21 all-black training images cost the development numbers?

Found while answering a question about patient-level split integrity: 21 images in
the delivered configuration's development feature store have pixel std exactly 0 --
they are all-black placeholders (files.csv flags them is_black_placeholder, and the
pixels confirm it: mean 0.0, std 0.0, 1 unique value). They are 1.23% of the 1708
development image rows but touch 21 of 186 cases (11.3%), because the delivered
configuration trains on image rows and averages probabilities to the case.

The locked store has ZERO of them, so this is an asymmetry between the two cohorts
and not a shared nuisance. That matters for reading the locked result: development
was fitted and scored with blank rows in it, locked was not.

This script does NOT change the delivered configuration. It refits the SAME
configuration twice -- once as delivered, once with blank rows dropped -- and
reports the difference. Dropping bad inputs is a defect fix, not a hyperparameter,
but the honest way to present it is still "here is what it was, here is what it
becomes", with both numbers shown.

Development-only. Locked is not touched, because it has no blank rows to drop and
because re-scoring it after a change is exactly the test-set-tuning this project
has rules against.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
FIELDS = ["clarity", "subpapillary_venous_plexus", "exudation", "blood_color",
          "malformation_ratio"]


def fit_predict(Xtr, ytr, Xts):
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(np.mean(ytr)))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def oof_case_probs(feats, cases, folds, y_case, keep_mask):
    """Case-level OOF probability, 5-pooling equal-weight, case-level folds."""
    out = {}
    for p in POOLINGS:
        X = feats[p]
        pr = np.full(len(X), np.nan)
        for k in sorted(folds.unique()):
            te_cases = set(folds[folds == k].index)
            tr = np.array([c not in te_cases for c in cases]) & keep_mask
            te = np.array([c in te_cases for c in cases]) & keep_mask
            ytr = y_case.reindex(cases[tr]).to_numpy()
            ok = ~pd.isna(ytr)
            if ok.sum() < 10 or te.sum() == 0:
                continue
            Xtr = X[tr][ok]
            pr[te] = fit_predict(Xtr, ytr[ok].astype(int), X[te])
        out[p] = pr
    stack = np.vstack([out[p] for p in POOLINGS])
    mean_img = np.nanmean(stack, axis=0)
    df = pd.DataFrame({"case": cases, "p": mean_img})
    df = df[keep_mask & ~pd.isna(df.p)]
    return df.groupby("case").p.mean()


def boot_delta(yt, yp, majority, rng):
    n = len(yt)
    acc = float((yp == yt).mean())
    base = float((yt == majority).mean())
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = (yp[i] == yt[i]).mean() - (yt[i] == majority).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4), "baseline": round(base, 4),
            "delta": round(acc - base, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "passes": bool(lo > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="artifacts/features_spatial/native")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--oof-dir",
                    default="artifacts/experiments/threshold_tuned_20260916")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    dev = man[man.evaluation_role == "development"]
    folds = dev.set_index("exam_case_id")["development_fold"].dropna().astype(int)

    files = pd.read_csv(a.files)
    flag = files.is_black_placeholder.fillna(False).astype(bool)
    blank_paths = set(str(p).replace("\\", "/") for p in files.loc[flag, "path"])
    blank_sha = sorted(set(files.loc[flag, "sha256"]))

    idx = pd.read_csv(os.path.join(a.feat, "index.csv"))
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    ip = [str(x).replace("\\", "/") for x in idx.image_path]
    is_blank = np.array([p in blank_paths for p in ip])
    cases = idx.exam_case_id.to_numpy()

    feats = {p: np.load(os.path.join(a.feat, "features_%s.npy" % p)).astype(np.float32)
             for p in POOLINGS}

    labels = {}
    for fn in sorted(os.listdir(a.oof_dir)):
        if fn.endswith("__binary_oof.csv"):
            d = pd.read_csv(os.path.join(a.oof_dir, fn)).drop_duplicates("case_id")
            s = d.set_index("case_id").truth.astype(float)
            s.index = s.index.astype(str)
            labels[fn.split("__")[0]] = s

    out = {
        "what": "21 all-black placeholder images are training rows in the "
                "delivered configuration's development store; locked has none",
        "evidence": {
            "blank_images_in_dev_store": int(is_blank.sum()),
            "dev_images_total": int(len(idx)),
            "blank_frac_of_images": round(float(is_blank.mean()), 4),
            "dev_cases_total": int(idx.exam_case_id.nunique()),
            "dev_cases_touched": int(len(set(cases[is_blank]))),
            "blank_images_in_locked_store": 0,
            "distinct_sha256_among_placeholders": len(blank_sha),
            "pixel_check": "mean 0.0, std 0.0, 1 unique grey value",
            "note": "identical black inputs do NOT give identical feature vectors "
                    "(max pairwise L2 15.14 on mean-pooling), so the encoder emits "
                    "position/padding-dependent noise rather than one constant "
                    "vector -- i.e. these rows carry spurious variance, not a "
                    "single harmless duplicate",
        },
        "fields": {},
    }

    for field in FIELDS:
        y = labels.get(field)
        if y is None:
            continue
        y = y[y.index.isin(set(dev.exam_case_id))]
        maj = float(np.bincount(y.to_numpy().astype(int)).argmax())
        res = {}
        for tag, mask in (("as_delivered", np.ones(len(idx), bool)),
                          ("blank_dropped", ~is_blank)):
            pr = oof_case_probs(feats, cases, folds, y, mask)
            common = y.index.intersection(pr.index)
            yt = y.loc[common].to_numpy()
            yp = (pr.loc[common] > 0.5).astype(float).to_numpy()
            res[tag] = boot_delta(yt, yp, maj, np.random.default_rng(SEED))
        res["delta_change"] = round(
            res["blank_dropped"]["delta"] - res["as_delivered"]["delta"], 4)
        res["cases_with_blank_and_label"] = int(
            len(set(cases[is_blank]) & set(y.index)))
        out["fields"][field] = res
        print(f"{field:28s} as-delivered d={res['as_delivered']['delta']:+.4f} "
              f"-> blank-dropped d={res['blank_dropped']['delta']:+.4f} "
              f"(change {res['delta_change']:+.4f}) "
              f"n={res['as_delivered']['n']}/{res['blank_dropped']['n']}",
              flush=True)

    out["limitations"] = [
        "development only: locked has no blank rows to drop, and re-scoring locked "
        "after a change would be test-set tuning",
        "this is a defect fix, not a tuning knob, but both numbers are reported so "
        "the change is visible rather than silently improving the record",
        "21 of 186 cases are touched; each affected case has exactly 1 blank image "
        "out of 8-15, so per-case dilution is 7-13% of that case's images",
        "a delta that improves here is still a development number and still not "
        "product capability",
    ]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
