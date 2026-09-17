#!/usr/bin/env python
"""Close the 6-field gap: the required fields not covered by the binary sweep.

artifacts/v1_field_specification.csv requires 17 fields. The image-level
ensemble round evaluated 11 of them. This script covers the remaining 6 under
the rulers appropriate to their type, so the "how many of the 17 are usable"
question has an answer for every row.

  papilla                  3-class (平坦 / 浅波纹状 / 波纹状)
    ruler: accuracy - single fixed majority answer. Same as the binary fields;
    a 3-class problem simply has a lower majority baseline (77/185 = 0.416),
    so the arithmetic ceiling is higher and the bar is harder to clear by luck.

  afferent/efferent/apex_diameter, loop_length   continuous
    ruler: MAE vs the single fixed median answer of the training folds, and R2.
    A regression cannot be scored against a majority class, so the honest
    no-model bar is "always answer the training median". delta_mae > 0 means
    the model beats that; verdict needs the CI upper bound of delta_mae < 0
    ... expressed here as improvement = base_mae - model_mae, needing CI
    lower bound > 0, keeping the same sign convention as the other fields.

    Two independent limits are recorded and neither is a model problem:
    label granularity (afferent has 24 distinct values over 165 cases, i.e.
    the label is quantised to whole microns) and the absence of any external
    micron calibration. No absolute-micron accuracy claim is made and
    calibration_factors.json is NOT used as evidence (it is back-fitted from
    labels and holds four mutually inconsistent values).

  flow_velocity            not evaluated, by data
    The specification itself records it as a placeholder with default=None and
    "placeholder has no visual measurement basis". There is no label column to
    score, so it is reported as NO_LABEL rather than as a failed field. The
    related flow_state labels exist but are a different, 7-class variable and
    are not a substitute.

Model and protocol are inherited unchanged from the delivered configuration:
image-level training on 1708 image rows, 5-pooling equal-weight ensemble,
logreg C=0.03 / PCA 64 unweighted for classification and ridge for regression,
case-level folds from development_fold, 2000 case-level bootstrap resamples
with the baseline recomputed inside each resample, no selection anywhere.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
RIDGE_ALPHA = 10.0
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
CLASS_FIELDS = ["papilla"]
NUMERIC_FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter",
                  "loop_length"]
NO_LABEL_FIELDS = ["flow_velocity"]


def load_oof(oof_dir, field, kind):
    path = os.path.join(oof_dir, "%s__%s_oof.csv" % (field, kind))
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).drop_duplicates("case_id")
    return d.set_index("case_id").truth


def fit_predict_cls(Xtr, ytr, Xts):
    classes = np.unique(ytr)
    if len(classes) < 2:
        return np.full((len(Xts), 1), 1.0), classes
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts), pipe.classes_


def fit_predict_reg(Xtr, ytr, Xts):
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         Ridge(alpha=RIDGE_ALPHA))
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xts)


def oof_classification(feats, img_case, y, folds):
    """Image-level training, 5-pooling equal-weight ensemble, multi-class."""
    cases = y.index.to_numpy()
    keep = np.isin(img_case, cases)
    ci = img_case[keep]
    yi = y.reindex(ci).to_numpy()
    fi = folds.reindex(ci).to_numpy(dtype=float)
    classes = np.unique(y.dropna().to_numpy())
    acc = {c: pd.Series(0.0, index=cases, dtype=float) for c in classes}
    seen = pd.Series(0, index=cases, dtype=int)

    for pool in POOLINGS:
        Xi = feats[pool][keep]
        prob = pd.DataFrame(0.0, index=cases, columns=classes, dtype=float)
        got = pd.Series(False, index=cases)
        for te in sorted(set(fi[~np.isnan(fi)])):
            tr, ts = fi != te, fi == te
            if tr.sum() == 0 or ts.sum() == 0:
                continue
            pr, cls = fit_predict_cls(Xi[tr], yi[tr], Xi[ts])
            df = pd.DataFrame(pr, index=ci[ts], columns=cls)
            df = df.groupby(level=0).mean().reindex(columns=classes, fill_value=0.0)
            prob.loc[df.index, classes] = df[classes].to_numpy()
            got.loc[df.index] = True
        for c in classes:
            acc[c] = acc[c].add(prob[c].where(got, 0.0), fill_value=0.0)
        seen = seen.add(got.astype(int), fill_value=0)

    P = pd.DataFrame({c: acc[c] for c in classes})
    pred = pd.Series(np.where(seen.to_numpy() > 0,
                              P.columns.to_numpy()[P.to_numpy().argmax(1)], None),
                     index=cases, dtype=object)
    return pred


def oof_regression(feats, img_case, y, folds):
    cases = y.index.to_numpy()
    keep = np.isin(img_case, cases)
    ci = img_case[keep]
    yi = y.reindex(ci).to_numpy(dtype=float)
    fi = folds.reindex(ci).to_numpy(dtype=float)
    parts, base = [], pd.Series(np.nan, index=cases, dtype=float)

    for pool in POOLINGS:
        Xi = feats[pool][keep]
        p = pd.Series(np.nan, index=cases, dtype=float)
        for te in sorted(set(fi[~np.isnan(fi)])):
            tr, ts = fi != te, fi == te
            if tr.sum() == 0 or ts.sum() == 0:
                continue
            pr = fit_predict_reg(Xi[tr], yi[tr], Xi[ts])
            s = pd.Series(pr, index=ci[ts]).groupby(level=0).mean()
            p.loc[s.index] = s.to_numpy()
            if pool == POOLINGS[0]:
                # no-model bar: the fixed training-fold median
                base.loc[s.index] = float(np.median(yi[tr]))
        parts.append(p)
    return pd.concat(parts, axis=1).mean(axis=1), base


def score_cls(y, pred, rng):
    ya = y.to_numpy()
    pa = pred.reindex(y.index).to_numpy()
    ok = pd.notna(ya) & pd.notna(pa)
    ya, pa = ya[ok], pa[ok]
    n = len(ya)
    vc = pd.Series(ya).value_counts(normalize=True)
    acc = float((pa == ya).mean())
    base = float(vc.max())
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ys, ps = ya[s], pa[s]
        d[b] = (ps == ys).mean() - pd.Series(ys).value_counts(normalize=True).max()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"kind": "classification", "n": n, "n_classes": int(len(vc)),
            "accuracy": acc, "baseline_constant": base,
            "delta_vs_constant": acc - base,
            "delta_ci95": [float(lo), float(hi)], "passes": bool(lo > 0),
            "class_distribution": {str(k): float(v) for k, v in vc.items()}}


def score_reg(y, pred, base, rng):
    ya = y.to_numpy(dtype=float)
    pa = pred.reindex(y.index).to_numpy(dtype=float)
    ba = base.reindex(y.index).to_numpy(dtype=float)
    ok = ~np.isnan(ya) & ~np.isnan(pa) & ~np.isnan(ba)
    ya, pa, ba = ya[ok], pa[ok], ba[ok]
    n = len(ya)
    mae_m = float(np.abs(pa - ya).mean())
    mae_b = float(np.abs(ba - ya).mean())
    ss = float(((ya - pa) ** 2).sum())
    st = float(((ya - ya.mean()) ** 2).sum())
    imp = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        imp[b] = (np.abs(ba[s] - ya[s]).mean() - np.abs(pa[s] - ya[s]).mean())
    lo, hi = np.percentile(imp, [2.5, 97.5])
    return {"kind": "regression", "n": n, "mae_model": mae_m,
            "mae_fixed_median_baseline": mae_b,
            "mae_improvement": mae_b - mae_m,
            "mae_improvement_ci95": [float(lo), float(hi)],
            "passes": bool(lo > 0),
            "r2": 1.0 - ss / st if st > 0 else float("nan"),
            "label_distinct_values": int(pd.Series(ya).nunique()),
            "label_granularity_note":
                "labels are quantised; distinct values vs n bounds achievable MAE",
            "no_micron_claim": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    roles = pd.read_csv(a.roles)
    folds = roles.set_index("exam_case_id").development_fold
    locked = set(folds[folds.isna()].index)
    folds = folds.dropna()
    assert len(folds) == 186, len(folds)

    idx = pd.read_csv(os.path.join(a.feat_dir, "index.csv"))
    img_case = idx.exam_case_id.to_numpy()
    assert not (set(img_case) & locked), "locked case in features"
    feats = {p: np.load(os.path.join(a.feat_dir, "features_%s.npy" % p))
             for p in POOLINGS}

    rng = np.random.default_rng(SEED)
    out = {"config": {"C": C_FIXED, "pca_dim": DIM_FIXED, "ridge_alpha": RIDGE_ALPHA,
                      "n_boot": N_BOOT, "seed": SEED, "members": POOLINGS,
                      "weights": "equal", "level": "image", "selection": "none",
                      "feat_dir": a.feat_dir},
           "locked_cases_seen": 0, "fields": {}}

    for field in CLASS_FIELDS:
        y = load_oof(a.oof_dir, field, "original_classes")
        if y is None:
            out["fields"][field] = {"kind": "classification", "verdict": "NO_LABEL"}
            print("%-22s NO_LABEL" % field)
            continue
        y = y.reindex(folds.index).dropna()
        pred = oof_classification(feats, img_case, y, folds)
        r = score_cls(y, pred, rng)
        r["verdict"] = "USABLE" if r["passes"] else "NOT_USABLE"
        out["fields"][field] = r
        print("%-22s %d-class acc %.3f base %.3f  delta %+.3f CI[%+.3f,%+.3f]  %s"
              % (field, r["n_classes"], r["accuracy"], r["baseline_constant"],
                 r["delta_vs_constant"], r["delta_ci95"][0], r["delta_ci95"][1],
                 r["verdict"]), flush=True)

    for field in NUMERIC_FIELDS:
        y = load_oof(a.oof_dir, field, "numeric")
        if y is None:
            out["fields"][field] = {"kind": "regression", "verdict": "NO_LABEL"}
            print("%-22s NO_LABEL" % field)
            continue
        y = pd.to_numeric(y, errors="coerce").reindex(folds.index).dropna()
        pred, base = oof_regression(feats, img_case, y, folds)
        r = score_reg(y, pred, base, rng)
        r["verdict"] = "USABLE" if r["passes"] else "NOT_USABLE"
        out["fields"][field] = r
        print("%-22s MAE model %.2f vs fixed-median %.2f  improve %+.2f "
              "CI[%+.2f,%+.2f]  R2 %.3f  distinct %d  %s"
              % (field, r["mae_model"], r["mae_fixed_median_baseline"],
                 r["mae_improvement"], r["mae_improvement_ci95"][0],
                 r["mae_improvement_ci95"][1], r["r2"],
                 r["label_distinct_values"], r["verdict"]), flush=True)

    for field in NO_LABEL_FIELDS:
        out["fields"][field] = {
            "kind": "placeholder", "verdict": "NO_LABEL",
            "reason": ("v1_field_specification.csv records this field as a fixed "
                       "compatibility rule with default=None and 'placeholder has "
                       "no visual measurement basis'. There is no label column to "
                       "score. flow_state exists but is a different 7-class "
                       "variable and is not a substitute.")}
        print("%-22s NO_LABEL (placeholder, no visual measurement basis)" % field)

    out["limitations"] = [
        "development set only (186 cases); not product capability",
        "regression fields: no external micron calibration exists; no absolute "
        "micron accuracy is claimed and calibration_factors.json is NOT used as "
        "evidence (it is back-fitted from labels and holds four inconsistent values)",
        "regression labels are quantised to whole units, which bounds achievable "
        "MAE independently of the model",
        "case labels broadcast to images are noisy by construction",
        "papilla is scored on the original 3 classes; no class merging was applied",
        "flow_velocity is unevaluable from data, not measured-and-failed",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()


