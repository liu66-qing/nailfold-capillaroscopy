#!/usr/bin/env python
"""THE COMPLETE PER-FIELD STATUS TABLE: all 21 fields, one fixed configuration.

Why this exists. The existing all-field numbers
(artifacts/experiments/full_field_fix_20260916/summary.json) were produced with
PER-FOLD SOURCE AND MODEL SELECTION -- each fold picked instance-vs-dino and
trees-vs-linear by validation score. That inflates every delta and makes fields
incomparable to one another. This script runs ALL fields through the SINGLE
delivered configuration so that one number means the same thing in every row.

Configuration, fixed, identical for every field:
  frozen DINOv2 ViT-B/14 -> 5 spatial poolings -> StandardScaler -> PCA(64)
  -> LogisticRegression(C=0.03); image rows; mean of pooling probabilities then
  mean over a case's images; threshold 0.5; no class weights; no selection.

Binary fields: accuracy minus the single-constant-answer baseline, bootstrap CI.
Multiclass fields: accuracy and balanced accuracy vs the majority-class constant.
Numeric fields: MAE vs the median-constant baseline, and the label granularity
that bounds it.

DEVELOPMENT ONLY. locked-47 is never read. Development numbers are NOT product
capability; they are the upper bound of what this project can claim to have
measured internally.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
MISSING = {"nan", "", "None", "NaN"}

# every rule below was RECOVERED by joining each *__binary_oof.csv truth column
# back to the raw manifest value -- they are what was actually used, not a guess
BINARY_MAP = {
    "clarity": {0: ["清晰"], 1: ["不清", "模糊"]},
    "capillary_count": {0: [">=7"], 1: ["5--6", "3--4", "<1"]},
    "crossing_ratio": {0: ["<=30%"], 1: ["30--60%", "60--80%", ">80%"]},
    "malformation_ratio": {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]},
    "vasomotion": {0: ["0--1"], 1: ["2--4"]},
    "rbc_aggregation": {0: ["无", "[无]"], 1: ["轻度", "中度", "重度"]},
    "wbc_count": {0: ["1--30"], 1: [">30"]},
    "microthrombus": {0: ["无"], 1: ["1--2", ">2"]},
    "blood_color": {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]},
    "exudation": {0: ["无"], 1: ["+", "++", "+++"]},
    "hemorrhage": {0: ["无"], 1: ["1--2"]},
    "subpapillary_venous_plexus": {0: ["不见"],
                                   1: ["可见1排", "可见2排", ">2排,扩张"]},
    "sweat_duct": {0: ["0--2", "0--2个/-指甲襞"], 1: ["3--4"]},
    "overall_assessment": {0: ["正常", "大致正常"],
                           1: ["轻度异常", "中度异常", "重度异常"]},
}

# fields with no defensible binary cut -> reported as multiclass only
MULTICLASS_ONLY = ["flow_state", "papilla"]

# fields whose label is a number, or a ratio of two others
NUMERIC = ["afferent_diameter", "efferent_diameter", "apex_diameter",
           "loop_length", "flow_speed_um_s"]
DERIVED = ["output_input_ratio"]


def to_float(s):
    try:
        return float(str(s).strip())
    except Exception:
        return np.nan


def fit_predict(Xtr, ytr, Xts, multiclass=False):
    if len(set(ytr)) < 2:
        return (np.full(len(Xts), float(ytr[0])) if multiclass
                else np.full(len(Xts), float(np.mean(ytr))))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    if multiclass:
        return pipe.predict(Xts).astype(float)
    return pipe.predict_proba(Xts)[:, 1]


def oof_probs(feats, dm, ix, y_case, folds, order, multiclass=False):
    """Case-level OOF over the fixed 5-pooling ensemble."""
    cases = ix.exam_case_id[dm].to_numpy()
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    fs = pd.Series([folds.get(c, np.nan) for c in order], index=order)
    for fo in sorted(fs.dropna().unique()):
        te = set(fs.index[fs == fo])
        tr = set(order) - te
        trm = dm & ix.exam_case_id.isin(tr).to_numpy()
        tem = dm & ix.exam_case_id.isin(te).to_numpy()
        if trm.sum() == 0 or tem.sum() == 0:
            continue
        ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
        if multiclass:
            # majority vote over poolings on predicted class
            preds = np.stack([fit_predict(feats[p][trm], ytr, feats[p][tem],
                                          True) for p in POOLINGS])
            df = pd.DataFrame(preds.T, index=ix.exam_case_id[tem].to_numpy())
            for c, grp in df.groupby(level=0):
                vals, cnt = np.unique(grp.to_numpy().ravel(), return_counts=True)
                out[pos[c]] = vals[cnt.argmax()]
        else:
            ps = [fit_predict(feats[p][trm], ytr, feats[p][tem])
                  for p in POOLINGS]
            s = pd.Series(np.mean(ps, axis=0),
                          index=ix.exam_case_id[tem].to_numpy()
                          ).groupby(level=0).mean()
            for c, v in s.items():
                if c in pos:
                    out[pos[c]] = v
    return out


def boot(yt, yp, maj, rng):
    n = len(yt)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = (yp[i] == yt[i]).mean() - (yt[i] == maj).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4), bool(lo > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="artifacts/features_spatial/native")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    m = m.set_index("exam_case_id")
    dev_ids = list(m.index[m.evaluation_role == "development"])
    lk_ids = set(m.index[m.evaluation_role == "locked_test"])
    folds = m.development_fold.to_dict()

    ix = pd.read_csv(os.path.join(a.store, "index.csv"))
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    assert not set(ix.exam_case_id) & lk_ids, "locked case in the dev store"
    feats = {p: np.load(os.path.join(a.store, f"features_{p}.npy"))
                 .astype(np.float32) for p in POOLINGS}

    rows, detail = [], {}

    def raw_dist(field, ids):
        v = m[field].astype(str).str.strip().reindex(ids)
        present = v[~v.isin(MISSING)]
        return present.value_counts()

    # ---------- binary fields ------------------------------------------------
    for field, bm in BINARY_MAP.items():
        y = {}
        for c in dev_ids:
            s = str(m.at[c, field]).strip()
            for lab, vals in bm.items():
                if s in vals:
                    y[c] = float(lab)
        y_case = pd.Series(y, dtype=float)
        order = sorted(y_case.index)
        yt = y_case.reindex(order).to_numpy()
        dm = ix.exam_case_id.isin(y_case.index).to_numpy()
        vc = raw_dist(field, dev_ids)
        unmapped = sorted(set(vc.index) - set(sum(bm.values(), [])))

        if len(set(yt)) < 2 or len(yt) < 30:
            rows.append({"field": field, "kind": "binary", "n": len(yt),
                         "status": "NOT_TESTABLE",
                         "why": "single class or n<30"})
            continue
        pr = oof_probs(feats, dm, ix, y_case, folds, order)
        ok = ~np.isnan(pr)
        yp = (pr[ok] >= 0.5).astype(float)
        ytk = yt[ok]
        maj = float(np.bincount(ytk.astype(int)).argmax())
        acc = float((yp == ytk).mean())
        base = float((ytk == maj).mean())
        lo, hi, passes = boot(ytk, yp, maj, np.random.default_rng(SEED))
        two = len(set(ytk)) == 2
        r = {
            "field": field, "kind": "binary", "n": int(len(ytk)),
            "n_labelled_of_186": int(len(yt)),
            "n_dropped_unmapped_or_missing": int(186 - len(yt)),
            "accuracy": round(acc, 4), "baseline_constant": round(base, 4),
            "delta": round(acc - base, 4), "ci95_lo": lo, "ci95_hi": hi,
            "passes": passes,
            "balanced_accuracy": round(float(balanced_accuracy_score(
                ytk.astype(int), yp.astype(int))), 4) if two else None,
            "macro_f1": round(float(f1_score(ytk, yp, average="macro",
                                             zero_division=0)), 4),
            "auroc": round(float(roc_auc_score(ytk, pr[ok])), 4) if two else None,
            "abnormal_share": round(float(ytk.mean()), 4),
            "predicted_abnormal_share": round(float(yp.mean()), 4),
            "collapsed_to_one_class": bool(len(set(yp)) == 1),
            "raw_classes_dev": int(len(vc)),
            "unmapped_raw_values": ",".join(unmapped) if unmapped else "",
        }
        rows.append(r)
        detail[field] = dict(r, binary_scheme={str(k): v for k, v in bm.items()},
                             raw_distribution_dev={str(k): int(v)
                                                   for k, v in vc.items()})
        print(f"{field:28s} bin n={r['n']:3d} acc={acc:.4f} base={base:.4f} "
              f"d={acc-base:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"{'PASS' if passes else 'fail'} collapsed={r['collapsed_to_one_class']}")

    # ---------- multiclass-only fields ---------------------------------------
    for field in MULTICLASS_ONLY:
        v = m[field].astype(str).str.strip().reindex(dev_ids)
        present = v[~v.isin(MISSING)]
        # drop bracketed/unconfirmable text rather than assign it a class
        clean = present[~present.str.startswith("[")]
        cats = sorted(clean.unique())
        code = {c: i for i, c in enumerate(cats)}
        y_case = clean.map(code).astype(float)
        order = sorted(y_case.index)
        yt = y_case.reindex(order).to_numpy()
        dm = ix.exam_case_id.isin(y_case.index).to_numpy()
        pr = oof_probs(feats, dm, ix, y_case, folds, order, multiclass=True)
        ok = ~np.isnan(pr)
        yp, ytk = pr[ok], yt[ok]
        maj = float(np.bincount(ytk.astype(int)).argmax())
        acc = float((yp == ytk).mean())
        base = float((ytk == maj).mean())
        lo, hi, passes = boot(ytk, yp, maj, np.random.default_rng(SEED))
        r = {"field": field, "kind": f"multiclass_{len(cats)}",
             "n": int(len(ytk)), "n_labelled_of_186": int(len(y_case)),
             "n_dropped_unmapped_or_missing": int(186 - len(y_case)),
             "accuracy": round(acc, 4), "baseline_constant": round(base, 4),
             "delta": round(acc - base, 4), "ci95_lo": lo, "ci95_hi": hi,
             "passes": passes,
             "balanced_accuracy": round(float(balanced_accuracy_score(
                 ytk.astype(int), yp.astype(int))), 4),
             "macro_f1": round(float(f1_score(ytk, yp, average="macro",
                                              zero_division=0)), 4),
             "auroc": None, "raw_classes_dev": int(len(cats)),
             "collapsed_to_one_class": bool(len(set(yp)) == 1),
             "unmapped_raw_values": ",".join(
                 sorted(set(present) - set(clean)))}
        rows.append(r)
        detail[field] = dict(r, classes=cats,
                             raw_distribution_dev={str(k): int(x) for k, x
                                                   in present.value_counts().items()})
        print(f"{field:28s} mc{len(cats)} n={r['n']:3d} acc={acc:.4f} "
              f"base={base:.4f} d={acc-base:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"{'PASS' if passes else 'fail'}")

    # ---------- numeric fields ----------------------------------------------
    for field in NUMERIC:
        vals = m[field].reindex(dev_ids).map(to_float)
        present = vals.dropna()
        if len(present) < 30:
            rows.append({"field": field, "kind": "numeric", "n": int(len(present)),
                         "status": "NOT_TESTABLE",
                         "why": f"only {len(present)} labelled of 186"})
            detail[field] = {"n_labelled": int(len(present)),
                             "status": "NOT_TESTABLE"}
            print(f"{field:28s} numeric n={len(present)} NOT_TESTABLE")
            continue
        order = sorted(present.index)
        yt = present.reindex(order).to_numpy()
        dm = ix.exam_case_id.isin(present.index).to_numpy()
        cases = ix.exam_case_id[dm].to_numpy()
        pos = {c: i for i, c in enumerate(order)}
        pred = np.full(len(order), np.nan)
        fs = pd.Series([folds.get(c, np.nan) for c in order], index=order)
        from sklearn.linear_model import Ridge
        for fo in sorted(fs.dropna().unique()):
            te = set(fs.index[fs == fo])
            tr = set(order) - te
            trm = dm & ix.exam_case_id.isin(tr).to_numpy()
            tem = dm & ix.exam_case_id.isin(te).to_numpy()
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            ytr = present.reindex(ix.exam_case_id[trm]).to_numpy()
            ps = []
            for p in POOLINGS:
                dim = max(2, min(DIM_FIXED, trm.sum() - 1, feats[p].shape[1]))
                pipe = make_pipeline(StandardScaler(),
                                     PCA(n_components=dim, random_state=SEED),
                                     Ridge(alpha=10.0))
                pipe.fit(feats[p][trm], ytr)
                ps.append(pipe.predict(feats[p][tem]))
            s = pd.Series(np.mean(ps, axis=0),
                          index=ix.exam_case_id[tem].to_numpy()
                          ).groupby(level=0).mean()
            for c, v in s.items():
                if c in pos:
                    pred[pos[c]] = v
        ok = ~np.isnan(pred)
        mae = float(np.abs(pred[ok] - yt[ok]).mean())
        med = float(np.median(yt[ok]))
        mae_base = float(np.abs(med - yt[ok]).mean())
        uq = np.unique(yt[ok])
        gaps = np.diff(uq)
        r = {"field": field, "kind": "numeric", "n": int(ok.sum()),
             "n_labelled_of_186": int(len(present)),
             "n_dropped_unmapped_or_missing": int(186 - len(present)),
             "mae": round(mae, 4), "mae_median_baseline": round(mae_base, 4),
             "mae_improvement": round(mae_base - mae, 4),
             "beats_median_baseline": bool(mae < mae_base),
             "n_distinct_label_values": int(len(uq)),
             "median_gap_between_adjacent_labels": round(float(np.median(gaps)), 4)
                                                   if len(gaps) else None,
             "mae_exceeds_median_label_gap": bool(
                 len(gaps) and mae > float(np.median(gaps))),
             "passes": False}
        rows.append(r)
        detail[field] = dict(r, label_values=[float(x) for x in uq[:40]])
        print(f"{field:28s} numeric n={r['n']:3d} MAE={mae:.3f} "
              f"medbase={mae_base:.3f} gain={mae_base-mae:+.3f} "
              f"distinct={len(uq)} medgap={r['median_gap_between_adjacent_labels']}")

    # ---------- derived --------------------------------------------------
    for field in DERIVED:
        vals = m[field].reindex(dev_ids).map(to_float).dropna()
        rows.append({"field": field, "kind": "derived_identity",
                     "n": int(len(vals)), "status": "MUST_NOT_BE_MODELLED",
                     "why": "output_input_ratio is efferent/afferent -- an "
                            "arithmetic identity of two other fields, not an "
                            "independent observation"})
        detail[field] = {"n_labelled": int(len(vals)),
                         "status": "MUST_NOT_BE_MODELLED",
                         "why": "arithmetic identity of efferent/afferent"}
        print(f"{field:28s} DERIVED IDENTITY -- not modelled")

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False, encoding="utf-8-sig")
    n_pass = int(df.get("passes", pd.Series(dtype=bool)).fillna(False).sum())
    out = {
        "what_this_is": "all 21 fields under ONE fixed configuration, "
                        "DEVELOPMENT OOF only",
        "why_it_differs_from_full_field_fix_20260916": (
            "that run selected source (instance vs dino) and model (trees vs "
            "linear) PER FOLD by validation score, which inflates every delta "
            "and makes fields mutually incomparable. This run fixes one "
            "configuration for all fields."),
        "config": {"encoder": "frozen DINOv2 ViT-B/14, sha256 0b8b82f8...",
                   "poolings": POOLINGS,
                   "estimator": f"StandardScaler->PCA({DIM_FIXED})->"
                                f"LogisticRegression(C={C_FIXED}); Ridge(alpha=10) "
                                f"for numeric",
                   "threshold": 0.5, "n_boot": N_BOOT, "seed": SEED,
                   "selection": "none"},
        "locked_cases_seen": 0,
        "n_fields": len(rows),
        "n_passing_on_development": n_pass,
        "fields": detail,
        "limitations": [
            "DEVELOPMENT OOF. These are NOT product capability. The only "
            "held-out numbers in this project are in group 3, where 0 of 5 "
            "fields pass under the stricter baseline.",
            "a 'PASS' here means only: OOF accuracy beat a single constant "
            "answer with a bootstrap CI lower bound above zero, on 186 internal "
            "exams from the same three same-source archives",
            "the binary cuts are recovered from the delivered pipeline, not "
            "clinically sanctioned; several merge three ordinal levels into one "
            "'abnormal' class and that merge is what makes the task easier",
            "numeric fields are scored in RAW LABEL UNITS. No micron claim is "
            "made or possible: there is no calibration (see group 5).",
            "no field here has any human pixel-level ground truth, and no field "
            "has a second reader, so label noise is unmeasured for all 21.",
        ],
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nfields:", len(rows), "| passing on development:", n_pass)
    print("wrote", a.out, "and", a.out_csv)


if __name__ == "__main__":
    main()
