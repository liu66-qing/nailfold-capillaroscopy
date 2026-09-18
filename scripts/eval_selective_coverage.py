#!/usr/bin/env python
"""Selective prediction: does abstaining on low-confidence cases buy real accuracy?

Every ruler so far forced an answer on all 186 cases. eval_geom_imagelevel.py
computes the OOF probability and then THROWS IT AWAY (it keeps only the
thresholded label), so abstention has never been measured. This script keeps the
probability and asks the delivery question instead of the modelling question:

    if the product is allowed to answer "image quality insufficient, please
    re-measure" on the least confident cases, does the accuracy on the cases it
    DOES answer clear the bar?

THE TRAP THIS SCRIPT IS BUILT TO CATCH
--------------------------------------
Confidence and class prior are confounded. A model whose confident cases are
simply the majority-class cases will show rising accuracy under coverage while
delivering nothing: the majority baseline rises just as fast. So the baseline is
RECOMPUTED INSIDE EACH COVERAGE SUBSET, never taken from the full set. A field
only counts if delta -- accuracy minus the one fixed majority answer measured on
the SAME retained cases -- has a bootstrap CI lower bound above 0.

Reporting both curves is the whole point. If accuracy climbs and delta stays
flat, that is the confound, and this route is refuted like the previous five.

Ruler, unchanged from the delivered configuration: image-level fit, folds by
development_fold only, case as the evaluation unit, C=0.03, 2000 case-level
bootstrap resamples, seed 20260918, prespecified arm 'concat'.
locked-47: never loaded, asserted, reported as locked_cases_seen: 0.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

N_BOOT = 2000
SEED = 20260918
C_FIXED = 0.03
PRESPECIFIED_ARM = "concat"
# prespecified grid, fixed before looking at any result
COVERAGES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]

FIELDS = {
    "clarity":            {"normal": ["清晰"]},
    "capillary_count":    {"normal": [">=7"]},
    "crossing_ratio":     {"normal": ["<=30%", "[<30%]", "10--30%"]},
    "malformation_ratio": {"normal": ["<=10%", "[<10%]"]},
    "papilla":            {"normal": ["波纹状", "浅波纹状"], "abnormal": ["平坦"]},
    "blood_color":        {"normal": ["红润", "淡红"]},
    "exudation":          {"normal": ["无", "０", "0"]},
    "microthrombus":      {"normal": ["无"]},
    "rbc_aggregation":    {"normal": ["无", "轻度"], "abnormal": ["中度", "重度"]},
    "afferent_diameter":  {"numeric_q": 0.75},
    "efferent_diameter":  {"numeric_q": 0.75},
    "apex_diameter":      {"numeric_q": 0.75},
    "loop_length":        {"numeric_q": 0.75},
}

ABNORMAL_STR = ["不清", "模糊", "30--60%", "60--80%", ">80%", "10--30%", ">60%",
                "5--6", "3--4", "1--2", "<1", "有", "少量", "中量", "大量",
                "轻度", "中度", "重度", "淡", "暗红", "紫红", "淡紫", "紫",
                "１", "２", "1", "2", "3", "＋", "+", "++", "+++"]


def binarise(s, cfg):
    """1 = abnormal. Unparseable text stays NaN, never becomes a class."""
    if "numeric_q" in cfg:
        v = pd.to_numeric(s, errors="coerce")
        if v.notna().sum() < 40:
            return pd.Series(np.nan, index=s.index, dtype=float)
        thr = float(v.quantile(cfg["numeric_q"]))
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out[v.notna()] = (v[v.notna()] > thr).astype(float)
        return out
    v = s.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[v.isin(cfg["normal"])] = 0.0
    out[v.isin(cfg.get("abnormal", ABNORMAL_STR))] = 1.0
    return out


def oof_image_level(Xi, img_case, y_case, fold_case):
    """Fit on IMAGE rows, predict per image, average back to the case."""
    yi = y_case.reindex(img_case).to_numpy()
    fi = fold_case.reindex(img_case).to_numpy()
    p = pd.Series(np.nan, index=range(len(Xi)), dtype=float)
    for k in sorted(pd.unique(fi[~pd.isna(fi)])):
        tr = (fi != k) & ~pd.isna(yi)
        te = fi == k
        if tr.sum() < 30 or len(np.unique(yi[tr])) < 2:
            continue
        mdl = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C_FIXED, max_iter=5000, solver="liblinear"),
        )
        mdl.fit(Xi[tr], yi[tr].astype(int))
        p.loc[np.where(te)[0]] = mdl.predict_proba(Xi[te])[:, 1]
    return pd.Series(p.to_numpy(), index=img_case).groupby(level=0).mean()


def score_subset(yt, yp, rng):
    """Accuracy minus ONE fixed majority answer, baseline redrawn per resample.

    yt/yp are already restricted to the RETAINED cases, so the majority answer
    is the majority of the retained cases. That is the whole safeguard: a model
    that only retains majority-class cases gets no credit here.
    """
    n = len(yt)
    if n < 30:
        return None
    acc = float((yp == yt).mean())
    maj = float(np.bincount(yt.astype(int)).argmax())
    base = float((yt == maj).mean())
    d = np.empty(N_BOOT)
    a = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        ys, ps = yt[i], yp[i]
        m = float(np.bincount(ys.astype(int)).argmax())
        a[b] = (ps == ys).mean()
        d[b] = a[b] - (ys == m).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    alo, ahi = np.percentile(a, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4),
            "acc_ci95": [round(float(alo), 4), round(float(ahi), 4)],
            "baseline": round(base, 4), "delta": round(acc - base, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "usable": bool(lo > 0),
            "abnormal_share": round(float(yt.mean()), 4)}


def coverage_curve(y, prob, rng_seed):
    """Sweep the prespecified coverage grid by |p-0.5| confidence."""
    lab = (prob > 0.5).astype(float)
    ok = prob.notna() & y.notna()
    yv = y[ok].to_numpy()
    pv = prob[ok].to_numpy()
    lv = lab[ok].to_numpy()
    conf = np.abs(pv - 0.5)
    order = np.argsort(-conf)          # most confident first
    n_all = len(yv)
    rows = []
    for cov in COVERAGES:
        k = int(round(cov * n_all))
        if k < 30:
            rows.append({"coverage": cov, "n": k, "status": "N_TOO_SMALL"})
            continue
        keep = order[:k]
        if len(np.unique(yv[keep])) < 2:
            rows.append({"coverage": cov, "n": k, "status": "SINGLE_CLASS_RETAINED"})
            continue
        s = score_subset(yv[keep], lv[keep], np.random.default_rng(rng_seed))
        if s is None:
            rows.append({"coverage": cov, "n": k, "status": "N_TOO_SMALL"})
            continue
        s["coverage"] = cov
        s["conf_threshold"] = round(float(conf[order[k - 1]]), 4)
        rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom",
                    default="artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="selective")
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
    print(f"aligned images: {len(mrg)} of geom {len(Gi)} / dino {len(idx)}",
          flush=True)

    gcols = [c for c in Gi.columns
             if c not in ("exam_case_id", "image_path", "fold")
             and pd.api.types.is_numeric_dtype(Gi[c])]
    dcols = [c for c in range(F.shape[1])]
    XG = mrg[gcols].to_numpy(dtype=np.float32)
    XD = mrg[dcols].to_numpy(dtype=np.float32)
    XC = np.hstack([XG, XD])
    img_case = pd.Index(mrg["exam_case_id"])

    out = {"config": {"tag": args.tag, "C": C_FIXED, "n_boot": N_BOOT,
                      "seed": SEED, "level": "image",
                      "coverages": COVERAGES,
                      "confidence_rule": "abs(prob - 0.5), most confident kept",
                      "geom_dims": len(gcols), "dino_dims": len(dcols),
                      "n_images": int(len(mrg)),
                      "n_cases": int(img_case.nunique()),
                      "locked_cases_seen": 0,
                      "prespecified_arm": PRESPECIFIED_ARM,
                      "geom_source": args.geom},
           "fields": {}}

    for field, cfg in FIELDS.items():
        if field not in dev.columns:
            out["fields"][field] = {"status": "NO_LABEL_COLUMN"}
            continue
        y = binarise(dev[field], cfg).dropna()
        y = y[y.index.isin(img_case)]
        if len(y) < 40 or y.nunique() < 2:
            out["fields"][field] = {"status": "INSUFFICIENT_LABELS",
                                    "labelled": int(len(y))}
            continue
        prob = oof_image_level(XC, img_case, y, fold_case)
        curve = coverage_curve(y.reindex(prob.index), prob, SEED)
        full = next((r for r in curve if r.get("coverage") == 1.00
                     and "delta" in r), None)
        best = None
        for r in curve:
            if "delta" in r and r["usable"]:
                best = r
                break
        out["fields"][field] = {
            "labelled_cases": int(len(y)),
            "full_coverage": full,
            "first_usable_coverage": best,
            "curve": curve,
        }
        line = " | ".join(
            f"cov{r['coverage']:.2f} n={r['n']} acc={r['accuracy']:.3f} "
            f"base={r['baseline']:.3f} d={r['delta']:+.4f}"
            f"{'*' if r['usable'] else ''}"
            for r in curve if "delta" in r)
        print(f"{field:20s} {line}", flush=True)

    out["limitations"] = [
        "development set only (n<=186); NOT product capability, no locked-47 result",
        "the coverage grid was prespecified; the first usable coverage is still "
        "a selected point and needs confirmation on held-out data before shipping",
        "abstention shrinks n, so CIs widen as coverage falls; a 'usable' flag at "
        "cov 0.50 rests on ~90 cases",
        "measurement fields binarised at the upper quartile, NOT a micron value",
        "geometry is pixel-scale; calibration_factors.json deliberately unused",
        "baseline is recomputed on the RETAINED cases at every coverage; a rising "
        "accuracy curve with a flat delta curve means the confidence is only "
        "tracking the class prior, which would refute this route",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
