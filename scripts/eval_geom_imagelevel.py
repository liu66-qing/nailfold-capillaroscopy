#!/usr/bin/env python
"""Instance geometry inside the DELIVERED framework: image-level + fixed ensemble.

Every geometry test so far collapsed each case to one row before fitting. The
delivered configuration does the opposite: it fits on image rows (case label
broadcast, folds split by case, predictions averaged back to the case), which
gives ~9x the fitting rows while the evaluation unit stays the case. That is
worth +0.06 on clarity alone (0.270 case-level vs 0.330 image-level), so
geometry has never been tested under the setup that actually ships.

This script therefore runs geometry, DINOv2 and their concatenation at the
IMAGE level, with the same fixed equal-weight ensembling and the same ruler:
accuracy minus ONE fixed majority answer, baseline recomputed inside all 2000
case-level bootstrap resamples, verdict by CI lower bound > 0, folds from
development_fold only, locked-47 never loaded.

It also accepts --geom pointing at either confidence threshold, so the
conf=0.05 extraction (4x more instances per image) can be tested without
changing any other knob.
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

FIELDS = {
    "clarity":            {"normal": ["清晰"]},
    "capillary_count":    {"normal": [">=7"]},
    "crossing_ratio":     {"normal": ["<=30%", "[<30%]", "10--30%"]},
    "malformation_ratio": {"normal": ["<=10%", "[<10%]"]},
    "papilla":            {"normal": ["波纹状", "浅波纹状"], "abnormal": ["平坦"]},
    "blood_color":        {"normal": ["红润", "淡红"]},
    "exudation":          {"normal": ["无", "０", "0"]},
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
    """Fit on IMAGE rows, predict per image, average back to the case.

    The case is still the evaluation unit and folds are still split by case, so
    no image of a test case is ever in a training fold. What changes is that the
    model sees ~9 rows per case instead of 1.
    """
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


def score(y, prob, rng):
    """Accuracy minus ONE fixed majority answer; baseline redrawn per resample."""
    lab = prob.where(prob.isna(), (prob > 0.5).astype(float))
    ok = lab.notna() & y.notna()
    yt, yp = y[ok].to_numpy(), lab[ok].to_numpy()
    n = len(yt)
    if n < 30:
        return None
    acc = float((yp == yt).mean())
    maj = float(np.bincount(yt.astype(int)).argmax())
    base = float((yt == maj).mean())
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        ys, ps = yt[i], yp[i]
        m = float(np.bincount(ys.astype(int)).argmax())
        d[b] = (ps == ys).mean() - (ys == m).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4), "baseline": round(base, 4),
            "delta": round(acc - base, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "usable": bool(lo > 0), "abnormal_share": round(float(yt.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom",
                    default="artifacts/features/seg_instance_v1/image_geometry.csv")
    ap.add_argument("--dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="conf025")
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

    # align geometry rows to dinov2 rows on image_path so both arms see the
    # same images in the same order
    Gi["image_path"] = Gi["image_path"].astype(str)
    mrg = Gi.merge(dmap, on=["exam_case_id", "image_path"], how="inner",
                   suffixes=("", "_d"))
    print(f"aligned images: {len(mrg)} of geom {len(Gi)} / dino {len(idx)}")

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
        arms = {}
        for nm, X in [("geom", XG), ("dino", XD), ("concat", XC)]:
            p = oof_image_level(X, img_case, y, fold_case)
            arms[nm] = score(y.reindex(p.index), p, np.random.default_rng(SEED))
        pre = arms[PRESPECIFIED_ARM]
        out["fields"][field] = {
            "labelled_cases": int(len(y)),
            "deliverable": pre,
            "arms": arms,
        }
        print(f"{field:20s} concat={pre['delta'] if pre else None} "
              f"{pre['ci95'] if pre else ''} | "
              f"geom {arms['geom']['delta'] if arms['geom'] else None} "
              f"dino {arms['dino']['delta'] if arms['dino'] else None}",
              flush=True)

    out["limitations"] = [
        "development set only (n<=186); NOT product capability, no locked-47 result",
        "measurement fields binarised at the upper quartile, NOT a micron value",
        "geometry is pixel-scale; calibration_factors.json deliberately unused",
        "only the prespecified 'concat' arm is a deliverable number",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
