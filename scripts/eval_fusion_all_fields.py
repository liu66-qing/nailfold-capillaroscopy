#!/usr/bin/env python
"""Does instance geometry fused with DINOv2 raise ANY field?

Two things have never been tried, and this script does both:

1. FUSION. The 209-dim instance geometry and the DINOv2 image features have
   only ever been evaluated separately. They carry different information:
   DINOv2 sees texture and colour (which is why it carries clarity and
   blood_color), geometry sees vessel shape and arrangement. A field that needs
   both -- "are the loops dilated AND is the background hazy" -- is invisible to
   either alone.

2. BREADTH. Geometry was only ever tested on the 3 structure fields. Fields
   like papilla (skin ridge shape, missed at +0.049) and the diameter fields
   are geometric quantities that were only ever attacked with whole-image
   features.

Ruler is unchanged and copied, not adapted: accuracy minus ONE fixed majority
answer, baseline recomputed inside all 2000 case-level bootstrap resamples,
verdict by CI lower bound > 0, folds from development_fold only, locked-47
never loaded.

Arms are prespecified: geom / dino / concat / prob-average. The reported
deliverable is the PRESPECIFIED arm ('concat'); the others are printed as
context. Reading the best arm after seeing all four is the selection bias this
project already refuted, so 'best_arm' is labelled non-deliverable.
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

# Binary cut per field. Placed at the clinically normal band, not tuned.
# 'normal' lists the raw report strings that mean "no abnormality".
FIELDS = {
    "clarity":            {"normal": ["清晰"]},
    "capillary_count":    {"normal": [">=7"]},
    "crossing_ratio":     {"normal": ["<=30%", "[<30%]", "10--30%"]},
    "malformation_ratio": {"normal": ["<=10%", "[<10%]"]},
    # papilla has no 清晰/不清 wording: the ridge is flat, rippled, or shallowly
    # rippled. 平坦 (flat) is the abnormal end -- loss of the normal ripple.
    "papilla":            {"normal": ["波纹状", "浅波纹状"],
                           "abnormal": ["平坦"]},
    "blood_color":        {"normal": ["红润", "淡红"]},
    "exudation":          {"normal": ["无", "０", "0"]},
    "hemorrhage":         {"normal": ["无", "０", "0"]},
    "sweat_duct":         {"normal": ["清晰"]},
    "afferent_diameter":  {"numeric_q": 0.75},
    "efferent_diameter":  {"numeric_q": 0.75},
    "apex_diameter":      {"numeric_q": 0.75},
    "loop_length":        {"numeric_q": 0.75},
}

ABNORMAL_STR = ["不清", "模糊", "30--60%", "60--80%", ">80%", "10--30%", ">60%",
                "5--6", "3--4", "1--2", "<1", "有", "少量", "中量", "大量",
                "轻度", "中度", "重度", "淡", "暗红", "紫红", "淡紫", "紫",
                "１", "２", "1", "2", "3", "＋", "+", "++", "+++"]


def binarise(s, cfg, folds=None):
    """1 = abnormal. Unparseable text becomes NaN, never silently a class.

    For the measurement fields there is no normal/abnormal wording, so the cut
    is the upper quartile of the *training* distribution -- 'is this vessel
    among the dilated quarter'. That is a real question with a real majority
    baseline, unlike predicting a micron value the labels do not support.
    """
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
    # a field may name its own abnormal values when they are not generic wording
    out[v.isin(cfg.get("abnormal", ABNORMAL_STR))] = 1.0
    return out


def oof_prob(X, y, folds):
    """Out-of-fold case-level probability. One fixed hyperparameter, no search."""
    p = pd.Series(np.nan, index=X.index, dtype=float)
    for k in sorted(pd.unique(folds)):
        tr, te = folds != k, folds == k
        if y[tr].nunique() < 2 or tr.sum() < 20:
            continue
        mdl = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C_FIXED, max_iter=5000, solver="liblinear"),
        )
        mdl.fit(X[tr], y[tr])
        p.loc[X.index[te]] = mdl.predict_proba(X[te])[:, 1]
    return p


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
    ap.add_argument("--geom", default="artifacts/features/seg_instance_v1/case_geometry.csv")
    ap.add_argument("--dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", default="artifacts/features/seg_instance_v1/fusion_all_fields.json")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])
    dev = man[man.evaluation_role == "development"].set_index("exam_case_id")

    G = pd.read_csv(args.geom, index_col=0)
    G.index = G.index.astype(str)
    assert not (set(G.index) & locked), "locked case in geometry"
    folds = G["fold"].astype(int)
    gcols = [c for c in G.columns if c != "fold"]

    # DINOv2 is per image; average to case so both arms share one row per case
    F = np.load(args.dino).astype(np.float32)
    idx = pd.read_csv(args.index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    assert len(idx) == len(F), f"index {len(idx)} != features {len(F)}"
    D = (pd.DataFrame(F, index=idx["exam_case_id"])
           .groupby(level=0).mean().reindex(G.index))
    D.columns = [f"d{i}" for i in range(D.shape[1])]
    assert not (set(D.index) & locked), "locked case in dinov2 rows"

    GEO = G[gcols]
    out = {"config": {"C": C_FIXED, "n_boot": N_BOOT, "seed": SEED,
                      "geom_dims": len(gcols), "dino_dims": D.shape[1],
                      "n_cases": int(G.shape[0]), "locked_cases_seen": 0,
                      "prespecified_arm": PRESPECIFIED_ARM,
                      "geom_source": args.geom},
           "fields": {}}

    for field, cfg in FIELDS.items():
        if field not in dev.columns:
            out["fields"][field] = {"status": "NO_LABEL_COLUMN"}
            continue
        y = binarise(dev[field], cfg).reindex(G.index)
        keep = y.notna()
        if keep.sum() < 40 or y[keep].nunique() < 2:
            out["fields"][field] = {"status": "INSUFFICIENT_LABELS",
                                    "labelled": int(keep.sum())}
            continue
        yk, fk = y[keep], folds[keep]
        share = float(min(yk.mean(), 1 - yk.mean()))

        pg = oof_prob(GEO.loc[keep], yk, fk)
        pd_ = oof_prob(D.loc[keep], yk, fk)
        pc = oof_prob(pd.concat([GEO.loc[keep], D.loc[keep]], axis=1), yk, fk)
        pa = pd.concat([pg, pd_], axis=1).mean(axis=1)

        arms = {}
        for nm, p in [("geom", pg), ("dino", pd_), ("concat", pc), ("prob_avg", pa)]:
            arms[nm] = score(yk, p, np.random.default_rng(SEED))

        pre = arms[PRESPECIFIED_ARM]
        best = max((a for a in arms.items() if a[1]), key=lambda kv: kv[1]["delta"])
        out["fields"][field] = {
            "labelled_cases": int(keep.sum()),
            "minority_share": round(share, 4),
            "deliverable": pre,
            "arms": arms,
            "best_arm_posthoc": {"arm": best[0], "delta": best[1]["delta"],
                                 "note": "NOT deliverable: chosen after seeing "
                                         "all arms, which is the refuted "
                                         "selection bias"},
        }
        print(f"{field:20s} pre={pre['delta'] if pre else None} "
              f"{pre['ci95'] if pre else ''} "
              f"| geom {arms['geom']['delta'] if arms['geom'] else None}"
              f" dino {arms['dino']['delta'] if arms['dino'] else None}"
              f" avg {arms['prob_avg']['delta'] if arms['prob_avg'] else None}",
              flush=True)

    out["limitations"] = [
        "development set only (n<=186); NOT product capability, no locked-47 result",
        "measurement fields are binarised at the upper quartile ('is it among "
        "the dilated quarter'), NOT a micron value; no absolute scale is claimed",
        "geometry is pixel-scale; calibration_factors.json deliberately unused",
        "only the prespecified 'concat' arm is a deliverable number",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
