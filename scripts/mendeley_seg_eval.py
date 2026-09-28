#!/usr/bin/env python
"""G3 sign test and G4 downstream LOAO for the segmenter arms (PREREG).

G3: Spearman rho(case-mean frac_low_circ, malformation level) per arm on all
    labelled cases (233 = dev + consumed locked-47, training data now), and a
    paired case bootstrap of rho(arm) - rho(S0).
G4: LOAO over the 3 archives. Reference A0 = the five DINOv2 poolings (the
    rag_heads_v2 recipe). Treatment = A0 + a sixth "geometry" route: the arm's
    case-level geometry block -> StandardScaler -> LogReg C=0.03, averaged with
    the five pooling routes at equal weight. Paired bootstrap BA(treat) - BA(A0).
    Fields: malformation_ratio, crossing_ratio only.

  python scripts/mendeley_seg_eval.py --arms S0_c005 S1_c005 S1n_c005
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402
from segmiss_phase3 import MALF_LEVEL  # noqa: E402

GEOM = "E:/nailfold_tmp/seg_geom"
OUT = os.path.join(ROOT, "artifacts/experiments/mendeley_sam_seg_20260927")
FIELDS = ["malformation_ratio", "crossing_ratio"]
N_BOOT = 2000


def case_geom(arm):
    img = pd.read_csv(os.path.join(GEOM, arm, "image_geometry.csv"), dtype={"exam_case_id": str})
    num = img.select_dtypes(include=[np.number]).columns.drop(["img_w", "img_h"], errors="ignore")
    return img.groupby("exam_case_id")[list(num)].mean()


def sign_test(arms, man):
    lvl = man.malformation_ratio.map(MALF_LEVEL).dropna()
    G = {a: case_geom(a)["frac_low_circ"].reindex(lvl.index) for a in arms}
    ok = np.logical_and.reduce([G[a].notna().to_numpy() for a in arms])
    y = lvl.to_numpy()[ok]
    out, rng = {}, np.random.default_rng(R.SEED)
    for a in arms:
        v = G[a].to_numpy()[ok]
        rho, p = spearmanr(v, y)
        out[a] = dict(rho=round(float(rho), 4), p=float("%.3g" % p), n=int(ok.sum()),
                      grad={int(k): round(float(m), 4) for k, m in
                            pd.Series(v).groupby(y).mean().items()})
    ref = arms[0]
    for a in arms[1:]:
        d = []
        va, vr = G[a].to_numpy()[ok], G[ref].to_numpy()[ok]
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            d.append(spearmanr(va[i], y[i])[0] - spearmanr(vr[i], y[i])[0])
        out[a]["delta_vs_%s" % ref] = [round(float(np.percentile(d, q)), 4) for q in (2.5, 50, 97.5)]
        out[a]["G3_pass"] = bool((out[a]["rho"] > 0 and out[ref]["rho"] <= 0)
                                 or np.percentile(d, 2.5) > 0)
    return out


def loao_geom(y, arch, ixa, Fa, geom):
    """Returns per-case (p_A0, p_treat) for binary y, LOAO over archives."""
    img_case = ixa.exam_case_id.to_numpy()
    y_img = y.reindex(img_case).to_numpy()
    lab = ~np.isnan(y_img)
    rows = []
    for a in sorted(arch.reindex(y.index).unique()):
        test = set(y.index[arch.reindex(y.index) == a]); train = set(y.index) - test
        tr, te = np.isin(img_case, list(train)) & lab, np.isin(img_case, list(test))
        P = C.fit_predict(Fa, img_case, np.where(lab, y_img, -1).astype(int), tr, te, [0, 1])
        gtr = geom.reindex(sorted(train)).dropna(how="all")
        gtr = gtr.loc[:, gtr.notna().mean() > 0.9]
        cols = gtr.columns
        med = gtr.median()
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=R.C_FIXED, max_iter=4000))
        pipe.fit(gtr.fillna(med), y.reindex(gtr.index))
        gte = geom.reindex(P.index)[cols].fillna(med)
        pg = pipe.predict_proba(gte)[:, 1]
        pa = P[1].to_numpy()
        pt = (5 * pa + pg) / 6
        for c, x1, x2 in zip(P.index, pa, pt):
            rows.append(dict(exam_case_id=c, archive=a, y=int(y[c]), p_a0=x1, p_treat=x2))
    return pd.DataFrame(rows).set_index("exam_case_id")


def paired(df):
    y = df.y.to_numpy(); ha = (df.p_a0 > 0.5).astype(int).to_numpy()
    ht = (df.p_treat > 0.5).astype(int).to_numpy()
    rng = np.random.default_rng(R.SEED); d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(balanced_accuracy_score(y[i], ht[i]) - balanced_accuracy_score(y[i], ha[i]))
    return dict(n=len(y), ba_a0=round(float(balanced_accuracy_score(y, ha)), 4),
                ba_treat=round(float(balanced_accuracy_score(y, ht)), 4),
                delta_ci=[round(float(np.percentile(d, q)), 4) for q in (2.5, 50, 97.5)],
                G4_pass=bool(np.percentile(d, 2.5) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    man, ixa, Fa, src, fold = C.load()
    arch = pd.Series(man.index.str.split("/").str[0], index=man.index)
    T = C.targets(man)
    geo_arms = [x for x in a.arms if "frac_low_circ" in case_geom(x).columns]
    res = dict(sign_test=sign_test(geo_arms, man) if geo_arms else {}, loao={})
    for arm in a.arms:
        g = case_geom(arm)
        res["loao"][arm] = {f: paired(loao_geom(T[f].astype(float), arch, ixa, Fa, g))
                            for f in FIELDS}
    res["note"] = ("233 labelled cases incl. consumed locked-47 (training data, not a test set); "
                   "segmenters were trained on vascular-dataset crops, some of which come from "
                   "our dev images (vessel polygons only, no clinical labels) -- same for every arm")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "eval%s.json" % a.tag)
    json.dump(res, open(p, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
