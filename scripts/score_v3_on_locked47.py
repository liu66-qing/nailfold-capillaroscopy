"""v3 candidates vs rag_heads_v1 on the contaminated locked-47 test set.

v1 = A0: the five DINOv2 pooling routes, fitted on dev 186 (the rag_heads_v1 recipe).
v3(arm) = A0 plus a sixth route: the segmenter arm's case-mean geometry, then
median-fill, StandardScaler and LogReg C=0.03, fitted on dev 186. This is the
same equal-weight averaging as the G4 gate:
    p = (5 * p_A0 + p_geom) / 6

Both models are fitted on dev 186 only and scored once on locked-47. The
contrast is paired: same cases, same A0 routes. Bootstrap BA(v3) - BA(v1),
plus a null in which the geometry rows are permuted across dev and test cases.
That null gives the gain from averaging alone, because Stage-1 null controls
showed that route averaging by itself produces positive deltas.

Contamination:
  * locked-47 was read 7 times before, and model selection was done on it in 3 runs;
  * the segmenters' vessel crops may include frames of locked images (vessel
    polygons only, no clinical labels; the same for every arm).
Descriptive only; not the preregistered prospective evaluation.

  python scripts/score_v3_on_locked47.py --arms S0_c005 UNET [S1_c005 S1n_c005 ANFCDET]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402
from mendeley_seg_eval import case_geom  # noqa: E402

OUT = os.path.join(ROOT, "artifacts/experiments/test_locked47_v3")
# the seven rag_heads fields plus crossing (the segmenter's own target)
FIELDS = ["clarity", "subpapillary_venous_plexus", "exudation", "loop_length",
          "blood_color", "microthrombus", "malformation_ratio", "crossing_ratio"]
PRIMARY = ["malformation_ratio", "crossing_ratio"]
N_BOOT, N_NULL = 2000, 20


def ba(y, p):
    return float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))


def geom_route(geom, train, test, y):
    gtr = geom.reindex(sorted(train)).dropna(how="all")
    gtr = gtr.loc[:, gtr.notna().mean() > 0.9]
    med = gtr.median()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=R.C_FIXED, max_iter=4000))
    pipe.fit(gtr.fillna(med), y.reindex(gtr.index))
    return pipe.predict_proba(geom.reindex(test)[gtr.columns].fillna(med))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    man, ixa, Fa, source, _ = C.load()
    T = C.targets(man)
    img_case = ixa.exam_case_id.to_numpy()
    geoms = {arm: case_geom(arm) for arm in a.arms}
    res, rows = dict(arms={}, v1={}), []
    for f in FIELDS:
        y = T[f].astype(float)
        train = sorted(c for c in y.index if source[c] == "dev")
        test = sorted(c for c in y.index if source[c] == "locked")
        assert len(set(train) & set(test)) == 0 and len(test) <= 47
        y_img = np.where(np.isin(img_case, train), y.reindex(img_case).fillna(-1), -1).astype(int)
        tr = y_img >= 0
        te = np.isin(img_case, test)
        pa = C.fit_predict(Fa, img_case, y_img, tr, te, [0, 1])[1].reindex(test).to_numpy()
        yt = y.reindex(test).astype(int).to_numpy()
        res["v1"][f] = dict(n=len(test), n_pos=int(yt.sum()), ba=round(ba(yt, pa), 4),
                            auroc=round(float(roc_auc_score(yt, pa)), 4))
        rng = np.random.default_rng(R.SEED)
        boot = [rng.integers(0, len(yt), len(yt)) for _ in range(N_BOOT)]
        boot = [i for i in boot if len(set(yt[i])) == 2]
        for arm, g in geoms.items():
            pt = (5 * pa + geom_route(g, train, test, y)) / 6
            ht, hv = (pt >= 0.5).astype(int), (pa >= 0.5).astype(int)
            d = [balanced_accuracy_score(yt[i], ht[i]) - balanced_accuracy_score(yt[i], hv[i])
                 for i in boot]
            # null: geometry rows shuffled across all 233 cases, then the same fit
            null = []
            for s in range(N_NULL):
                r2 = np.random.default_rng(1000 + s)
                gs = g.copy()
                gs.index = r2.permutation(g.index.to_numpy())
                pn = (5 * pa + geom_route(gs, train, test, y)) / 6
                null.append(ba(yt, pn) - ba(yt, pa))
            real = ba(yt, pt) - ba(yt, pa)
            res["arms"].setdefault(arm, {})[f] = dict(
                ba_v1=round(ba(yt, pa), 4), ba_v3=round(ba(yt, pt), 4),
                auroc_v3=round(float(roc_auc_score(yt, pt)), 4),
                delta=round(real, 4),
                delta_ci95=[round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)],
                null_permuted_geometry=dict(
                    median=round(float(np.median(null)), 4),
                    p95=round(float(np.percentile(null, 95)), 4),
                    frac_ge_real=round(float(np.mean(np.array(null) >= real)), 3)),
                primary=f in PRIMARY)
            for c, x1, x2, t in zip(test, pa, pt, yt):
                rows.append(dict(arm=arm, field=f, exam_case_id=c, y_true=int(t),
                                 p_v1=round(float(x1), 4), p_v3=round(float(x2), 4)))
            print(f, arm, res["arms"][arm][f], flush=True)
    res["design"] = ("fit on dev 186 only, scored once on locked-47; v3 = (5*A0 + geometry)/6; "
                     "contaminated test set (7 prior reads, selection in 3); descriptive")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "v3_vs_v1%s.json" % a.tag), "w"), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_case%s.csv" % a.tag), index=False)


if __name__ == "__main__":
    main()
