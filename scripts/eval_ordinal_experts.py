"""Ordinal three-band experts for the measurement rows and papilla.

Why a new expert (declared before any number was seen):
  The shipped three-band rows are composed from two independent one-sided binary
  heads. The low-side heads have 15% positives and collapse (efferent_lo /
  apex_lo recall 0), and neither head sees the order of the three bands or the
  continuous value the doctor wrote. This expert changes the task form and the
  input, not only the weights:

  1. target  = the continuous value, rank-transformed inside the training fold
               (papilla: its ordered class 0/1/2). One shared direction for all
               three bands, every case contributes, not only the 15% tail.
  2. sources = A0 (DINOv2 case mean, 5 poolings, PCA64), COL, SEG, DET as in the
               router, plus MOR: explicit per-loop morphometry from the S0 masks
               (limb widths from the distance transform on the vessel pixels,
               apex/bottom widths, loop height, ridge length, contrast), which no
               earlier source measured. Each source -> RidgeCV score (alpha picked
               by GCV on the training fold only).
  3. fusion  = O-stack: ordinal logit on the five inner-OOF source scores
               (per-field learned weights + two cut points), fitted inside each
               outer training set. Decision = argmax posterior (Bayes rule for
               accuracy), the same rule the shipped rows use.

Variants, all reported (3 x 5 = 15 comparisons against the shipped row):
  O-stack (primary), O-mean (equal z-weights), O-stack without MOR (attribution).

Protocol: the same 5 outer folds as eval_calibrated_router.py (development_fold,
fold for locked), inner 4-fold inside each training set. Baselines: the training
fold's majority band, and the shipped composed row (calibrated_3band_*.csv).
Paired bootstrap 2000. Locked-47 subset descriptive only.

    python scripts/eval_ordinal_experts.py --morph .tmp_probe/morph_image.csv
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
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

OUT = os.path.join(E.OUT, "ordinal")
ALPHAS = np.logspace(0, 4, 9)
SOURCES = ["A0", "COL", "SEG", "DET", "MOR"]
FIELDS = {"afferent": "afferent_diameter", "efferent": "efferent_diameter",
          "apex": "apex_diameter", "loop": "loop_length", "papilla": "papilla"}


# ---------- ordinal logit: P(y <= j) = sigmoid(c_j - x.w) ----------
def _sig(z):
    return 1 / (1 + np.exp(-z))


def ord_fit(X, y, l2=1e-2):
    X = np.atleast_2d(X)
    d = X.shape[1]

    def unpack(t):
        return t[:d], np.array([t[d], t[d] + np.exp(t[d + 1])])

    def nll(t):
        w, c = unpack(t)
        s = X @ w
        cum = np.stack([_sig(c[0] - s), _sig(c[1] - s)], 1)
        P = np.stack([cum[:, 0], cum[:, 1] - cum[:, 0], 1 - cum[:, 1]], 1)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None)).sum() + l2 * (w ** 2).sum()

    t0 = np.r_[np.zeros(d), -0.5, 0.0]
    r = minimize(nll, t0, method="L-BFGS-B")
    return unpack(r.x)


def ord_proba(model, X):
    w, c = model
    s = np.atleast_2d(X) @ w
    cum = np.stack([_sig(c[0] - s), _sig(c[1] - s)], 1)
    return np.stack([cum[:, 0], cum[:, 1] - cum[:, 0], 1 - cum[:, 1]], 1)


# ---------- per-source scorers ----------
def fit_source(name, X, tr, target):
    Xtr = X.reindex(tr)
    if name == "A0":
        pipes = []
        for p, F in X.attrs["pools"].items():
            pipe = make_pipeline(StandardScaler(), PCA(64, random_state=R.SEED),
                                 RidgeCV(alphas=ALPHAS))
            pipes.append((p, pipe.fit(F.reindex(tr).to_numpy(), target)))
        return ("A0", pipes)
    keep = Xtr.columns[Xtr.notna().mean() > 0.9]
    med = Xtr[keep].median()
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pipe.fit(Xtr[keep].fillna(med).to_numpy(), target)
    return ("case", keep, med, pipe)


def score_source(model, X, te):
    if model[0] == "A0":
        return np.mean([pipe.predict(X.attrs["pools"][p].reindex(te).to_numpy())
                        for p, pipe in model[1]], 0)
    _, keep, med, pipe = model
    return pipe.predict(X.reindex(te)[keep].fillna(med).to_numpy())


def rank_target(v):
    return (rankdata(v) - 0.5) / len(v)


def source_scores(srcs, tr, te, val):
    t = rank_target(val.reindex(tr).to_numpy())
    return np.stack([score_source(fit_source(s, srcs[s], tr, t), srcs[s], te)
                     for s in srcs], 1)


def run_field(key, val, y3, fold, srcs_all):
    cases = sorted(set(y3.index) & set(srcs_all["SEG"].index))
    y3, val, fo = y3.reindex(cases), val.reindex(cases), fold.reindex(cases)
    variants = {"O-stack": SOURCES, "O-mean": SOURCES, "O-stack-noMOR": SOURCES[:4]}
    out = {v: [] for v in variants}
    for k in sorted(fo.unique()):
        te = sorted(fo.index[fo == k])
        tr = sorted(fo.index[fo != k])
        tf = fo.reindex(tr)
        inner = []
        for j in sorted(tf.unique()):
            ite, itr = sorted(tf.index[tf == j]), sorted(tf.index[tf != j])
            inner.append(pd.DataFrame(source_scores(srcs_all, itr, ite, val),
                                      index=ite, columns=SOURCES))
        S_in = pd.concat(inner).reindex(tr)
        S_te = pd.DataFrame(source_scores(srcs_all, tr, te, val), index=te, columns=SOURCES)
        ytr = y3.reindex(tr).to_numpy()
        mode = int(np.bincount(ytr, minlength=3).argmax())
        mu, sd = S_in.mean(), S_in.std().replace(0, 1)
        for v, cols in variants.items():
            Zi, Zt = ((S_in[cols] - mu[cols]) / sd[cols]), ((S_te[cols] - mu[cols]) / sd[cols])
            if v == "O-mean":
                Zi, Zt = Zi.mean(1).to_frame(), Zt.mean(1).to_frame()
            m = ord_fit(Zi.to_numpy(), ytr)
            P = ord_proba(m, Zt.to_numpy())
            out[v].append(pd.DataFrame(dict(y=y3.reindex(te).to_numpy(), pred=P.argmax(1),
                                            p0=P[:, 0], p1=P[:, 1], p2=P[:, 2],
                                            mode=mode, fold=int(k)), index=te))
        print(key, "fold", k, "done", flush=True)
    return key, {v: pd.concat(d) for v, d in out.items()}


def boot(y, a, b, rng, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append((a[i] == y[i]).mean() - (b[i] == y[i]).mean())
    return [round(float(np.percentile(d, q)), 4) for q in (2.5, 50, 97.5)]


def load_mor(path, ixa):
    d = pd.read_csv(path, dtype={"exam_case_id": str})
    assert (d.image_path.to_numpy() == ixa.image_path.to_numpy()).all()
    return d.drop(columns=["image_path"]).groupby("exam_case_id").median()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man, ixa, Fa, source, fold = C.load()
    routes = E.load_case_routes(ixa)
    img_case = ixa.exam_case_id.to_numpy()
    A0 = pd.DataFrame(index=sorted(set(img_case)))
    A0.attrs["pools"] = {p: pd.DataFrame(F, index=img_case).groupby(level=0).mean()
                         for p, F in Fa.items()}
    srcs = {"A0": A0, "COL": routes["COL"], "SEG": routes["SEG"], "DET": routes["DET"],
            "MOR": load_mor(a.morph, ixa)}
    jobs = []
    for key, f in FIELDS.items():
        if key == "papilla":
            y3 = R.target(man, "papilla")[0].astype(int)
            val = y3.astype(float)
        else:
            lo, hi = E.MEAS[f]
            val = pd.to_numeric(R.clean_column(man, f), errors="coerce").dropna()
            y3 = pd.Series(np.where(val > hi, 2, np.where(val < lo, 0, 1)), index=val.index)
        jobs.append((key, val, y3))
    res = Parallel(n_jobs=5)(delayed(run_field)(k, v, y, fold, srcs) for k, v, y in jobs)
    rng = np.random.default_rng(R.SEED)
    summary = {}
    for key, outs in res:
        ship = pd.read_csv(os.path.join(E.OUT, "calibrated_3band_%s.csv" % key),
                           dtype={"exam_case_id": str}).set_index("exam_case_id")
        summary[key] = {}
        for v, df in outs.items():
            df.to_csv(os.path.join(OUT, "%s_%s.csv" % (key, v)))
            ix = df.index.intersection(ship.index)
            d, s = df.loc[ix], ship.loc[ix]
            assert (d.y.to_numpy() == s.y.to_numpy()).all(), key
            y, h, m, h0 = d.y.to_numpy(), d.pred.to_numpy(), d["mode"].to_numpy(), s.pred.to_numpy()
            lk = (source.reindex(ix) == "locked").to_numpy()
            summary[key][v] = dict(
                n=int(len(y)), counts=np.bincount(y, minlength=3).tolist(),
                acc=round(float((h == y).mean()), 4), mode_acc=round(float((m == y).mean()), 4),
                shipped_acc=round(float((h0 == y).mean()), 4),
                minus_mode=boot(y, h, m, rng), minus_shipped=boot(y, h, h0, rng),
                recall=[round(float(r), 3) for r in recall_score(y, h, labels=[0, 1, 2],
                                                                   average=None, zero_division=0)],
                shipped_recall=[round(float(r), 3) for r in recall_score(
                    y, h0, labels=[0, 1, 2], average=None, zero_division=0)],
                qwk=round(float(cohen_kappa_score(y, h, weights="quadratic")), 4),
                shipped_qwk=round(float(cohen_kappa_score(y, h0, weights="quadratic")), 4),
                pred_share=np.round(np.bincount(h, minlength=3) / len(h), 3).tolist(),
                locked_acc=round(float((h[lk] == y[lk]).mean()), 4),
                locked_shipped_acc=round(float((h0[lk] == y[lk]).mean()), 4))
            r = summary[key][v]
            print("%-8s %-14s acc %.3f mode %.3f ship %.3f | -mode %s -ship %s | rec %s (ship %s) qwk %.3f/%.3f" % (
                key, v, r["acc"], r["mode_acc"], r["shipped_acc"], r["minus_mode"], r["minus_shipped"],
                r["recall"], r["shipped_recall"], r["qwk"], r["shipped_qwk"]), flush=True)
    json.dump(dict(summary=summary, protocol=__doc__),
              open(os.path.join(OUT, "ordinal_experts.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
