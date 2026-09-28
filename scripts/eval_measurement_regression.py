"""Measurement fields as one regression per field, cut at the printed reference range.

Alternative to the two one-sided heads (Codex round-4 suggestion): a single model
with a shared direction. Target log(value), winsorised at the training 2/98%.
Features: four blocks per case (A0 mean-pooling case mean, COL, SEG, DET), each
median-filled, standardised and PCA'd to 16 dims inside the training fold, then
Ridge(alpha=30) (fixed, declared). Band probabilities from a normal residual model
whose sd is the training folds' inner-OOF residual sd. Five-fold OOF over all
labelled cases of the 233 plus LOAO.

    python scripts/eval_measurement_regression.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

ALPHA, DIM = 30.0, 16


def blocks(ixa, Fa):
    A0 = pd.DataFrame(Fa["mean"], index=ixa.exam_case_id).groupby(level=0).mean()
    b = E.load_case_routes(ixa)
    b["A0"] = A0
    return b


def design(B, train, test):
    tr, te = [], []
    for k, X in B.items():
        Xt = X.reindex(train)
        keep = Xt.columns[Xt.notna().mean() > 0.9]
        med = Xt[keep].median()
        p = make_pipeline(StandardScaler(), PCA(DIM, random_state=R.SEED))
        tr.append(p.fit_transform(Xt[keep].fillna(med)))
        te.append(p.transform(X.reindex(test)[keep].fillna(med)))
    return np.hstack(tr), np.hstack(te)


def fit_pred(B, v, train, test):
    lo, hi = np.percentile(v.reindex(train), [2, 98])
    yt = np.log(v.reindex(train).clip(lo, hi))
    Xtr, Xte = design(B, train, test)
    m = Ridge(alpha=ALPHA).fit(Xtr, yt)
    return m.predict(Xte)


def band_probs(mu, sd, lo, hi):
    a, b = np.log(lo), np.log(hi)
    p_lo = norm.cdf((a - mu) / sd)
    p_hi = 1 - norm.cdf((b - mu) / sd)
    return np.stack([p_lo, 1 - p_lo - p_hi, p_hi], 1)


def run(B, v, groups, lo, hi):
    out = pd.DataFrame(index=v.index, columns=["mu", "sd"], dtype=float)
    for g in sorted(groups.unique()):
        te = list(v.index[groups == g])
        tr = list(v.index[groups != g])
        # inner residual sd on the training side only
        res = []
        ig = groups.reindex(tr)
        for j in sorted(ig.unique()):
            itr = [c for c in tr if ig[c] != j]
            ite = [c for c in tr if ig[c] == j]
            res.append(np.log(v.reindex(ite)) - fit_pred(B, v, itr, ite))
        sd = float(np.std(np.concatenate(res)))
        out.loc[te, "mu"] = fit_pred(B, v, tr, te)
        out.loc[te, "sd"] = sd
    P = band_probs(out.mu.to_numpy(), out.sd.to_numpy(), lo, hi)
    return out, P


def main():
    man, ixa, Fa, source, fold = C.load()
    B = blocks(ixa, Fa)
    arch = man.archive
    for f, (lo, hi) in E.MEAS.items():
        v = pd.to_numeric(R.clean_column(man, f), errors="coerce").dropna()
        v = v[v.index.isin(set(ixa.exam_case_id)) & (v > 0)]
        y = np.where(v > hi, 2, np.where(v < lo, 0, 1))
        for name, groups in (("5fold", fold.reindex(v.index)), ("loao", arch.reindex(v.index))):
            o, P = run(B, v, groups, lo, hi)
            h = P.argmax(1)
            # declared balanced rule: divide by the band prior of the whole set
            pri = np.bincount(y, minlength=3) / len(y)
            hb = (P / pri).argmax(1)
            rho = pd.Series(o.mu.to_numpy()).corr(pd.Series(np.log(v.to_numpy())), method="spearman")
            mode = np.bincount(y).argmax()
            print("%-18s %-5s rho %.3f | acc %.3f mode %.3f BA %.3f rec %s | BArule BA %.3f rec %s share %s" % (
                f, name, rho, (h == y).mean(), (y == mode).mean(), balanced_accuracy_score(y, h),
                np.round(recall_score(y, h, average=None, zero_division=0), 2),
                balanced_accuracy_score(y, hb),
                np.round(recall_score(y, hb, average=None, zero_division=0), 2),
                np.round(np.bincount(hb, minlength=3) / len(y), 2)), flush=True)
            if name == "5fold":
                pd.DataFrame(dict(y=y, value=v.to_numpy(), mu=o.mu, sd=o.sd, p_lo=P[:, 0],
                                  p_ref=P[:, 1], p_hi=P[:, 2]), index=v.index).to_csv(
                    os.path.join(E.OUT, "regression_%s.csv" % f))


if __name__ == "__main__":
    main()
