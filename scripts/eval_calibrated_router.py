"""Fixed four-expert router + nested calibration + accuracy-optimal decision.

Router (same for every field, no selection): p = mean(A0, COL, SEG, DET).
The heads are strongly shrunk (C=0.03), so p is not a probability. Inside each
outer training set an inner 5-fold OOF p is produced and a 1-D logistic
calibration logit(p) -> y is fitted on it; the outer test cases get the
calibrated posterior q. Nothing from the outer test fold touches the fit.

Decision rules reported:
  acc   q > 0.5 (Bayes rule for accuracy; this is what a report line should use)
  ba    p > training prevalence (balanced-accuracy rule, for comparison)
  mode  always the training majority (the "fixed dictionary" answer)
Three-band measurement rows are composed from the calibrated one-sided heads:
P(hi), P(lo), P(ref) = 1 - P(hi) - P(lo) (floored at 0), argmax.
papilla uses flat / wavy heads the same way; capillary_count uses its binary
"below the >=7 band" head (the 3--4 and lower bands have 18 cases).

Five-fold over all labelled cases of the 233. Paired bootstrap accuracy(acc) -
accuracy(mode). Locked-47 subset reported descriptively.

    python scripts/eval_calibrated_router.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

BIN = [t for g in E.GROUP.values() for t in g if t not in ("papilla_3", "capillary_3")]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def p_all(train, test, y, ctx):
    P = E.all_arm_probs(train, test, y, [0, 1], ctx)["ALL"]
    return P[1]


def oof(name, y, fold, ctx):
    y = y[y.index.isin(ctx["cases"])]
    fo = fold.reindex(y.index)
    rows = []
    for k in sorted(fo.unique()):
        te, tr = set(fo.index[fo == k]), set(fo.index[fo != k])
        tf = fo.reindex(sorted(tr))
        inner = pd.concat([p_all(tr - set(tf.index[tf == j]), set(tf.index[tf == j]), y, ctx)
                           for j in sorted(tf.unique())])
        cal = LogisticRegression(C=1e4, max_iter=1000).fit(
            logit(inner.to_numpy())[:, None], y.reindex(inner.index).to_numpy())
        po = p_all(tr, te, y, ctx)
        q = cal.predict_proba(logit(po.to_numpy())[:, None])[:, 1]
        prior = float(y.reindex(sorted(tr)).mean())
        for c, p, qq in zip(po.index, po.to_numpy(), q):
            rows.append(dict(target=name, exam_case_id=c, y=int(y[c]), prior=prior,
                             p=float(p), q=float(qq), fold=int(k)))
    return pd.DataFrame(rows)


def boot_acc_delta(y, h, m, rng, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append((h[i] == y[i]).mean() - (m[i] == y[i]).mean())
    return [round(float(np.percentile(d, q)), 4) for q in (2.5, 50, 97.5)]


def summarise(y, h, m, lk, rng, score=None):
    s = dict(n=int(len(y)), counts=np.bincount(y).tolist(),
             acc=round(float((h == y).mean()), 4), mode_acc=round(float((m == y).mean()), 4),
             acc_minus_mode=boot_acc_delta(y, h, m, rng),
             ba=round(float(balanced_accuracy_score(y, h)), 4),
             recall=[round(float(r), 4) for r in recall_score(
                 y, h, labels=sorted(set(y)), average=None, zero_division=0)],
             pred_share=np.round(np.bincount(h, minlength=len(set(y))) / len(y), 3).tolist(),
             locked_acc=round(float((h[lk] == y[lk]).mean()), 4),
             locked_mode_acc=round(float((m[lk] == y[lk]).mean()), 4))
    if score is not None and len(set(y)) == 2:
        s["auroc"] = round(float(roc_auc_score(y, score)), 4)
    return s


def main():
    man, ixa, Fa, source, fold = C.load()
    ctx = dict(Fa=Fa, img_case=ixa.exam_case_id.to_numpy(),
               routes=E.load_case_routes(ixa), cases=set(ixa.exam_case_id))
    T = E.build_targets(man)
    df = pd.concat(Parallel(n_jobs=6)(delayed(oof)(n, T[n], fold, ctx) for n in BIN))
    df.to_csv(os.path.join(E.OUT, "calibrated_oof.csv"), index=False)
    rng = np.random.default_rng(R.SEED)
    res = {}
    for n, g in df.groupby("target", sort=False):
        y = g.y.to_numpy()
        lk = (source.reindex(g.exam_case_id) == "locked").to_numpy()
        m = (g.prior > 0.5).astype(int).to_numpy()
        res[n] = dict(
            acc_rule=summarise(y, (g.q > 0.5).astype(int).to_numpy(), m, lk, rng, g.q),
            ba_rule=summarise(y, (g.p > g.prior).astype(int).to_numpy(), m, lk, rng))
        for thr in (0.6, 0.7, 0.8):
            hi, lo = g.q >= thr, g.q <= 1 - thr
            res[n]["tier_%.1f" % thr] = dict(
                share_pos=round(float(hi.mean()), 3), ppv=round(float(y[hi].mean()), 3) if hi.any() else None,
                share_neg=round(float(lo.mean()), 3), npv=round(float(1 - y[lo].mean()), 3) if lo.any() else None)
        a = res[n]["acc_rule"]
        print("%-27s acc %.3f mode %.3f d %s | BA(acc) %.3f BA(ba) %.3f auc %.3f | lock %.3f/%.3f" % (
            n, a["acc"], a["mode_acc"], a["acc_minus_mode"], a["ba"], res[n]["ba_rule"]["ba"],
            a["auroc"], a["locked_acc"], a["locked_mode_acc"]), flush=True)
    comp = {}

    def three(hi_t, lo_t, mapping):
        hi = df[df.target == hi_t].set_index("exam_case_id")
        lo = df[df.target == lo_t].set_index("exam_case_id")
        ix = hi.index.intersection(lo.index)
        hi, lo = hi.loc[ix], lo.loc[ix]
        y3 = mapping(hi.y.to_numpy(), lo.y.to_numpy())
        P = np.stack([lo.q, np.clip(1 - hi.q - lo.q, 0, None), hi.q], 1)
        h3 = P.argmax(1)
        # the majority band of the training folds, per fold
        m3 = np.zeros(len(ix), int)
        for k in hi.fold.unique():
            f = (hi.fold != k).to_numpy()
            m3[~f] = np.bincount(y3[f], minlength=3).argmax()
        lk = (source.reindex(ix) == "locked").to_numpy()
        s = summarise(y3, h3, m3, lk, rng)
        pd.DataFrame(dict(y=y3, pred=h3, p_lo=P[:, 0], p_ref=P[:, 1], p_hi=P[:, 2]),
                     index=ix).to_csv(os.path.join(E.OUT, "calibrated_3band_%s.csv" % hi_t.split("_")[0]))
        return s

    both = lambda h, l: np.where(h == 1, 2, np.where(l == 1, 0, 1))
    for f in ["afferent", "efferent", "apex", "loop"]:
        comp[f] = three(f + "_hi", f + "_lo", both)
    comp["papilla"] = three("papilla_flat", "papilla_wavy", both)  # 0 波纹, 1 浅波纹, 2 平坦
    for f, s in comp.items():
        print("%-10s 3band acc %.3f mode %.3f d %s BA %.3f rec %s share %s lock %.3f/%.3f" % (
            f, s["acc"], s["mode_acc"], s["acc_minus_mode"], s["ba"], s["recall"], s["pred_share"],
            s["locked_acc"], s["locked_mode_acc"]), flush=True)
    json.dump(dict(binary=res, three_band=comp, protocol=__doc__),
              open(os.path.join(E.OUT, "calibrated_router.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
