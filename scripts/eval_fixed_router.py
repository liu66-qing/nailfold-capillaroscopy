"""Fixed router: every field gets the same four-expert average, no selection.

p = mean(A0, COL, SEG, DET) probabilities; decision: p1 > training prevalence.
Five-fold OOF over all labelled cases of the 233 (development_fold / fold).
Paired bootstrap BA(ALL) - BA(A0 same threshold) and BA(ALL) - BA(A0 at p>=0.5,
the shipped rule). The measurement 3-band row is composed from the two one-sided
heads: hi -> 偏大, else lo -> 偏小, else 参考; if both fire, the larger margin wins.

    python scripts/eval_fixed_router.py
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
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

BIN = [t for g in E.GROUP.values() for t in g if t not in ("papilla_3", "capillary_3")]


def oof(name, y, fold, ctx):
    y = y[y.index.isin(ctx["cases"])]
    fo = fold.reindex(y.index)
    rows = []
    for k in sorted(fo.unique()):
        te, tr = set(fo.index[fo == k]), set(fo.index[fo != k])
        prior = float(y.reindex(sorted(tr)).mean())
        P = E.all_arm_probs(tr, te, y, [0, 1], ctx)
        for c in P["ALL"].index:
            r = dict(target=name, exam_case_id=c, y=int(y[c]), prior=prior,
                     p_all=float(P["ALL"].loc[c, 1]), p_a0=float(P["A0"].loc[c, 1]))
            r.update({"p_" + a: float(P[a].loc[c, 1]) for a in E.BASE})
            rows.append(r)
    return pd.DataFrame(rows)


def paired(y, h1, h0, rng, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(balanced_accuracy_score(y[i], h1[i]) - balanced_accuracy_score(y[i], h0[i]))
    return [round(float(np.percentile(d, q)), 4) for q in (2.5, 50, 97.5)]


def main():
    man, ixa, Fa, source, fold = C.load()
    ctx = dict(Fa=Fa, img_case=ixa.exam_case_id.to_numpy(),
               routes=E.load_case_routes(ixa), cases=set(ixa.exam_case_id))
    T = E.build_targets(man)
    df = pd.concat(Parallel(n_jobs=6)(delayed(oof)(n, T[n], fold, ctx) for n in BIN))
    df.to_csv(os.path.join(E.OUT, "fixed_router_oof.csv"), index=False)
    res, rng = {}, np.random.default_rng(R.SEED)
    for n, g in df.groupby("target", sort=False):
        y = g.y.to_numpy()
        h = (g.p_all > g.prior).astype(int).to_numpy()
        h0 = (g.p_a0 > g.prior).astype(int).to_numpy()
        h05 = (g.p_a0 >= 0.5).astype(int).to_numpy()
        arch = man.archive.reindex(g.exam_case_id).to_numpy()
        lk = (source.reindex(g.exam_case_id) == "locked").to_numpy()
        res[n] = dict(n=len(y), pos=int(y.sum()), mode_acc=round(max(y.mean(), 1 - y.mean()), 4),
                      ba=round(float(balanced_accuracy_score(y, h)), 4),
                      auroc=round(float(roc_auc_score(y, g.p_all)), 4),
                      acc=round(float((y == h).mean()), 4),
                      recall=[round(float(r), 4) for r in recall_score(y, h, average=None)],
                      vs_a0_same_rule=paired(y, h, h0, rng),
                      vs_a0_shipped_rule=paired(y, h, h05, rng),
                      a0_shipped_ba=round(float(balanced_accuracy_score(y, h05)), 4),
                      archive_ba=[round(float(balanced_accuracy_score(y[arch == a], h[arch == a])), 4)
                                  for a in sorted(set(arch))],
                      locked_subset_ba=round(float(balanced_accuracy_score(y[lk], h[lk])), 4))
        print(n, json.dumps(res[n]), flush=True)
    # composed three bands for the four measurement fields
    comp = {}
    for f in ["afferent", "efferent", "apex", "loop"]:
        hi = df[df.target == f + "_hi"].set_index("exam_case_id")
        lo = df[df.target == f + "_lo"].set_index("exam_case_id")
        ix = hi.index.intersection(lo.index)
        hi, lo = hi.loc[ix], lo.loc[ix]
        y3 = np.where(hi.y == 1, 2, np.where(lo.y == 1, 0, 1))
        mh, ml = hi.p_all - hi.prior, lo.p_all - lo.prior
        p3 = np.where((mh > 0) & (mh >= ml), 2, np.where(ml > 0, 0, 1))
        comp[f] = dict(n=len(ix), counts=np.bincount(y3, minlength=3).tolist(),
                       ba3=round(float(balanced_accuracy_score(y3, p3)), 4),
                       recall3=[round(float(r), 4) for r in recall_score(y3, p3, average=None)],
                       acc=round(float((y3 == p3).mean()), 4),
                       mode_acc=round(float(np.bincount(y3).max() / len(y3)), 4))
        print(f, "3band", comp[f], flush=True)
    json.dump(dict(binary=res, composed_3band=comp, protocol=__doc__),
              open(os.path.join(E.OUT, "fixed_router.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
