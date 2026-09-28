"""Leave-one-archive-out check for the fixed four-expert router.

Train on two archives (all labelled cases of the 233 in them), test on the third.
AUROC and prevalence-rule BA for ALL, each single route and A0, per archive.
Answers: do the COL / SEG / DET gains transfer across acquisition batches, or do
they learn an archive's colour/device signature?

    python scripts/eval_router_loao.py
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
import eval_calibrated_router as K  # noqa: E402

ARMS = ["A0", "COL", "SEG", "DET", "ALL", "A0+SEG", "A0+DET", "SEG+DET"]


def run(name, y, arch, ctx):
    y = y[y.index.isin(ctx["cases"])]
    rows = []
    for a in sorted(arch.unique()):
        te = set(y.index[arch.reindex(y.index) == a])
        tr = set(y.index) - te
        prior = float(y.reindex(sorted(tr)).mean())
        P = E.all_arm_probs(tr, te, y, [0, 1], ctx)
        for arm in ARMS:
            p = P[arm][1]
            yt = y.reindex(p.index).to_numpy()
            rows.append(dict(target=name, archive=a, arm=arm, n=len(yt),
                             auroc=roc_auc_score(yt, p) if len(set(yt)) == 2 else np.nan,
                             ba=balanced_accuracy_score(yt, (p > prior).astype(int))))
    return pd.DataFrame(rows)


def main():
    man, ixa, Fa, source, fold = C.load()
    ctx = dict(Fa=Fa, img_case=ixa.exam_case_id.to_numpy(),
               routes=E.load_case_routes(ixa), cases=set(ixa.exam_case_id))
    T = E.build_targets(man)
    df = pd.concat(Parallel(n_jobs=6)(delayed(run)(n, T[n], man.archive, ctx) for n in K.BIN))
    df.to_csv(os.path.join(E.OUT, "router_loao.csv"), index=False)
    piv = df.pivot_table(index="target", columns="arm", values="auroc", aggfunc="mean")[ARMS]
    pb = df.pivot_table(index="target", columns="arm", values="ba", aggfunc="mean")[ARMS]
    pd.set_option("display.width", 200)
    print("mean LOAO AUROC\n", piv.round(3).reindex(K.BIN))
    print("mean LOAO BA\n", pb.round(3).reindex(K.BIN))
    print("per-archive AUROC, ALL\n", df[df.arm == "ALL"].pivot(index="target", columns="archive",
                                                               values="auroc").round(3).reindex(K.BIN))


if __name__ == "__main__":
    main()
