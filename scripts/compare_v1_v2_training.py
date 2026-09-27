"""Does adding the 47 locked cases to training (rag_heads_v2) rescue fields?

Paired comparison on identical test cases, leave-one-archive-out:
  test  : every labelled case (dev + locked) of the held-out archive
  arm v1: trained on the dev cases of the other two archives only
  arm v2: trained on dev + locked cases of the other two archives
Everything else is identical: A0 features, five poolings, StandardScaler ->
PCA64 -> LogReg C=0.03, mean over poolings then over the case's images.
Binary fields: threshold 0.5, plus abstention at the 40th percentile of
|p-0.5| on an inner 5-fold OOF of the training archives. Three-class fields:
argmax of the mean probability, no abstention.

"Internally usable" (fixed before running, same bar for both arms):
  full-coverage BA >= 0.65, BA 95% CI lower bound > 0.55, every class recall
  >= 0.50, and BA >= 0.60 in every held-out archive.
The count of usable fields per arm answers the question. Paired bootstrap of
BA(v2) - BA(v1) over cases gives the per-field gain.

  python scripts/compare_v1_v2_training.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_field_matrix_three_arms as R  # noqa: E402
from build_field_matrix import IN_SCOPE  # noqa: E402

LOCKED_FEAT = ROOT / "artifacts/experiments/locked_consumed_20260924/features_locked"
OUT = ROOT / "artifacts/experiments/v1_vs_v2_training_20260928"
KEEP, N_BOOT = 0.60, 2000
BAR = dict(ba=0.65, ci_low=0.55, recall=0.50, archive_ba=0.60)


def load():
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    ix, F, _ = R.load_arm(R.ANCHOR)
    lix = pd.read_csv(LOCKED_FEAT / "index.csv", dtype={"exam_case_id": str})
    LF = {p: np.load(LOCKED_FEAT / ("features_%s.npy" % p)).astype(np.float32)
          for p in R.POOLINGS}
    ixa = pd.concat([ix, lix], ignore_index=True)
    Fa = {p: np.concatenate([F[p], LF[p]]) for p in R.POOLINGS}
    source = pd.Series(np.where(man.development_fold.isna(), "locked", "dev"),
                       index=man.index)
    fold = man.development_fold.where(man.development_fold.notna(), man.fold).astype(int)
    return man, ixa, Fa, source, fold


def targets(man):
    out = {}
    for f in IN_SCOPE:
        if f == "loop_length":
            v = pd.to_numeric(R.clean_column(man, f), errors="coerce").dropna()
            out[f] = (v > 250).astype(int)
        else:
            y, k, _ = R.target(man, f)
            out[f] = y.astype(int)
    return out


def fit_predict(F, img_case, y_img, tr, te, classes):
    probs = []
    for p in R.POOLINGS:
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=R.DIM_FIXED, random_state=R.SEED),
                             LogisticRegression(C=R.C_FIXED, max_iter=4000))
        pipe.fit(F[p][tr], y_img[tr])
        pr = pipe.predict_proba(F[p][te])
        full = np.zeros((te.sum(), len(classes)))
        for j, c in enumerate(pipe.classes_):
            full[:, classes.index(c)] = pr[:, j]
        probs.append(full)
    return pd.DataFrame(np.mean(probs, 0), index=img_case[te]).groupby(level=0).mean()


def loao(y, arch, source, fold, ixa, Fa, arm):
    img_case = ixa.exam_case_id.to_numpy()
    lab = np.isin(img_case, y.index.to_numpy())
    y_img = np.where(lab, y.reindex(img_case).fillna(-1).to_numpy(), -1).astype(int)
    classes = sorted(y.unique())
    binary = len(classes) == 2
    allowed = set(y.index) if arm == "v2" else set(y.index[source.reindex(y.index) == "dev"])
    rows = []
    for a in sorted(arch.unique()):
        test = set(y.index[arch.reindex(y.index) == a])
        train = {c for c in allowed if arch[c] != a}
        tr, te = np.isin(img_case, list(train)), np.isin(img_case, list(test))
        P = fit_predict(Fa, img_case, y_img, tr, te, classes)
        margin = 0.0
        if binary:
            inner = []
            tf = fold.reindex(sorted(train))
            for fo in sorted(tf.unique()):
                it = set(tf.index[tf == fo])
                s = fit_predict(Fa, img_case, y_img,
                                np.isin(img_case, list(train - it)),
                                np.isin(img_case, list(it)), classes)
                inner.extend(s[1].tolist())
            margin = float(np.quantile(np.abs(np.array(inner) - 0.5), 1 - KEEP))
        for c, row in P.iterrows():
            pr = row.to_numpy()
            rows.append(dict(exam_case_id=c, archive=a, source=source[c],
                             y_true=int(y[c]), pred=int(classes[int(pr.argmax())]),
                             p1=float(pr[-1]) if binary else np.nan,
                             abstain=bool(binary and abs(pr[-1] - 0.5) < margin),
                             margin=margin, n_train=len(train)))
    return pd.DataFrame(rows).set_index("exam_case_id").sort_index()


def summarise(df, classes):
    y, h = df.y_true.to_numpy(), df.pred.to_numpy()
    rec = recall_score(y, h, labels=classes, average=None, zero_division=0)
    rng = np.random.default_rng(R.SEED)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == len(classes):
            bs.append(balanced_accuracy_score(y[i], h[i]))
    arch_ba = {a: round(float(balanced_accuracy_score(g.y_true, g.pred)), 4)
               for a, g in df.groupby("archive") if g.y_true.nunique() == len(classes)}
    out = dict(n=len(y), ba=round(float(balanced_accuracy_score(y, h)), 4),
               ci=[round(float(np.percentile(bs, 2.5)), 4),
                   round(float(np.percentile(bs, 97.5)), 4)],
               recalls=[round(float(r), 4) for r in rec], archive_ba=arch_ba,
               majority_class_share=round(float(pd.Series(y).value_counts(normalize=True).max()), 4))
    out["usable"] = bool(out["ba"] >= BAR["ba"] and out["ci"][0] > BAR["ci_low"]
                         and min(rec) >= BAR["recall"]
                         and len(arch_ba) == 3 and min(arch_ba.values()) >= BAR["archive_ba"])
    ab = df.abstain.to_numpy()
    if ab.any() and len(set(y[~ab])) == 2:
        r2 = recall_score(y[~ab], h[~ab], labels=classes, average=None, zero_division=0)
        out["selective"] = dict(coverage=round(float((~ab).mean()), 4),
                                ba=round(float(balanced_accuracy_score(y[~ab], h[~ab])), 4),
                                recalls=[round(float(r), 4) for r in r2])
    return out


def run_field(args):
    f, y = args
    man, ixa, Fa, source, fold = _G
    arch = man.archive
    res = {}
    dfs = {}
    for arm in ("v1", "v2"):
        d = loao(y, arch, source, fold, ixa, Fa, arm)
        d.to_csv(OUT / ("per_case_%s_%s.csv" % (f, arm)))
        dfs[arm] = d
        res[arm] = summarise(d, sorted(y.unique()))
    a, b = dfs["v1"], dfs["v2"].reindex(dfs["v1"].index)
    assert (a.y_true == b.y_true).all()
    rng = np.random.default_rng(R.SEED)
    yy, ha, hb = a.y_true.to_numpy(), a.pred.to_numpy(), b.pred.to_numpy()
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yy), len(yy))
        if len(set(yy[i])) == len(set(yy)):
            d.append(balanced_accuracy_score(yy[i], hb[i]) - balanced_accuracy_score(yy[i], ha[i]))
    res["delta_v2_minus_v1"] = dict(
        ba=round(res["v2"]["ba"] - res["v1"]["ba"], 4),
        ci=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        predictions_changed=int((ha != hb).sum()))
    # the same comparison restricted to the 186 dev test cases, where v1's
    # training never contained a same-hospital case that v2 had and v1 lacked
    m = (a.source == "dev").to_numpy()
    res["dev_test_only_ba"] = dict(v1=round(float(balanced_accuracy_score(yy[m], ha[m])), 4),
                                   v2=round(float(balanced_accuracy_score(yy[m], hb[m])), 4))
    return f, res


_G = None


def _init():
    global _G
    _G = load()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man, *_ = load()
    ys = targets(man)
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(max_workers=len(ys), initializer=_init) as ex:
        for f, r in ex.map(run_field, list(ys.items())):
            out[f] = r
            print(f, "v1", r["v1"]["ba"], r["v1"]["usable"], "| v2", r["v2"]["ba"],
                  r["v2"]["usable"], "| delta", r["delta_v2_minus_v1"], flush=True)
    summary = dict(bar=BAR, fields=out,
                   usable_v1=[f for f, r in out.items() if r["v1"]["usable"]],
                   usable_v2=[f for f, r in out.items() if r["v2"]["usable"]])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("usable v1:", summary["usable_v1"])
    print("usable v2:", summary["usable_v2"])


if __name__ == "__main__":
    main()
