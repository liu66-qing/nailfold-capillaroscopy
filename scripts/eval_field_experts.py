"""Per-field expert routing on all 233 cases, with the route chosen inside the CV.

Question: the user's design gives measurement fields (afferent/efferent/apex,
capillary_count, loop_length, papilla, crossing) to a measurement expert and
appearance fields (clarity, SVP, exudation, blood_color, microthrombus,
malformation) to a classification expert. Does giving each field its own
feature source and its own task form beat the single A0 model?

Feature routes (all exist for all 233 cases; no labels were used to build them):
  A0   DINOv2-B deployed, 5 poolings, PCA64, LogReg C=0.03 (image level, case mean)
  COL  handcrafted colour / sharpness statistics over 4 regions (case mean)
  SEG  S0 segmenter instance geometry, conf 0.05 (case mean)
  DET  Capillary-Dataset detector (bushy/crossing/hairpin/tortuous) case features
  plus fixed equal-weight averages: A0+COL, A0+SEG, A0+DET, SEG+DET, ALL
Case-level routes: median fill, StandardScaler, LogReg C=0.03.

Task forms, declared here before any number was seen. Measurement fields use the
report's printed reference-range edges as binary "outside the range on this side"
questions, because the 3-band form was at chance and the label grid is 1 unit.

Decision rule, declared: binary -> positive when p > training prevalence;
multiclass -> argmax of p / training prior. p >= 0.5 collapses imbalanced
targets to the majority class, which is the "guessing" the user saw.

Protocol: 5-fold CV over all labelled cases of the 233 (development_fold for
dev, fold for locked). The route is picked by inner-CV balanced accuracy inside
each outer training set only (nested), so the reported number carries no
selection bias. Also reported: every route's plain outer OOF, the always-mode
baseline (BA 0.5 by construction, plus accuracy), per archive, and the
locked-47 subset of the outer predictions.

    python scripts/eval_field_experts.py [--jobs 4]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

OUT = os.path.join(ROOT, "artifacts/experiments/expert_routes_20260928")
DET_DEV = os.path.join(ROOT, "artifacts/experiments/rescue_external_20260922/local_features/external/case_features.csv")
DET_LOCK = os.path.join(ROOT, "artifacts/experiments/locked_consumed_20260924/features_locked/case_features.csv")
SEG = "E:/nailfold_tmp/seg_geom/S0_c005/image_geometry.csv"
BASE = ["A0", "COL", "SEG", "DET"]
COMBOS = {"A0+COL": ["A0", "COL"], "A0+SEG": ["A0", "SEG"], "A0+DET": ["A0", "DET"],
          "SEG+DET": ["SEG", "DET"], "ALL": BASE}
ARMS = BASE + list(COMBOS)
N_BOOT = 2000

MEAS = {"afferent_diameter": (9.0, 13.0), "efferent_diameter": (11.0, 17.0),
        "apex_diameter": (12.0, 18.0), "loop_length": (150.0, 250.0)}
GROUP = {"measurement": ["afferent_hi", "afferent_lo", "efferent_hi", "efferent_lo",
                         "apex_hi", "apex_lo", "loop_hi", "loop_lo", "capillary_lo",
                         "capillary_3", "papilla_3", "papilla_flat", "papilla_wavy",
                         "crossing"],
         "classification": ["clarity", "subpapillary_venous_plexus", "exudation",
                            "blood_color", "microthrombus", "malformation",
                            "flow_state", "rbc_aggregation"]}


def build_targets(man):
    t = {}
    for f, (lo, hi) in MEAS.items():
        v = pd.to_numeric(R.clean_column(man, f), errors="coerce").dropna()
        k = f.split("_")[0]
        t[k + "_hi"] = (v > hi).astype(int)
        t[k + "_lo"] = (v < lo).astype(int)
    for f in ["clarity", "subpapillary_venous_plexus", "exudation", "blood_color",
              "microthrombus", "flow_state", "rbc_aggregation"]:
        t[f] = R.target(man, f)[0].astype(int)
    t["malformation"] = R.target(man, "malformation_ratio")[0].astype(int)
    t["crossing"] = R.target(man, "crossing_ratio")[0].astype(int)
    cap = R.target(man, "capillary_count")[0].astype(int)
    t["capillary_3"] = cap
    t["capillary_lo"] = (cap > 0).astype(int)
    pap = R.target(man, "papilla")[0].astype(int)
    t["papilla_3"] = pap
    t["papilla_flat"] = (pap == 2).astype(int)
    t["papilla_wavy"] = (pap == 0).astype(int)
    return t


def load_case_routes(ixa):
    col = np.load(os.path.join(OUT, "color", "features.npy"))
    cix = pd.read_csv(os.path.join(OUT, "color", "index.csv"), dtype={"exam_case_id": str})
    assert (cix.image_path.to_numpy() == ixa.image_path.to_numpy()).all()
    COL = pd.DataFrame(col, index=cix.exam_case_id).groupby(level=0).mean()
    seg = pd.read_csv(SEG, dtype={"exam_case_id": str})
    num = seg.select_dtypes(include=[np.number]).columns.drop(["img_w", "img_h"], errors="ignore")
    SEGF = seg.groupby("exam_case_id")[list(num)].mean()
    d = []
    for p in (DET_DEV, DET_LOCK):
        x = pd.read_csv(p, dtype={"exam_case_id": str}, encoding="utf-8-sig")
        d.append(x.set_index("exam_case_id"))
    DET = pd.concat(d)
    DET = DET.drop(columns=[c for c in DET.columns if not np.issubdtype(DET[c].dtype, np.number)])
    DET = DET.astype(float)
    return {"COL": COL, "SEG": SEGF, "DET": DET}


def case_fit(X, train, test, y, classes):
    Xtr = X.reindex(sorted(train))
    keep = Xtr.columns[Xtr.notna().mean() > 0.9]
    Xtr = Xtr[keep]
    med = Xtr.median()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=R.C_FIXED, max_iter=4000))
    pipe.fit(Xtr.fillna(med).to_numpy(), y.reindex(Xtr.index).to_numpy())
    te = sorted(test)
    pr = pipe.predict_proba(X.reindex(te)[keep].fillna(med).to_numpy())
    full = np.zeros((len(te), len(classes)))
    for j, c in enumerate(pipe.classes_):
        full[:, classes.index(c)] = pr[:, j]
    return pd.DataFrame(full, index=te)


def route_probs(arm, train, test, y, classes, ctx):
    if arm == "A0":
        img_case = ctx["img_case"]
        y_img = np.where(np.isin(img_case, list(train)), y.reindex(img_case).fillna(-1), -1).astype(int)
        P = C.fit_predict(ctx["Fa"], img_case, y_img, y_img >= 0,
                          np.isin(img_case, list(test)), classes)
        return P.reindex(sorted(test))
    return case_fit(ctx["routes"][arm], train, test, y, classes)


def decide(P, prior):
    P = np.asarray(P)
    if P.shape[1] == 2:
        return (P[:, 1] > prior[1]).astype(int)
    return (P / prior).argmax(1)


def all_arm_probs(train, test, y, classes, ctx):
    base = {a: route_probs(a, train, test, y, classes, ctx) for a in BASE}
    for k, parts in COMBOS.items():
        base[k] = sum(base[p] for p in parts) / len(parts)
    return base


def run_target(name, y, fold, source, arch, ctx):
    y = y[y.index.isin(ctx["cases"])]
    classes = sorted(y.unique())
    fo = fold.reindex(y.index)
    outer = {a: [] for a in ARMS}
    nested, chosen = [], []
    for k in sorted(fo.unique()):
        te = set(fo.index[fo == k])
        tr = set(fo.index[fo != k])
        prior = y.reindex(sorted(tr)).value_counts(normalize=True).reindex(classes).to_numpy()
        # inner CV on the outer training set only
        inner = {a: [] for a in ARMS}
        tf = fo.reindex(sorted(tr))
        for j in sorted(tf.unique()):
            ite = set(tf.index[tf == j])
            itr = tr - ite
            ip = y.reindex(sorted(itr)).value_counts(normalize=True).reindex(classes).to_numpy()
            for a, P in all_arm_probs(itr, ite, y, classes, ctx).items():
                inner[a].append(pd.Series(decide(P.to_numpy(), ip), index=P.index))
        yt = y.reindex(sorted(tr))
        score = {a: balanced_accuracy_score(yt, pd.concat(inner[a]).reindex(yt.index))
                 for a in ARMS}
        best = max(ARMS, key=lambda a: (round(score[a], 4), -ARMS.index(a)))
        chosen.append(best)
        Po = all_arm_probs(tr, te, y, classes, ctx)
        for a, P in Po.items():
            outer[a].append(pd.DataFrame(P.to_numpy(), index=P.index).assign(
                _pred=decide(P.to_numpy(), prior)))
        nested.append(outer[best][-1].assign(_arm=best))
    res = dict(target=name, n=int(len(y)), classes=[int(c) for c in classes],
               class_counts={int(c): int(v) for c, v in y.value_counts().sort_index().items()},
               mode_accuracy=round(float(y.value_counts(normalize=True).max()), 4),
               chosen_per_fold=chosen, arms={})

    def summ(df):
        yy = y.reindex(df.index).to_numpy()
        h = df._pred.to_numpy().astype(int)
        h = np.array(classes)[h] if len(classes) > 2 else h
        rng = np.random.default_rng(R.SEED)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(yy), len(yy))
            if len(set(yy[i])) == len(classes):
                bs.append(balanced_accuracy_score(yy[i], h[i]))
        P = df[[c for c in df.columns if not str(c).startswith("_")]].to_numpy()
        auc = roc_auc_score(yy, P[:, 1]) if len(classes) == 2 else \
            roc_auc_score(yy, P / P.sum(1, keepdims=True), multi_class="ovr")
        s = dict(ba=round(float(balanced_accuracy_score(yy, h)), 4),
                 ba_ci95=[round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)],
                 auroc=round(float(auc), 4),
                 accuracy=round(float((yy == h).mean()), 4),
                 recall=[round(float(r), 4) for r in recall_score(
                     yy, h, labels=classes, average=None, zero_division=0)])
        arch_ba = {}
        for a in sorted(arch.unique()):
            m = arch.reindex(df.index).to_numpy() == a
            if len(set(yy[m])) == len(classes):
                arch_ba[a] = round(float(balanced_accuracy_score(yy[m], h[m])), 4)
        s["archive_ba"] = arch_ba
        m = source.reindex(df.index).to_numpy() == "locked"
        if len(set(yy[m])) == len(classes):
            s["locked_subset_ba"] = round(float(balanced_accuracy_score(yy[m], h[m])), 4)
        return s, pd.DataFrame(dict(y=yy, pred=h), index=df.index)

    for a in ARMS:
        res["arms"][a] = summ(pd.concat(outer[a]))[0]
    s, per = summ(pd.concat(nested))
    res["nested"] = s
    per["arm"] = pd.concat(nested)._arm
    per["target"] = name
    print(name, json.dumps(dict(nested=s["ba"], ci=s["ba_ci95"], chosen=chosen,
                                A0=res["arms"]["A0"]["ba"])), flush=True)
    return res, per.reset_index().rename(columns={"index": "exam_case_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    man, ixa, Fa, source, fold = C.load()
    routes = load_case_routes(ixa)
    cases = set(ixa.exam_case_id)
    for k, X in routes.items():
        assert cases <= set(X.index), (k, len(cases - set(X.index)))
    ctx = dict(Fa=Fa, img_case=ixa.exam_case_id.to_numpy(), routes=routes, cases=cases)
    T = build_targets(man)
    names = a.only or GROUP["measurement"] + GROUP["classification"]
    arch = man.archive
    out = Parallel(n_jobs=a.jobs)(delayed(run_target)(n, T[n], fold, source, arch, ctx)
                                  for n in names)
    os.makedirs(OUT, exist_ok=True)
    tag = "" if not a.only else "_" + "_".join(a.only)
    json.dump(dict(results={r["target"]: r for r, _ in out}, groups=GROUP,
                   protocol=__doc__), open(os.path.join(OUT, "experts%s.json" % tag), "w",
                                           encoding="utf-8"), indent=1, ensure_ascii=False)
    pd.concat([p for _, p in out]).to_csv(os.path.join(OUT, "experts_per_case%s.csv" % tag),
                                          index=False)


if __name__ == "__main__":
    main()
