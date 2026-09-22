#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired evaluation of the medical-encoder arms against the deployed anchor.

Everything scoreable is fixed by protocol.yaml before this file was run: the arms,
the endpoints, the head, the C grid, the folds, and the gates. This script chooses
nothing except C, and it chooses that inside training folds only.

What it reports per field and arm:
  accuracy, delta vs the single-constant-answer baseline, balanced accuracy,
  per-class recall, AUROC, miss rate, false-alarm rate, and -- the number the gates
  are written against -- the PAIRED per-case difference against the anchor with a
  case-level bootstrap CI.

Paired matters. Two arms evaluated on the same 186 cases share all the case-level
noise, so an unpaired comparison of two wide CIs hides a difference that a paired
test can resolve, and vice versa: an arm can look better on point estimate while
losing on more cases than it wins.

Also runs leave-one-archive-out with C selected inside the retained archives, so the
held-out archive takes no part in selection.

DEVELOPMENT ONLY. locked-47 is never loaded.

  PYTHONIOENCODING=utf-8 python scripts/eval_medical_encoders.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"

SEED, N_BOOT = 20260921, 2000
C_GRID = [0.003, 0.03, 0.3]          # the only tuned parameter
POOLINGS = ["cls", "mean", "max", "topk_mean", "std"]
READOUT = "cls"                       # protocol round one: single global vector
ANCHOR = "anchor_dinov2b_deployed"

PRIMARY = "malformation_ratio"
SECONDARY = "papilla"
RETAINED = ["clarity", "exudation", "subpapillary_venous_plexus", "blood_color"]
EXPLORATORY = ["rbc_aggregation"]

# identical to the shipped configuration's mapping, so numbers stay comparable
BINARY_MAP = {
    "malformation_ratio": {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]},
    "clarity": {0: ["清晰"], 1: ["不清", "模糊"]},
    "exudation": {0: ["无"], 1: ["+", "++", "+++"]},
    "subpapillary_venous_plexus": {0: ["不见"],
                                   1: ["可见1排", "可见2排", ">2排,扩张"]},
    "blood_color": {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]},
    "rbc_aggregation": {0: ["无", "[无]"], 1: ["轻度", "中度", "重度"]},
}
MULTICLASS = {"papilla": ["波纹状", "浅波纹状", "平坦"]}
FIELDS = list(BINARY_MAP) + list(MULTICLASS)


def load_arm(arm: str):
    d = EXP / "features" / arm
    ix = pd.read_csv(d / "index.csv", dtype={"exam_case_id": str})
    feats = {p: np.load(d / f"features_{p}.npy").astype(np.float32)
             for p in POOLINGS}
    return ix, feats


def case_labels(man: pd.DataFrame, field: str):
    """Case-level truth, dirty values dropped rather than coerced to normal."""
    out = {}
    if field in BINARY_MAP:
        for cid, raw in man[field].items():
            s = str(raw).strip()
            for lab, vals in BINARY_MAP[field].items():
                if s in vals:
                    out[cid] = float(lab)
    else:
        order = MULTICLASS[field]
        for cid, raw in man[field].items():
            s = str(raw).strip()
            if s in order:
                out[cid] = float(order.index(s))
    return pd.Series(out, dtype=float)


def fit_one(Xtr, ytr, Xts, C, multiclass):
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(ytr[0]))
    dim = max(2, min(64, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return (pipe.predict(Xts).astype(float) if multiclass
            else pipe.predict_proba(Xts)[:, 1])


def case_scores(X, ix, mask, y_case, tr_cases, te_cases, C, multiclass):
    """Fit on training-fold images, then average predictions within a case."""
    trm = mask & ix.exam_case_id.isin(tr_cases).to_numpy()
    tem = mask & ix.exam_case_id.isin(te_cases).to_numpy()
    if trm.sum() == 0 or tem.sum() == 0:
        return pd.Series(dtype=float)
    ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
    pred = fit_one(X[trm], ytr, X[tem], C, multiclass)
    s = pd.Series(pred, index=ix.exam_case_id[tem].to_numpy())
    if multiclass:                      # majority vote of image-level classes
        return s.groupby(level=0).agg(
            lambda g: float(pd.Series(g).value_counts().idxmax()))
    return s.groupby(level=0).mean()
def nested_oof(X, ix, y_case, folds, multiclass):
    """Outer OOF with C selected on an inner split of the training folds only."""
    order = sorted(y_case.index)
    mask = ix.exam_case_id.isin(y_case.index).to_numpy()
    out = pd.Series(np.nan, index=order, dtype=float)
    chosen = {}
    fser = pd.Series({c: folds.get(c, np.nan) for c in order})
    outer = sorted(fser.dropna().unique())
    for fo in outer:
        te = set(fser.index[fser == fo])
        tr = sorted(set(order) - te)
        inner_folds = [f for f in outer if f != fo]
        best, best_score = C_GRID[0], -np.inf
        for C in C_GRID:
            accs = []
            for fi in inner_folds:                      # inner CV inside train
                ite = set(fser.index[fser == fi]) & set(tr)
                itr = sorted(set(tr) - ite)
                if not ite or not itr:
                    continue
                s = case_scores(X, ix, mask, y_case, itr, ite, C, multiclass)
                if s.empty:
                    continue
                yt = y_case.reindex(s.index).to_numpy()
                yp = s.to_numpy() if multiclass else (s.to_numpy() >= 0.5)
                accs.append(float((yp.astype(float) == yt).mean()))
            if accs and np.mean(accs) > best_score:
                best_score, best = float(np.mean(accs)), C
        chosen[float(fo)] = best
        s = case_scores(X, ix, mask, y_case, tr, sorted(te), best, multiclass)
        out.loc[s.index] = s.to_numpy()
    return out, chosen


def metrics(y, p, multiclass):
    """Everything the protocol asks for; unestimable quantities are marked."""
    yp = p if multiclass else (p >= 0.5).astype(float)
    vals, cnt = np.unique(y, return_counts=True)
    const = float(vals[cnt.argmax()])           # single-constant-answer baseline
    acc = float((yp == y).mean())
    base = float((y == const).mean())
    rec = {str(int(c)): (round(float((yp[y == c] == c).mean()), 4)
                         if (y == c).sum() else None) for c in vals}
    auroc = None
    if not multiclass and len(vals) == 2:
        auroc = round(float(roc_auc_score(y, p)), 4)
    fn = fp = miss = fa = None
    if len(vals) == 2:
        pos, neg = vals.max(), vals.min()
        fn = int(((y == pos) & (yp == neg)).sum())
        fp = int(((y == neg) & (yp == pos)).sum())
        miss = round(fn / max(1, int((y == pos).sum())), 4)
        fa = round(fp / max(1, int((y == neg).sum())), 4)
    return dict(n=int(len(y)), accuracy=round(acc, 4),
                baseline_constant=round(base, 4), delta=round(acc - base, 4),
                balanced_accuracy=round(float(balanced_accuracy_score(y, yp)), 4),
                per_class_recall=rec, auroc=auroc, fn=fn, fp=fp,
                miss_rate=miss, false_alarm_rate=fa,
                n_classes_present=int(len(vals)),
                collapsed_to_one_class=bool(len(np.unique(yp)) == 1),
                auroc_unestimable=bool(auroc is None))


def paired(y, p_arm, p_anchor, multiclass, rng):
    """Case-level paired bootstrap of the BA difference, arm minus anchor."""
    def ba(pp, idx):
        yy = y[idx]
        pv = pp[idx] if multiclass else (pp[idx] >= 0.5).astype(float)
        if len(np.unique(yy)) < 2:
            return np.nan
        return float(balanced_accuracy_score(yy, pv))

    full = np.arange(len(y))
    obs = ba(p_arm, full) - ba(p_anchor, full)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d[b] = ba(p_arm, i) - ba(p_anchor, i)
    d = d[~np.isnan(d)]
    lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
    ya = p_arm if multiclass else (p_arm >= 0.5).astype(float)
    yb = p_anchor if multiclass else (p_anchor >= 0.5).astype(float)
    return dict(ba_gain=round(float(obs), 4),
                ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4),
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                cases_arm_right_anchor_wrong=int(((ya == y) & (yb != y)).sum()),
                cases_anchor_right_arm_wrong=int(((yb == y) & (ya != y)).sum()))
def loao(X, ix, y_case, arch, multiclass):
    """Leave-one-archive-out; the held-out archive takes no part in choosing C."""
    order = sorted(y_case.index)
    mask = ix.exam_case_id.isin(y_case.index).to_numpy()
    res = {}
    for held in sorted(set(arch.reindex(order).dropna())):
        te = [c for c in order if arch.get(c) == held]
        tr = [c for c in order if arch.get(c) != held]
        if len(te) < 15 or len(tr) < 30:
            res[str(held)] = dict(skipped="too few cases")
            continue
        inner = pd.Series({c: arch.get(c) for c in tr})
        best, best_score = C_GRID[0], -np.inf
        for C in C_GRID:                      # select within retained archives
            accs = []
            for ha in sorted(set(inner.dropna())):
                ite = [c for c in tr if arch.get(c) == ha]
                itr = [c for c in tr if arch.get(c) != ha]
                if len(ite) < 10 or len(itr) < 20:
                    continue
                s = case_scores(X, ix, mask, y_case, itr, ite, C, multiclass)
                if s.empty:
                    continue
                yt = y_case.reindex(s.index).to_numpy()
                yp = s.to_numpy() if multiclass else (s.to_numpy() >= 0.5)
                accs.append(float((yp.astype(float) == yt).mean()))
            if accs and np.mean(accs) > best_score:
                best_score, best = float(np.mean(accs)), C
        s = case_scores(X, ix, mask, y_case, tr, te, best, multiclass)
        if s.empty:
            res[str(held)] = dict(skipped="no predictions")
            continue
        yt = y_case.reindex(s.index).to_numpy()
        m = metrics(yt, s.to_numpy(), multiclass)
        res[str(held)] = dict(C=best, n=m["n"], balanced_accuracy=m["balanced_accuracy"],
                              delta=m["delta"], auroc=m["auroc"])
    return res


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    folds = dev.development_fold.to_dict()
    arch = dev.archive
    locked = set(man.index[man.development_fold.isna()])

    arms = sorted(p.name for p in (EXP / "features").iterdir() if p.is_dir())
    if ANCHOR not in arms:
        raise RuntimeError("anchor features missing")
    rng = np.random.default_rng(SEED)

    data, per_arm = {}, {}
    for arm in arms:
        ix, feats = load_arm(arm)
        if set(ix.exam_case_id) & locked:
            raise RuntimeError("locked case in %s features" % arm)
        data[arm] = (ix, feats[READOUT])

    rows, oof_records = [], []
    for field in FIELDS:
        multiclass = field in MULTICLASS
        y_case = case_labels(dev, field)
        if y_case.nunique() < 2 or len(y_case) < 30:
            continue
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy()
        preds = {}
        for arm, (ix, X) in data.items():
            p, chosen = nested_oof(X, ix, y_case, folds, multiclass)
            p = p.reindex(order)
            preds[arm] = p.to_numpy()
            m = metrics(y[~p.isna().to_numpy()], p.dropna().to_numpy(), multiclass)
            rec = dict(field=field, arm=arm, readout=READOUT,
                       kind="multiclass" if multiclass else "binary",
                       C_per_fold=chosen, **m)
            per_arm[(field, arm)] = rec
            rows.append(rec)
            for cid, v in p.items():
                oof_records.append(dict(field=field, arm=arm, exam_case_id=cid,
                                        y_true=y_case[cid], pred=v,
                                        fold=folds.get(cid),
                                        archive=arch.get(cid)))
        base = preds[ANCHOR]
        ok = ~(np.isnan(base) | np.isnan(y.astype(float)))
        for arm in arms:
            if arm == ANCHOR:
                continue
            okk = ok & ~np.isnan(preds[arm])
            pr = paired(y[okk], preds[arm][okk], base[okk], multiclass, rng)
            per_arm[(field, arm)]["paired_vs_anchor"] = pr
        print("  %-28s done (%s)" % (field, "multiclass" if multiclass else "binary"),
              flush=True)

    # LOAO only for the two endpoints the gates are written against
    loao_out = {}
    for field in [PRIMARY, SECONDARY]:
        multiclass = field in MULTICLASS
        y_case = case_labels(dev, field)
        loao_out[field] = {arm: loao(data[arm][1], data[arm][0], y_case, arch,
                                     multiclass) for arm in arms}
        print("  LOAO %-24s done" % field, flush=True)

    EXP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof_records).to_csv(EXP / "predictions_oof.csv", index=False,
                                     encoding="utf-8-sig")
    summary = pd.DataFrame([
        dict(field=r["field"], arm=r["arm"], n=r["n"], acc=r["accuracy"],
             delta=r["delta"], ba=r["balanced_accuracy"], auroc=r["auroc"],
             miss_rate=r["miss_rate"], collapsed=r["collapsed_to_one_class"],
             ba_gain=(r.get("paired_vs_anchor") or {}).get("ba_gain"),
             ci_lo=(r.get("paired_vs_anchor") or {}).get("ci95_lo"),
             ci_hi=(r.get("paired_vs_anchor") or {}).get("ci95_hi"))
        for r in rows])
    summary.to_csv(EXP / "paired_summary.csv", index=False, encoding="utf-8-sig")
    (EXP / "paired_metrics.json").write_text(json.dumps(dict(
        readout=READOUT, seed=SEED, n_boot=N_BOOT, C_grid=C_GRID,
        anchor=ANCHOR, arms=arms,
        governance=dict(locked_cases_seen=0, n_development_cases=int(len(dev))),
        per_field=[dict(r) for r in rows], loao=loao_out,
    ), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n== primary endpoint: %s ==" % PRIMARY)
    print(summary[summary.field == PRIMARY].to_string(index=False))
    print("\n== all fields x arms ==")
    print(summary.to_string(index=False))
    print("\nwrote %s" % (EXP / "paired_metrics.json"))


if __name__ == "__main__":
    main()
