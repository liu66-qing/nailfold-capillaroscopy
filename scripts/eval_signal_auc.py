"""Is there ANY signal in the dead fields? Discrimination, measured separately
from the accuracy-vs-majority verdict.

Why a second ruler, and why this is not metric shopping:

  accuracy - majority baseline  answers "can this field be delivered as a LABEL".
                                It stays the launch ruler. Nothing here replaces it.
  AUC                           answers "does the feature carry ANY ordering
                                information about this field". It is base-rate
                                independent, so a field with 6% prevalence can show
                                real discrimination while being unable to beat a
                                0.94 majority baseline.

These answer different questions and both are reported. A field with AUC clearly
above 0.5 but delta ~ 0 has signal that cannot survive being forced into a hard
label at this prevalence. That is a DELIVERY FORM finding (rank for review rather
than assert a label), not a capability upgrade, and it is labelled as such below.

Explicitly NOT claimed here:
  - AUC > 0.5 does not mean a field can launch.
  - AUC is computed out-of-fold on the development set. It is not product capability.
  - a field that only reaches significance on one of many feature views is a
    multiple-comparison artifact; the prespecified view is reported alongside.

Also reports NPV at the threshold achieving >= 0.90 sensitivity, because rule-out
is the only positioning the deployment analysis found defensible, and NPV is what
that positioning actually rests on.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRESPEC = "topk_mean"
PCA_GRID = (32, 64)
N_BOOT = 2000
TARGET_SENS = 0.90


def fit_oof_proba(X, y, folds):
    """Out-of-fold positive-class probability; PCA dim chosen on inner val fold only."""
    out = pd.Series(index=y.index, dtype=float)
    for test_fold in sorted(folds.unique()):
        val_fold = (test_fold + 1) % 5
        te = folds[folds == test_fold].index
        va = folds[folds == val_fold].index
        inner = folds[(folds != test_fold) & (folds != val_fold)].index
        if len(va) == 0 or y.loc[inner].nunique() < 2:
            continue
        best_dim, best = PCA_GRID[0], -1.0
        for dim in PCA_GRID:
            d = min(dim, len(inner) - 1, X.shape[1])
            if d < 2:
                continue
            m = make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                              LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000))
            m.fit(X.loc[inner], y.loc[inner])
            if y.loc[va].nunique() < 2:
                continue
            s = roc_auc_score(y.loc[va], m.predict_proba(X.loc[va])[:, 1])
            if s > best:
                best, best_dim = s, dim
        tr = folds[folds != test_fold].index
        if y.loc[tr].nunique() < 2:
            continue
        d = min(best_dim, len(tr) - 1, X.shape[1])
        m = make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000))
        m.fit(X.loc[tr], y.loc[tr])
        out.loc[te] = m.predict_proba(X.loc[te])[:, 1]
    return out.dropna()


def npv_at_sensitivity(y, p, target=TARGET_SENS):
    """Sweep thresholds, take the one with the highest NPV that still reaches
    `target` sensitivity. Returns None if no threshold reaches it."""
    order = np.argsort(-p.values)
    ys = y.loc[p.index].values[order]
    ps = p.values[order]
    pos_total = ys.sum()
    if pos_total == 0:
        return None
    best = None
    tp = fp = 0
    for i in range(len(ys)):
        if ys[i] == 1:
            tp += 1
        else:
            fp += 1
        sens = tp / pos_total
        if sens < target:
            continue
        # everything after i is predicted negative
        fn = pos_total - tp
        tn = (len(ys) - i - 1) - fn
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        cand = {"threshold": float(ps[i]), "sensitivity": float(sens),
                "specificity": float(spec), "npv": float(npv),
                "flagged_fraction": float((i + 1) / len(ys))}
        if best is None or (npv == npv and npv > best["npv"]):
            best = cand
    return best


def auc_ci(y, p, rng):
    yv, pv = y.loc[p.index].values, p.values
    boots = np.empty(N_BOOT)
    n = len(yv)
    filled = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(yv[idx])) < 2:
            continue
        boots[filled] = roc_auc_score(yv[idx], pv[idx])
        filled += 1
    if filled < 100:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots[:filled], [2.5, 97.5])
    return float(lo), float(hi)


def case_matrix(feats, index, keys):
    mats = [pd.DataFrame(feats[k].astype(np.float32)) for k in keys]
    wide = pd.concat(mats, axis=1)
    wide.columns = range(wide.shape[1])
    wide["exam_case_id"] = index["exam_case_id"].values
    return wide.groupby("exam_case_id").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--preset", default="native")
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    roles = pd.read_csv(a.roles, usecols=["exam_case_id", "evaluation_role"])
    locked = set(roles.loc[roles.evaluation_role.ne("development"), "exam_case_id"])

    d = Path(a.features_root) / a.preset
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["locked_cases_seen"] == 0, "feature set reports locked cases seen"
    index = pd.read_csv(d / "index.csv")
    assert not (set(index.exam_case_id) & locked), "locked case in feature index"
    feats = {k: np.load(d / ("features_%s.npy" % k)) for k in meta["poolings"]}

    views = {PRESPEC: [PRESPEC], "mean": ["mean"], "max": ["max"], "std": ["std"],
             "topk_mean+std": ["topk_mean", "std"]}

    results = {}
    print("%-28s %5s %6s %7s %16s %8s %8s %7s" % (
        "field", "pos", "prev", "auc", "auc_ci95", "npv@90", "spec", "view"))
    for path in sorted(Path(a.oof_dir).glob("*__binary_oof.csv")):
        field = path.name.replace("__binary_oof.csv", "")
        t = pd.read_csv(path).drop_duplicates("case_id").set_index("case_id")
        assert not (set(t.index) & locked), "%s OOF has locked cases" % field
        y = t["truth"].astype(int)
        folds = t["fold"].astype(int)
        if y.nunique() < 2:
            results[field] = {"skipped": "single-class truth"}
            continue

        entry = {"positives": int(y.sum()), "n": int(len(y)),
                 "prevalence": float(y.mean()), "views": {}}
        for vname, keys in views.items():
            cm = case_matrix(feats, index, keys)
            ids = cm.index.intersection(y.index)
            if len(ids) < 40:
                continue
            p = fit_oof_proba(cm.loc[ids], y.loc[ids], folds.loc[ids])
            if len(p) < 40 or y.loc[p.index].nunique() < 2:
                continue
            rng = np.random.default_rng(20260917)
            auc = float(roc_auc_score(y.loc[p.index], p))
            lo, hi = auc_ci(y, p, rng)
            entry["views"][vname] = {
                "auc": auc, "auc_ci95": [lo, hi],
                "significant": bool(lo == lo and lo > 0.5),
                "rule_out": npv_at_sensitivity(y, p),
            }
        results[field] = entry
        if entry["views"]:
            pre = entry["views"].get(PRESPEC) or list(entry["views"].values())[0]
            vn = PRESPEC if PRESPEC in entry["views"] else list(entry["views"])[0]
            ro = pre["rule_out"] or {}
            print("%-28s %5d %6.3f %7.3f  [%+.3f,%+.3f] %8s %8s %7s" % (
                field, entry["positives"], entry["prevalence"], pre["auc"],
                pre["auc_ci95"][0], pre["auc_ci95"][1],
                ("%.3f" % ro["npv"]) if ro.get("npv") == ro.get("npv") else "-",
                ("%.3f" % ro["specificity"]) if ro.get("specificity") == ro.get("specificity") else "-",
                vn), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "preset": a.preset,
        "prespecified_view": PRESPEC,
        "bootstrap": N_BOOT,
        "target_sensitivity": TARGET_SENS,
        "locked_cases_used": 0,
        "interpretation": (
            "AUC answers whether any ordering signal exists; it does NOT replace "
            "accuracy-minus-majority-baseline as the launch ruler and does NOT mean "
            "a field can be delivered as a label."),
        "limitations": [
            "development set, out-of-fold; not product capability",
            "AUC is base-rate independent, so a significant AUC at 2% prevalence "
            "still yields an unusable hard label",
            "NPV@90%% sensitivity is computed at a threshold chosen on the same OOF "
            "predictions, so it is optimistic; a held-out threshold would be lower",
            "5 views per field; per-field maxima are optimistically biased",
        ],
        "results": results,
    }, indent=2), encoding="utf-8")
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
