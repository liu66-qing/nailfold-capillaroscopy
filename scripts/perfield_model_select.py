"""Per-field model selection: a different model per field, chosen honestly.

Three things this does differently from every earlier run.

1. CONSTANT PREDICTOR for extreme skew (the user's instruction). When the minority
   share is <= --constant-threshold, the field is delivered as "always answer the
   majority class". That is a design decision, not a model result: delta is exactly
   0.000 by construction, accuracy equals the majority baseline. It is honest, it
   is stable, and it removes the collapse-pretending-to-be-capability problem.

2. SELECTION INSIDE THE FOLD LOOP. Earlier scripts fitted many feature views and
   then reported the best per field, which is optimistically biased. Here the whole
   choice -- feature source, model family, class weighting, decision threshold --
   is made on the inner validation fold only, then refit and applied once to the
   untouched test fold. The reported OOF delta is therefore unbiased with respect
   to that search. This is the main methodological fix.

3. ACCURACY-ALIGNED OBJECTIVE. Every earlier model used
   class_weight="balanced", which maximises balanced accuracy -- but the launch
   ruler is accuracy minus the majority baseline. On skewed fields (rbc_aggregation
   82%, overall_assessment 86.5%) balanced weighting actively sacrifices the
   quantity being measured. Both weightings, and a tuned decision threshold, are
   now candidates and the inner fold decides.

Ruler and protocol are otherwise unchanged: labels reused verbatim from the frozen
threshold_tuned OOF tables, per-fold train-split majority baseline recomputed
inside every bootstrap, case-level bootstrap, locked-47 never touched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BOOT = 2000
SEED = 20260917


# ---------------------------------------------------------------- feature banks
def load_banks(spatial_root, features_root, preset, index_ref):
    """All sources are verified row-aligned to the reference index before use."""
    banks = {}
    d = Path(spatial_root) / preset
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["locked_cases_seen"] == 0, "spatial features report locked cases"
    idx = pd.read_csv(d / "index.csv")
    assert idx.image_path.tolist() == index_ref.image_path.tolist(), "spatial misaligned"
    for k in meta["poolings"]:
        banks["dino_" + k] = np.load(d / ("features_%s.npy" % k)).astype(np.float32)

    for name, sub in [("geom", "geometry_dev"), ("roi", "roi_quality_dev_v1")]:
        p = Path(features_root) / sub
        if not (p / "features.npy").exists():
            continue
        i = pd.read_csv(p / "index.csv")
        if i.image_path.tolist() != index_ref.image_path.tolist():
            print("SKIP %s: index misaligned" % name)
            continue
        banks[name] = np.nan_to_num(np.load(p / "features.npy").astype(np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)

    hul = Path(features_root) / "hulumed_visual" / "vision_mean.npy"
    if hul.exists():
        a = np.load(hul).astype(np.float32)
        if len(a) == len(index_ref):
            banks["hulumed"] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return banks


def case_matrix(banks, index, keys):
    mats = [pd.DataFrame(banks[k]) for k in keys]
    wide = pd.concat(mats, axis=1)
    wide.columns = range(wide.shape[1])
    wide["exam_case_id"] = index["exam_case_id"].values
    return wide.groupby("exam_case_id").mean()


# ---------------------------------------------------------------- candidate space
FEATURE_SETS = [
    ("dino_topk_mean",),
    ("dino_mean",),
    ("dino_max",),
    ("dino_std",),
    ("dino_topk_mean", "dino_std"),
    ("dino_mean", "dino_max"),
    ("geom",),
    ("dino_topk_mean", "geom"),
    ("dino_mean", "geom"),
    ("roi",),
    ("dino_topk_mean", "roi"),
    ("hulumed",),
    ("dino_topk_mean", "hulumed"),
    ("geom", "roi"),
]


def models():
    """Model family x class weighting, as FACTORIES so no estimator instance is
    ever shared between folds or processes. Both weightings are offered because
    the ruler is plain accuracy, not balanced accuracy."""
    out = []
    for cw in [None, "balanced"]:
        for C in [0.03, 0.1, 1.0]:
            for dim in [32, 64]:
                out.append((
                    "logreg_C%g_d%d_%s" % (C, dim, cw), "pca", dim,
                    (lambda C=C, cw=cw: LogisticRegression(
                        C=C, class_weight=cw, max_iter=4000))))
    for cw in [None, "balanced"]:
        out.append((
            "hgb_%s" % cw, "raw", None,
            (lambda cw=cw: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                min_samples_leaf=10, l2_regularization=1.0,
                class_weight=cw, random_state=0))))
        out.append((
            "rf_%s" % cw, "raw", None,
            (lambda cw=cw: RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                class_weight=cw, random_state=0, n_jobs=1))))
    return out


def build(kind, dim, make_est, n_train, n_feat):
    """make_est is a factory; a fresh estimator is created for every fit."""
    if kind == "pca":
        d = min(dim, n_train - 1, n_feat)
        if d < 2:
            return None
        return make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                             make_est())
    return make_pipeline(StandardScaler(), make_est())


def best_threshold(y, p):
    """Threshold maximising PLAIN accuracy (the ruler), not balanced accuracy."""
    cand = np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.05, 0.95, 19))]))
    best, best_acc = 0.5, -1.0
    for t in cand:
        acc = float(((p >= t).astype(int) == y).mean())
        if acc > best_acc:
            best_acc, best = acc, float(t)
    return best, best_acc


# ---------------------------------------------------------------- baseline & score
def fold_baseline_acc(y, folds, ids):
    """Majority-class accuracy, majority taken per fold from its TRAIN split."""
    total = 0.0
    for test_fold in sorted(folds.unique()):
        train = folds[folds != test_fold].index
        mode = y.loc[train].mode()
        if len(mode) == 0:
            continue
        mask = folds.loc[ids].values == test_fold
        if mask.sum():
            total += float((y.loc[ids].values[mask] == mode.iloc[0]).mean()) * mask.sum()
    return total / len(ids)


def score(y, pred, folds, rng):
    """Two baselines, both reported, because they differ materially on near-balanced
    fields:

      per_fold  majority taken per fold from its TRAIN split. On a 50/50 field the
                train majority can flip between folds and score BELOW chance, which
                depresses the baseline and inflates delta.
      constant  max(prevalence, 1-prevalence): the single fixed answer a no-model
                system would actually ship. This is the stricter, more honest bar
                and is what `delta_vs_constant` is measured against.

    A field only counts as beating "do nothing" if delta_vs_constant CI lower
    bound > 0.
    """
    ids = pred.index
    acc = float((pred == y.loc[ids]).mean())
    base_fold = fold_baseline_acc(y, folds, ids)
    yv = y.loc[ids]
    base_const = float(max(yv.mean(), 1 - yv.mean()))
    pos, neg = yv == 1, yv == 0
    arr = np.asarray(ids)
    d_fold = np.empty(N_BOOT)
    d_const = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = pd.Index(rng.choice(arr, size=len(arr), replace=True))
        a = float((pred.loc[s].values == y.loc[s].values).mean())
        ys = y.loc[s]
        d_fold[b] = a - fold_baseline_acc(y, folds, s)
        d_const[b] = a - max(ys.mean(), 1 - ys.mean())
    lo, hi = np.percentile(d_fold, [2.5, 97.5])
    clo, chi = np.percentile(d_const, [2.5, 97.5])
    return {"n": int(len(ids)), "accuracy": acc,
            "baseline": base_fold, "delta": acc - base_fold,
            "delta_ci95": [float(lo), float(hi)],
            "baseline_constant": base_const,
            "delta_vs_constant": acc - base_const,
            "delta_vs_constant_ci95": [float(clo), float(chi)],
            "sensitivity": float((pred[pos] == 1).mean()) if pos.any() else None,
            "specificity": float((pred[neg] == 0).mean()) if neg.any() else None,
            "predictions": {str(k): int(v) for k, v in pred.items()}}


# ---------------------------------------------------------------- nested per fold
def run_fold(test_fold, y, folds, banks, index, tune_threshold):
    """Choose feature set + model + threshold on the INNER val fold, then refit on
    all non-test folds and predict the test fold once."""
    val_fold = (test_fold + 1) % 5
    te = folds[folds == test_fold].index
    va = folds[folds == val_fold].index
    inner = folds[(folds != test_fold) & (folds != val_fold)].index
    if len(va) == 0 or len(te) == 0 or y.loc[inner].nunique() < 2 or y.loc[va].nunique() < 2:
        return None

    best = None
    for fs in FEATURE_SETS:
        if not all(k in banks for k in fs):
            continue
        cm = case_matrix(banks, index, list(fs))
        ids = cm.index.intersection(y.index)
        Xi, Xv = cm.loc[cm.index.intersection(inner)], cm.loc[cm.index.intersection(va)]
        if len(Xi) < 20 or len(Xv) < 5:
            continue
        yi, yv = y.loc[Xi.index], y.loc[Xv.index]
        if yi.nunique() < 2:
            continue
        for mname, kind, dim, make_est in models():
            m = build(kind, dim, make_est, len(Xi), Xi.shape[1])
            if m is None:
                continue
            try:
                m.fit(Xi, yi)
                pv = m.predict_proba(Xv)[:, 1]
            except Exception:
                continue
            thr, acc = (best_threshold(yv.values, pv) if tune_threshold
                        else (0.5, float(((pv >= 0.5).astype(int) == yv.values).mean())))
            if best is None or acc > best["acc"]:
                best = {"acc": acc, "fs": fs, "model": mname, "kind": kind,
                        "dim": dim, "thr": thr}
    if best is None:
        return None

    cm = case_matrix(banks, index, list(best["fs"]))
    tr = cm.index.intersection(folds[folds != test_fold].index)
    tei = cm.index.intersection(te)
    if len(tei) == 0 or y.loc[tr].nunique() < 2:
        return None
    make_est = {n: f for n, _, _, f in models()}[best["model"]]
    m = build(best["kind"], best["dim"], make_est, len(tr), cm.shape[1])
    m.fit(cm.loc[tr], y.loc[tr])
    p = m.predict_proba(cm.loc[tei])[:, 1]
    pred = pd.Series((p >= best["thr"]).astype(int), index=tei)
    return {"pred": pred, "choice": {"fold": int(test_fold), "features": list(best["fs"]),
                                     "model": best["model"], "threshold": best["thr"],
                                     "inner_val_acc": best["acc"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial-root", required=True)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--preset", default="native")
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--constant-threshold", type=float, default=0.08,
                    help="minority share at or below which the field is delivered "
                         "as a constant majority-class answer")
    ap.add_argument("--no-threshold-tuning", action="store_true")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--fields", nargs="*", default=None)
    a = ap.parse_args()

    roles = pd.read_csv(a.roles, usecols=["exam_case_id", "evaluation_role"])
    locked = set(roles.loc[roles.evaluation_role.ne("development"), "exam_case_id"])

    index = pd.read_csv(Path(a.spatial_root) / a.preset / "index.csv")
    assert not (set(index.exam_case_id) & locked), "locked case in feature index"
    banks = load_banks(a.spatial_root, a.features_root, a.preset, index)
    print("feature banks: %s" % ", ".join("%s(%d)" % (k, v.shape[1])
                                          for k, v in sorted(banks.items())), flush=True)

    results = {}
    for path in sorted(Path(a.oof_dir).glob("*__binary_oof.csv")):
        field = path.name.replace("__binary_oof.csv", "")
        if a.fields and field not in a.fields:
            continue
        t = pd.read_csv(path).drop_duplicates("case_id").set_index("case_id")
        assert not (set(t.index) & locked), "%s OOF has locked cases" % field
        y = t["truth"].astype(int)
        folds = t["fold"].astype(int)
        if y.nunique() < 2:
            results[field] = {"skipped": "single-class truth"}
            continue

        prev = float(y.mean())
        minority = min(prev, 1 - prev)
        maj = int(y.mode().iloc[0])

        # --- constant predictor for extreme skew (user's instruction) -----------
        if minority <= a.constant_threshold:
            base = fold_baseline_acc(y, folds, y.index)
            results[field] = {
                "delivery": "constant_majority",
                "constant_answer": maj,
                "n": int(len(y)), "positives": int(y.sum()), "prevalence": prev,
                "minority_share": minority,
                "accuracy": float(max(prev, 1 - prev)), "baseline": base,
                "baseline_constant": float(max(prev, 1 - prev)),
                "delta": 0.0, "delta_ci95": [0.0, 0.0],
                "delta_vs_constant": 0.0, "delta_vs_constant_ci95": [0.0, 0.0],
                "sensitivity": 0.0 if maj == 0 else 1.0,
                "specificity": 1.0 if maj == 0 else 0.0,
                "rationale": (
                    "minority share %.3f <= %.3f; a perfect classifier could gain at "
                    "most %+.3f, below the 0.08 seed-noise band. Delivered as a fixed "
                    "majority-class answer by design. delta is 0 by construction, not "
                    "a measured capability." % (minority, a.constant_threshold, minority)),
            }
            print("%-28s CONSTANT  answer=%d  acc=%.3f (=baseline)  minority=%.3f" % (
                field, maj, base, minority), flush=True)
            continue

        # --- nested per-field model selection ---------------------------------
        outs = Parallel(n_jobs=a.jobs, backend="loky")(
            delayed(run_fold)(f, y, folds, banks, index, not a.no_threshold_tuning)
            for f in sorted(folds.unique()))
        outs = [o for o in outs if o is not None]
        if not outs:
            results[field] = {"skipped": "no fold produced a model"}
            continue
        pred = pd.concat([o["pred"] for o in outs]).sort_index()
        pred = pred[~pred.index.duplicated()]
        common = pred.index.intersection(y.index)
        if len(common) < 40:
            results[field] = {"skipped": "too few predicted cases (%d)" % len(common)}
            continue
        rng = np.random.default_rng(SEED)
        s = score(y, pred.loc[common], folds, rng)
        s.update({"delivery": "model", "positives": int(y.sum()), "prevalence": prev,
                  "minority_share": minority,
                  "per_fold_choices": [o["choice"] for o in outs]})
        results[field] = s
        picks = {}
        for o in outs:
            key = "+".join(o["choice"]["features"]) + " | " + o["choice"]["model"]
            picks[key] = picks.get(key, 0) + 1
        top = max(picks.items(), key=lambda kv: kv[1])
        print("%-28s MODEL  acc=%.3f  vs_fold=%+.3f[%+.3f]  vs_const=%+.3f[%+.3f]  "
              "%dx %s" % (field, s["accuracy"], s["delta"], s["delta_ci95"][0],
                          s["delta_vs_constant"], s["delta_vs_constant_ci95"][0],
                          top[1], top[0]), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": ("per-field nested selection: feature set, model family, class "
                     "weighting and decision threshold all chosen on the inner "
                     "validation fold (test+1)%5 only, then refit on all non-test "
                     "folds and applied once to the untouched test fold"),
        "constant_threshold": a.constant_threshold,
        "threshold_tuning": not a.no_threshold_tuning,
        "bootstrap": N_BOOT,
        "feature_banks": {k: int(v.shape[1]) for k, v in sorted(banks.items())},
        "candidate_space": {"feature_sets": len(FEATURE_SETS), "models": len(models())},
        "locked_cases_used": 0,
        "limitations": [
            "development set only, out-of-fold; NOT product capability",
            "two baselines are reported: per-fold train majority (which can flip "
            "class between folds on near-balanced fields and thus score below "
            "chance, inflating delta) and the constant majority answer a no-model "
            "system would actually ship. delta_vs_constant is the stricter bar and "
            "is the one that decides whether a field beats doing nothing",
            "constant_majority fields have delta 0 BY CONSTRUCTION -- that is a "
            "delivery decision, not a measurement, and must never be quoted as skill",
            "nested selection removes the bias from choosing per field, but the "
            "inner fold is small (~37 cases) so the choice itself is noisy",
            "class_weight=None candidates optimise plain accuracy, which is the "
            "ruler; they will look worse on balanced accuracy and sensitivity",
            "threshold tuned on the inner fold only; still optimistic vs a truly "
            "held-out threshold",
        ],
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
