"""Does patch-level (max / top-k / std) pooling rescue any dead field?

Protocol is copied from eval_fullfov_fields.py so the numbers are comparable:
  - labels are NOT re-derived. truth/fold/baseline are read verbatim from
    threshold_tuned_20260916/*__binary_oof.csv, so binarisation, folds and
    exclusions are byte-identical to AUDIT_all_fields_20260915.
  - baseline = per-fold TRAIN-split majority class, recomputed inside every
    bootstrap resample (a fixed baseline inflates the CI).
  - PCA dim chosen on the inner validation fold (test+1)%5 only.
  - case-level bootstrap, 2000 resamples.
  - verdict: CI lower bound > 0 = usable, upper < 0 = dead, straddling 0 = null.

The one thing that changes is the FEATURE. Specifically we ask whether a
per-dimension max or top-k over 1813 patches beats the global mean for the sparse
localized findings (hemorrhage, microthrombus, rbc_aggregation, crossings), which
is the mechanism that a case-mean of a global token cannot represent.

MULTIPLE COMPARISONS: this scores many pooling combinations per field. The
per-field best is therefore optimistically biased. Both the best AND the
prespecified single candidate (topk_mean) are reported, and only the prespecified
one may be quoted as a result.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The pooling that was decided on before seeing any result, for the sparse-lesion
# hypothesis. Reported separately from the post-hoc best.
PRESPECIFIED = "topk_mean"
POOLINGS = ["cls", "mean", "max", "topk_mean", "std"]
PCA_GRID = (32, 64)
N_BOOT = 2000


def fold_baseline_acc(y, folds, ids):
    """Majority-class accuracy where the majority is taken per fold from its TRAIN split."""
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


def fit_oof(X, y, folds):
    """Out-of-fold predictions; PCA dim selected on the inner val fold only."""
    pred = pd.Series(index=y.index, dtype=float)
    for test_fold in sorted(folds.unique()):
        val_fold = (test_fold + 1) % 5
        te = folds[folds == test_fold].index
        va = folds[folds == val_fold].index
        inner = folds[(folds != test_fold) & (folds != val_fold)].index
        if len(va) == 0 or y.loc[inner].nunique() < 2:
            continue
        best_dim, best_acc = PCA_GRID[0], -1.0
        for dim in PCA_GRID:
            d = min(dim, len(inner) - 1, X.shape[1])
            if d < 2:
                continue
            m = make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                              LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000))
            m.fit(X.loc[inner], y.loc[inner])
            acc = float((m.predict(X.loc[va]) == y.loc[va]).mean())
            if acc > best_acc:
                best_acc, best_dim = acc, dim
        tr = folds[folds != test_fold].index
        if y.loc[tr].nunique() < 2:
            continue
        d = min(best_dim, len(tr) - 1, X.shape[1])
        m = make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000))
        m.fit(X.loc[tr], y.loc[tr])
        pred.loc[te] = m.predict(X.loc[te])
    return pred.dropna()


def scored(y, pred, folds, rng):
    """Accuracy minus per-fold-train majority baseline, with case-level bootstrap CI."""
    ids = pred.index
    acc = float((pred == y.loc[ids]).mean())
    base = fold_baseline_acc(y, folds, ids)
    pos, neg = y.loc[ids] == 1, y.loc[ids] == 0
    sens = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    deltas = np.empty(N_BOOT)
    arr = np.asarray(ids)
    for b in range(N_BOOT):
        s = pd.Index(rng.choice(arr, size=len(arr), replace=True))
        deltas[b] = float((pred.loc[s].values == y.loc[s].values).mean()) - \
            fold_baseline_acc(y, folds, s)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n": int(len(ids)), "accuracy": acc, "baseline": base,
            "delta": acc - base, "delta_ci95": [float(lo), float(hi)],
            "sensitivity": sens, "specificity": spec}


def case_matrix(feats, index, keys):
    """Concatenate the chosen poolings, then average frames within a case."""
    mats = [pd.DataFrame(feats[k].astype(np.float32)) for k in keys]
    wide = pd.concat(mats, axis=1)
    wide.columns = range(wide.shape[1])
    wide["exam_case_id"] = index["exam_case_id"].values
    return wide.groupby("exam_case_id").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", nargs="+", default=["native", "square_baseline"])
    a = ap.parse_args()

    roles = pd.read_csv(a.roles, usecols=["exam_case_id", "evaluation_role"])
    locked = set(roles.loc[roles.evaluation_role.ne("development"), "exam_case_id"])

    # Load every requested preset, verifying no locked case is present.
    banks = {}
    for preset in a.presets:
        d = Path(a.features_root) / preset
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        assert meta["locked_cases_seen"] == 0, "%s reports locked cases seen" % preset
        index = pd.read_csv(d / "index.csv")
        leak = set(index.exam_case_id) & locked
        assert not leak, "%s index contains locked cases: %s" % (preset, sorted(leak))
        feats = {k: np.load(d / ("features_%s.npy" % k)) for k in POOLINGS}
        banks[preset] = (index, feats)
        print("loaded %-16s images=%d cases=%d" % (preset, len(index),
                                                   index.exam_case_id.nunique()), flush=True)

    # Candidate feature views: single poolings plus a few informative pairs.
    combos = [(k,) for k in POOLINGS] + [
        ("mean", "max"), ("mean", "topk_mean"), ("mean", "std"),
        ("cls", "topk_mean"), ("topk_mean", "std"),
    ]

    results = {}
    for oof_path in sorted(Path(a.oof_dir).glob("*__binary_oof.csv")):
        field = oof_path.name.replace("__binary_oof.csv", "")
        table = pd.read_csv(oof_path)
        need = {"case_id", "fold", "truth", "prediction", "baseline"}
        if not need.issubset(table.columns):
            results[field] = {"skipped": "missing columns %s" % sorted(need - set(table.columns))}
            continue
        table = table.drop_duplicates("case_id").set_index("case_id")
        bad = set(table.index) & locked
        assert not bad, "%s OOF contains locked cases: %s" % (field, sorted(bad))

        y = table["truth"].astype(int)
        folds = table["fold"].astype(int)
        if y.nunique() < 2:
            results[field] = {"skipped": "single-class truth"}
            continue

        audit_acc = float((table["prediction"] == table["truth"]).mean())
        entry = {"reference_audit": {
            "accuracy": audit_acc,
            "baseline": fold_baseline_acc(y, folds, y.index),
            "delta": audit_acc - fold_baseline_acc(y, folds, y.index)}}

        for preset, (index, feats) in banks.items():
            for combo in combos:
                cm = case_matrix(feats, index, list(combo))
                ids = cm.index.intersection(y.index)
                if len(ids) < 40:
                    continue
                X, yy, ff = cm.loc[ids], y.loc[ids], folds.loc[ids]
                pred = fit_oof(X, yy, ff)
                if len(pred) < 40:
                    continue
                rng = np.random.default_rng(20260917)
                entry["%s:%s" % (preset, "+".join(combo))] = scored(yy, pred, ff, rng)
        results[field] = entry
        keys = [k for k in entry if k != "reference_audit"]
        if keys:
            best = max(keys, key=lambda k: entry[k]["delta_ci95"][0])
            pre = [k for k in keys if k.endswith(":" + PRESPECIFIED)]
            pre_txt = ""
            if pre:
                p = max(pre, key=lambda k: entry[k]["delta_ci95"][0])
                pre_txt = "  prespec=%+.3f lo=%+.3f" % (entry[p]["delta"], entry[p]["delta_ci95"][0])
            print("%-28s audit=%+.3f best=%+.3f lo=%+.3f %s%s" % (
                field, entry["reference_audit"]["delta"], entry[best]["delta"],
                entry[best]["delta_ci95"][0], best, pre_txt), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": "labels reused verbatim from threshold_tuned OOF; only features swapped",
        "bootstrap": N_BOOT,
        "prespecified_pooling": PRESPECIFIED,
        "locked_cases_used_for_fitting_or_scoring": 0,
        "limitations": [
            "development set only; no locked-47 involvement; not product capability",
            "per-field best is selected post hoc over %d views and is optimistically "
            "biased -- only the prespecified pooling may be quoted" % (len(combos) * len(banks)),
            "frozen encoder: max/topk/std can only surface what DINOv2 already separates",
        ],
        "results": results,
    }, indent=2), encoding="utf-8")
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
