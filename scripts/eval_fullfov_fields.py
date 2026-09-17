"""Does full field of view rescue fields that square-crop features could not do?

Labels are NOT re-derived. They are read verbatim from the frozen OOF files of
threshold_tuned_20260916 (columns: case_id, fold, truth, prediction, baseline),
so the binarisation, the fold assignment and the excluded cases are byte-identical
to the audit. This script only swaps the FEATURES and refits.

Protocol (matches AUDIT_all_fields_20260915):
  - case-level folds, taken from the OOF file's own `fold` column
  - PCA dim chosen on inner validation fold (test+1)%5, never on the test fold
  - LogisticRegression C=0.1, class_weight=balanced, max_iter=4000
  - baseline = per-fold TRAIN majority, recomputed inside every bootstrap resample
  - case-level bootstrap 2000x

square_control is produced by the same extraction script at 518x518 with the same
CenterCrop as the original artifact, so native-vs-square isolates field of view and
nothing else. Comparing straight to the old artifact would confound FOV with
loader/dtype differences.

locked-47: the OOF files contain development cases only; asserted against the
manifest before any fitting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/root/nailfold")
MAN = ROOT / "artifacts/manifest/locked_evaluation_v1_reviewed.csv"
OOF = ROOT / "artifacts/experiments/threshold_tuned_20260916"
PCA_DIMS = (32, 64)
BOOT = 2000


def case_matrix(features, index):
    frame = pd.DataFrame(features.astype(np.float32))
    frame["exam_case_id"] = index.exam_case_id.values
    return frame.groupby("exam_case_id").mean()


def fit_oof(X, y, folds):
    """Out-of-fold predictions. PCA dim picked on the inner validation fold only."""
    pred = pd.Series(index=X.index, dtype=int)
    chosen = {}
    for test_fold in sorted(folds.unique()):
        val_fold = (test_fold + 1) % 5
        train = folds[(folds != test_fold) & (folds != val_fold)].index
        val = folds[folds == val_fold].index
        test = folds[folds == test_fold].index
        if len(val) == 0 or y.loc[train].nunique() < 2:
            train = folds[folds != test_fold].index
            val = train
        best_score, best_dim = -1.0, PCA_DIMS[0]
        for dim in PCA_DIMS:
            eff = int(min(dim, len(train) - 1, X.shape[1]))
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=eff, random_state=0),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000),
            )
            model.fit(X.loc[train], y.loc[train])
            score = float((model.predict(X.loc[val]) == y.loc[val]).mean())
            if score > best_score:
                best_score, best_dim = score, eff
        refit = folds[folds != test_fold].index
        eff = int(min(best_dim, len(refit) - 1, X.shape[1]))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=eff, random_state=0),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000),
        )
        model.fit(X.loc[refit], y.loc[refit])
        pred.loc[test] = model.predict(X.loc[test]).astype(int)
        chosen[int(test_fold)] = eff
    return pred, chosen


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


def scored(y, pred, folds, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(y.index)
    accuracy = float((pred.values == y.values).mean())
    baseline = fold_baseline_acc(y, folds, ids)
    deltas = np.empty(BOOT)
    for i in range(BOOT):
        sample = ids[rng.choice(len(ids), len(ids), replace=True)]
        hit = float((pred.loc[sample].values == y.loc[sample].values).mean())
        deltas[i] = hit - fold_baseline_acc(y, folds, sample)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "baseline": baseline,
        "delta": accuracy - baseline,
        "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def load_feature_sets(names, locked):
    feats = {}
    for name in names:
        directory = ROOT / "artifacts/features_fullfov" / name
        meta = json.loads((directory / "metadata.json").read_text())
        assert meta["locked_cases_seen"] == 0, name
        index = pd.read_csv(directory / "index.csv")
        assert not (set(index.exam_case_id) & locked), f"{name}: locked leak"
        cls = case_matrix(np.load(directory / "features_cls.npy"), index)
        patch = case_matrix(np.load(directory / "features_meanpatch.npy"), index)
        feats[f"{name}:cls"] = cls
        feats[f"{name}:patch"] = patch
        both = pd.concat([cls, patch], axis=1)
        both.columns = range(both.shape[1])
        feats[f"{name}:cls+patch"] = both
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(MAN, low_memory=False)
    locked = set(manifest[manifest.development_fold.isna()].exam_case_id)
    feats = load_feature_sets(args.sets, locked)

    report = {
        "protocol": (
            "labels+folds copied verbatim from threshold_tuned_20260916 OOF files; "
            "features swapped; PCA dim on inner val fold only; baseline = per-fold "
            "train majority recomputed in each bootstrap"
        ),
        "locked_cases_used_for_fitting_or_scoring": 0,
        "bootstrap": BOOT,
        "feature_sets": sorted(feats),
        "results": {},
    }

    for path in sorted(OOF.glob("*__binary_oof.csv")):
        field = path.name.replace("__binary_oof.csv", "")
        table = pd.read_csv(path).rename(columns={"case_id": "exam_case_id"})
        table = table.set_index("exam_case_id")
        assert not (set(table.index) & locked), f"{field}: locked in OOF"
        y = table.truth.astype(int)
        folds = table.fold.astype(int)
        if y.nunique() < 2:
            report["results"][field] = {"skipped": "single-class truth"}
            print(f"{field:34s} SKIP single-class", flush=True)
            continue

        audit_acc = float((table.prediction == table.truth).mean())
        audit_base = float((table.baseline == table.truth).mean())
        entry = {"reference_audit": {"accuracy": audit_acc, "baseline": audit_base,
                                     "delta": audit_acc - audit_base}}
        for name, matrix in feats.items():
            ids = [i for i in y.index if i in matrix.index]
            if len(ids) < len(y) * 0.9:
                entry[name] = {"skipped": f"only {len(ids)}/{len(y)} cases have features"}
                continue
            sub_y, sub_folds = y.loc[ids], folds.loc[ids]
            pred, chosen = fit_oof(matrix.loc[ids], sub_y, sub_folds)
            result = scored(sub_y, pred, sub_folds, seed=abs(hash(field + name)) % (2 ** 31))
            result["pca_dim_per_fold"] = chosen
            entry[name] = result
        report["results"][field] = entry

        usable = [(k, v) for k, v in entry.items() if k != "reference_audit" and "delta" in v]
        if usable:
            name, best = max(usable, key=lambda kv: kv[1]["delta_ci95"][0])
            print(f"{field:34s} audit={audit_acc - audit_base:+.3f}  "
                  f"best={best['delta']:+.3f} lo={best['delta_ci95'][0]:+.3f}  {name}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
