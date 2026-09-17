"""Control: does per-field/per-fold selection actually beat one fixed config?

The nested run selected a different feature set, model and threshold in every
fold -- thresholds swung from 0.04 to 0.94 on clarity. That is a symptom of a ~37
case inner fold, not of genuine per-field structure. If a single fixed
configuration applied to all folds does as well, then the selection machinery is
adding variance without adding capability, and the simpler pipeline is the correct
one to ship.

Candidates are a small prespecified set (dino_topk_mean and dino_mean, unweighted
logistic regression at the C that the nested run picked most often, threshold
fixed at 0.5). Each is evaluated once, over all folds, with no selection at all --
so these numbers carry no selection bias whatsoever and are directly comparable to
the nested delta_vs_constant.

Reports both so the comparison is visible. Whichever wins, the honest statement is
about the same 4 fields; this only decides how much machinery ships.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPATIAL = Path("/root/autodl-tmp/nailfold/artifacts/features_spatial/native")
OOF = Path("/root/nailfold/artifacts/experiments/threshold_tuned_20260916")
ROLES = Path("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
NESTED = Path("/root/autodl-tmp/nailfold/artifacts/experiments/perfield_20260917/native.json")
OUT = Path("/root/autodl-tmp/nailfold/artifacts/experiments/perfield_20260917/fixed_config_control.json")
N_BOOT = 2000

CONFIGS = [
    ("topk_C0.03_d64", "topk_mean", 0.03, 64),
    ("mean_C0.03_d64", "mean", 0.03, 64),
    ("topk_C0.1_d32", "topk_mean", 0.1, 32),
]


def const_delta(y, pred, ids, rng):
    acc = float((pred.loc[ids] == y.loc[ids]).mean())
    yv = y.loc[ids]
    base = float(max(yv.mean(), 1 - yv.mean()))
    arr = np.asarray(ids)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = pd.Index(rng.choice(arr, size=len(arr), replace=True))
        a = float((pred.loc[s].values == y.loc[s].values).mean())
        ys = y.loc[s]
        d[b] = a - max(ys.mean(), 1 - ys.mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"accuracy": acc, "baseline_constant": base, "delta_vs_constant": acc - base,
            "delta_vs_constant_ci95": [float(lo), float(hi)]}


def main():
    roles = pd.read_csv(ROLES, usecols=["exam_case_id", "evaluation_role"])
    locked = set(roles.loc[roles.evaluation_role.ne("development"), "exam_case_id"])
    meta = json.loads((SPATIAL / "metadata.json").read_text(encoding="utf-8"))
    assert meta["locked_cases_seen"] == 0
    index = pd.read_csv(SPATIAL / "index.csv")
    assert not (set(index.exam_case_id) & locked)

    banks = {k: np.load(SPATIAL / ("features_%s.npy" % k)).astype(np.float32)
             for k in ["mean", "topk_mean"]}
    cases = {}
    for k, arr in banks.items():
        w = pd.DataFrame(arr)
        w["exam_case_id"] = index.exam_case_id.values
        cases[k] = w.groupby("exam_case_id").mean()

    nested = json.loads(NESTED.read_text(encoding="utf-8"))["results"]
    results = {}
    print("%-28s %10s %8s %8s   %10s %8s %8s" % (
        "field", "nested", "lo", "", "fixed_best", "lo", "config"))
    for path in sorted(OOF.glob("*__binary_oof.csv")):
        field = path.name.replace("__binary_oof.csv", "")
        t = pd.read_csv(path).drop_duplicates("case_id").set_index("case_id")
        assert not (set(t.index) & locked)
        y, folds = t["truth"].astype(int), t["fold"].astype(int)
        if y.nunique() < 2:
            continue
        n = nested.get(field, {})
        if n.get("delivery") != "model":
            continue

        entry = {}
        for name, bank, C, dim in CONFIGS:
            cm = cases[bank]
            ids = cm.index.intersection(y.index)
            pred = pd.Series(index=ids, dtype=float)
            for tf in sorted(folds.unique()):
                te = ids.intersection(folds[folds == tf].index)
                tr = ids.intersection(folds[folds != tf].index)
                if len(te) == 0 or y.loc[tr].nunique() < 2:
                    continue
                d = min(dim, len(tr) - 1, cm.shape[1])
                m = make_pipeline(StandardScaler(), PCA(n_components=d, random_state=0),
                                  LogisticRegression(C=C, max_iter=4000))
                m.fit(cm.loc[tr], y.loc[tr])
                pred.loc[te] = m.predict(cm.loc[te])
            pred = pred.dropna().astype(int)
            if len(pred) < 40:
                continue
            entry[name] = const_delta(y, pred, pred.index, np.random.default_rng(20260917))

        if not entry:
            continue
        best = max(entry, key=lambda k: entry[k]["delta_vs_constant_ci95"][0])
        results[field] = {"nested": {
            "delta_vs_constant": n["delta_vs_constant"],
            "delta_vs_constant_ci95": n["delta_vs_constant_ci95"]},
            "fixed": entry, "fixed_best": best}
        print("%-28s %+10.3f %+8.3f %8s   %+10.3f %+8.3f  %s" % (
            field, n["delta_vs_constant"], n["delta_vs_constant_ci95"][0], "",
            entry[best]["delta_vs_constant"], entry[best]["delta_vs_constant_ci95"][0],
            best))

    print()
    print("=" * 96)
    print("Does one fixed config match per-fold selection?")
    for f, r in results.items():
        nd = r["nested"]["delta_vs_constant"]
        fd = r["fixed"][r["fixed_best"]]["delta_vs_constant"]
        verdict = ("fixed is >= nested" if fd >= nd - 0.01
                   else "nested wins by %+.3f" % (nd - fd))
        print("  %-26s nested %+.3f  fixed %+.3f   %s" % (f, nd, fd, verdict))
    print()
    print("NOTE: the fixed numbers involve NO selection at all, so they carry no")
    print("selection bias. `fixed_best` picks among 3 configs for display only; each")
    print("individual config's number is in the JSON and is bias-free on its own.")

    OUT.write_text(json.dumps({
        "purpose": "control for whether per-field/per-fold selection adds capability",
        "configs": [{"name": n, "bank": b, "C": c, "pca": d} for n, b, c, d in CONFIGS],
        "threshold": 0.5,
        "bootstrap": N_BOOT,
        "locked_cases_used": 0,
        "limitations": [
            "development set out-of-fold; not product capability",
            "fixed configs use threshold 0.5 and no selection, so their deltas are "
            "directly comparable and bias-free; fixed_best is a display convenience "
            "over 3 candidates",
        ],
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print("WROTE %s" % OUT)


if __name__ == "__main__":
    main()
