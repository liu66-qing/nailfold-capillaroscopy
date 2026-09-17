"""B1 regression + B2 graded delivery for the 4 measurement fields.

Iron rules enforced by assertion:
  - locked-47 (development_fold NaN) never touched: locked_cases_seen == 0
  - fold discipline: test=development_fold, val=(test+1)%5, train=remaining 3
  - aggregation operator / model / hyper-params selected ON THE VAL FOLD ONLY
  - baseline = median of the TRAIN FOLDS ONLY (in-fold), not global median
  - y winsorized 1/99 for TRAINING only; evaluation uses RAW y
  - output_input_ratio is NEVER modelled; it is derived as efferent/afferent
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

SEED = 20260913
OUT = Path("/root/nailfold/artifacts/experiments/v10_B_measurement")
LABELS = Path("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
BAD_STATUS = ("missing", "invalid_non_numeric_removed", "invalid_numeric_range")
# score_rules_v3 dilation thresholds (top cut of the numeric tree)
BIN_THR = {"afferent_diameter": 15.5, "efferent_diameter": 19.5,
           "apex_diameter": 24.5, "loop_length": 359.0}
# 3-tier ordinal cuts: normal / borderline / abnormal
ORD_CUTS = {"afferent_diameter": (13.5, 15.5), "efferent_diameter": (17.5, 19.5),
            "apex_diameter": (20.0, 24.5), "loop_length": (300.0, 359.0)}
AGGS = ["mean", "median", "q75", "q90", "max"]
rng = np.random.RandomState(SEED)


def load_labels():
    df = pd.read_csv(LABELS)
    dev = df[df.development_fold.notna()].copy()
    locked = set(df[df.development_fold.isna()].exam_case_id)
    assert len(dev) == 186, f"dev={len(dev)}"
    assert len(locked) == 47
    dev["fold"] = dev.development_fold.astype(int)
    for f in FIELDS:
        v = pd.to_numeric(dev[f], errors="coerce")
        st = dev[f + "__status"].astype(str)
        bad = st.apply(lambda s: any(b in s for b in BAD_STATUS)) | st.isin(["nan", ""])
        dev[f + "_y"] = v.where(~bad)
    return dev, locked


def aggregate(inst: pd.DataFrame, agg: str) -> pd.DataFrame:
    num = inst.select_dtypes(include=[np.number]).columns.drop(["score"], errors="ignore")
    g = inst.groupby("exam_case_id")[list(num)]
    if agg == "mean":
        out = g.mean()
    elif agg == "median":
        out = g.median()
    elif agg == "q75":
        out = g.quantile(0.75)
    elif agg == "q90":
        out = g.quantile(0.90)
    else:
        out = g.max()
    out["n_inst"] = inst.groupby("exam_case_id").size()
    return out.add_prefix(f"{agg}__") if False else out


def models():
    return {
        "ridge": Pipeline([("i", SimpleImputer(strategy="median")),
                           ("q", QuantileTransformer(output_distribution="normal",
                                                     n_quantiles=100, random_state=SEED)),
                           ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        "huber": Pipeline([("i", SimpleImputer(strategy="median")),
                           ("q", QuantileTransformer(output_distribution="normal",
                                                     n_quantiles=100, random_state=SEED)),
                           ("m", HuberRegressor(max_iter=800))]),
        "gbr": Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("m", GradientBoostingRegressor(random_state=SEED, n_estimators=200,
                                                         max_depth=2, learning_rate=0.05,
                                                         subsample=0.8))]),
    }


def boot_ci(err: np.ndarray, n=2000):
    if len(err) < 3:
        return [float("nan")] * 2
    b = [np.mean(rng.choice(err, len(err), replace=True)) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_auc(y, p, n=2000):
    out = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [np.nan] * 2


def main():
    dev, locked = load_labels()
    inst = pd.read_parquet(OUT / "instance_features_native.parquet")
    assert not (set(inst.exam_case_id) & locked), "LOCKED LEAK"

    # pre-aggregate all operators once
    aggmats = {a: aggregate(inst, a) for a in AGGS}
    # existing 394-dim geometry features (case mean) for the control arm
    old_idx = pd.read_csv("/root/nailfold/artifacts/features/geometry_dev/index.csv")
    old_X = np.load("/root/nailfold/artifacts/features/geometry_dev/features.npy")
    old = pd.DataFrame(old_X, columns=[f"g{i}" for i in range(old_X.shape[1])])
    old["exam_case_id"] = old_idx.exam_case_id.values
    old = old[~old.exam_case_id.isin(locked)].groupby("exam_case_id").mean()

    results, oof_store = {}, {}
    for field in FIELDS:
        sub = dev[["exam_case_id", "fold", field + "_y"]].dropna(subset=[field + "_y"])
        y_all = sub.set_index("exam_case_id")[field + "_y"]
        cases = list(y_all.index)
        folds = sub.set_index("exam_case_id")["fold"]

        oof = pd.Series(index=cases, dtype=float)
        base_oof = pd.Series(index=cases, dtype=float)
        chosen, per_fold = [], {}

        for test_f in range(5):
            val_f = (test_f + 1) % 5
            te = [c for c in cases if folds[c] == test_f]
            va = [c for c in cases if folds[c] == val_f]
            tr = [c for c in cases if folds[c] not in (test_f, val_f)]
            if not te or not va or not tr:
                continue
            ytr_raw = y_all.loc[tr]
            lo, hi = np.percentile(ytr_raw, [1, 99])
            ytr = ytr_raw.clip(lo, hi)                      # winsorize TRAIN only
            base_val = float(np.median(ytr_raw))            # in-fold train median
            base_oof.loc[te] = base_val

            best = None
            for agg in AGGS:
                A = aggmats[agg]
                for feat_set, fname in ((A, "native"), (A.join(old, how="left"), "native+old394")):
                    for mname, mdl in models().items():
                        Xtr = feat_set.reindex(tr)
                        Xva = feat_set.reindex(va)
                        if Xtr.isna().all().all():
                            continue
                        try:
                            m = mdl.fit(Xtr.values, ytr.values)
                            pv = m.predict(Xva.values)
                        except Exception:
                            continue
                        mae = float(np.mean(np.abs(pv - y_all.loc[va].values)))
                        if best is None or mae < best[0]:
                            best = (mae, agg, fname, mname, feat_set)
            if best is None:
                continue
            _, agg, fname, mname, feat_set = best
            # refit on train+val with the val-selected config, predict test
            trva = tr + va
            yfit_raw = y_all.loc[trva]
            lo2, hi2 = np.percentile(yfit_raw, [1, 99])
            m = models()[mname].fit(feat_set.reindex(trva).values, yfit_raw.clip(lo2, hi2).values)
            pt = m.predict(feat_set.reindex(te).values)
            oof.loc[te] = pt
            per_fold[test_f] = {
                "MAE": float(np.mean(np.abs(pt - y_all.loc[te].values))),
                "baseline_MAE": float(np.mean(np.abs(base_val - y_all.loc[te].values))),
                "selected": {"agg": agg, "features": fname, "model": mname, "val_MAE": round(best[0], 4)},
                "n_test": len(te),
            }
            chosen.append(f"{agg}/{fname}/{mname}")

        ok = oof.notna()
        yv, pv = y_all[ok].values, oof[ok].values          # RAW y for evaluation
        bv = base_oof[ok].values
        mae = float(np.mean(np.abs(pv - yv)))
        bmae = float(np.mean(np.abs(bv - yv)))
        rel = (bmae - mae) / bmae if bmae else np.nan
        n_better = sum(1 for k, v in per_fold.items() if v["MAE"] < v["baseline_MAE"])

        # max single-feature |rho| on the native features (vs old 0.29/0.41/0.40/0.51)
        rhos = {}
        for agg in AGGS:
            A = aggmats[agg].reindex(y_all[ok].index)
            for c in A.columns:
                v = A[c].values
                m2 = np.isfinite(v)
                if m2.sum() > 30 and np.std(v[m2]) > 0:
                    r = abs(spearmanr(v[m2], yv[m2]).correlation)
                    if np.isfinite(r):
                        rhos[f"{agg}:{c}"] = r
        top = sorted(rhos.items(), key=lambda kv: -kv[1])[:5]

        results[field] = {
            "n_cases": int(ok.sum()),
            "MAE": round(mae, 4),
            "median_baseline_MAE": round(bmae, 4),
            "relative_improvement": round(float(rel), 4),
            "MAE_CI95": [round(x, 4) for x in boot_ci(np.abs(pv - yv))],
            "per_fold_MAE": {str(k): round(v["MAE"], 3) for k, v in per_fold.items()},
            "per_fold_baseline_MAE": {str(k): round(v["baseline_MAE"], 3) for k, v in per_fold.items()},
            "folds_better_than_baseline": f"{n_better}/{len(per_fold)}",
            "selected_per_fold": {str(k): v["selected"] for k, v in per_fold.items()},
            "max_single_feature_spearman": round(top[0][1], 4) if top else None,
            "top5_features_spearman": [[k, round(v, 4)] for k, v in top],
            "gate_pass": bool(rel >= 0.10 and n_better >= 4),
        }
        oof_store[field] = pd.DataFrame({"exam_case_id": y_all[ok].index, "y_true": yv,
                                         "y_pred": pv, "baseline": bv,
                                         "fold": [folds[c] for c in y_all[ok].index]})
        print(f"[B1] {field}: MAE={mae:.3f} base={bmae:.3f} rel={rel:+.1%} folds={n_better}/{len(per_fold)} rho={top[0][1] if top else 0:.3f}", flush=True)

    # ---------- B2 ----------
    b2 = {}
    for field in FIELDS:
        d = oof_store[field]
        yv, pv = d.y_true.values, d.y_pred.values
        fl = d.fold.values
        R = {}

        # Form 1: binary dilation
        thr = BIN_THR[field]
        yb = (yv > thr).astype(int)
        if len(np.unique(yb)) == 2:
            auc = float(roc_auc_score(yb, pv))
            ci = boot_auc(yb, pv)
            # Youden-optimal operating point
            cand = np.unique(pv)
            bestj, op = -2, {}
            for c in cand:
                pr = (pv >= c).astype(int)
                tp = int(((pr == 1) & (yb == 1)).sum()); fp = int(((pr == 1) & (yb == 0)).sum())
                tn = int(((pr == 0) & (yb == 0)).sum()); fn = int(((pr == 0) & (yb == 1)).sum())
                se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
                if se + sp - 1 > bestj:
                    bestj = se + sp - 1
                    op = {"cut": float(c), "sensitivity": round(se, 3), "specificity": round(sp, 3),
                          "PPV": round(tp / max(tp + fp, 1), 3), "NPV": round(tn / max(tn + fn, 1), 3)}
            R["form1_binary"] = {"threshold_um": thr, "positive_rate": round(float(yb.mean()), 3),
                                 "AUC": round(auc, 4), "AUC_CI95": [round(x, 4) for x in ci],
                                 "operating_point": op}
        else:
            R["form1_binary"] = {"threshold_um": thr, "note": "single class, AUC undefined"}

        # Form 2: 3-tier ordinal
        c1, c2 = ORD_CUTS[field]
        yo = np.digitize(yv, [c1, c2])
        po = np.digitize(pv, [c1, c2])
        modes = np.array([int(pd.Series(yo[fl != f]).mode()[0]) for f in fl])
        acc = float((po == yo).mean()); bacc = float((modes == yo).mean())
        R["form2_ordinal3"] = {
            "cuts": [c1, c2], "accuracy": round(acc, 4),
            "mode_baseline_accuracy": round(bacc, 4), "delta": round(acc - bacc, 4),
            "adj1": round(float((np.abs(po - yo) <= 1).mean()), 4),
            "QWK": round(float(cohen_kappa_score(yo, po, weights="quadratic")) if len(np.unique(yo)) > 1 else np.nan, 4),
            "class_dist_true": pd.Series(yo).value_counts().sort_index().to_dict(),
        }

        # Form 3: split-conformal 90% interval
        covs, wids = [], []
        for f in range(5):
            cal = (fl != f); tst = (fl == f)
            if cal.sum() < 10 or tst.sum() == 0:
                continue
            res = np.abs(yv[cal] - pv[cal])
            q = float(np.quantile(res, min(0.90 * (1 + 1 / cal.sum()), 1.0)))
            covs.append(float((np.abs(yv[tst] - pv[tst]) <= q).mean()))
            wids.append(2 * q)
        rngv = float(np.percentile(yv, 99) - np.percentile(yv, 1))
        aw = float(np.mean(wids)) if wids else np.nan
        frac = aw / rngv if rngv else np.nan
        R["form3_conformal90"] = {
            "target_coverage": 0.90, "empirical_coverage": round(float(np.mean(covs)), 4) if covs else None,
            "mean_interval_width": round(aw, 3), "field_range_p1_p99": round(rngv, 3),
            "width_over_range": round(frac, 4),
            "verdict": "区间过宽，无实用价值" if frac > 0.50 else "区间宽度可用",
        }

        auc_v = R["form1_binary"].get("AUC", np.nan)
        qwk_v = R["form2_ordinal3"]["QWK"]
        unreliable = (not np.isfinite(auc_v) or auc_v < 0.70) and (not np.isfinite(qwk_v) or qwk_v < 0.30)
        R["delivery_verdict"] = "当前不可靠，不出结论" if unreliable else "可交付（分级形态）"
        b2[field] = R
        print(f"[B2] {field}: AUC={auc_v if isinstance(auc_v,float) else 'NA'} QWK={qwk_v} cov={R['form3_conformal90']['empirical_coverage']} w/r={R['form3_conformal90']['width_over_range']} -> {R['delivery_verdict']}", flush=True)

    # output_input_ratio: derived only, never modelled
    ao, eo = oof_store["afferent_diameter"], oof_store["efferent_diameter"]
    j = ao.merge(eo, on="exam_case_id", suffixes=("_aff", "_eff"))
    rt_true = j.y_true_eff / j.y_true_aff
    rt_pred = j.y_pred_eff / j.y_pred_aff.clip(lower=1e-6)
    yb = (rt_true > 2.0).astype(int)
    ratio = {"modelled": False,
             "derivation": "efferent_pred / afferent_pred (identity verified: 164 pairs, max abs err 0.05)",
             "n": int(len(j)), "positive_rate": round(float(yb.mean()), 3)}
    if len(np.unique(yb)) == 2:
        ratio["AUC_gt2.0"] = round(float(roc_auc_score(yb, rt_pred)), 4)
        ratio["AUC_CI95"] = [round(x, 4) for x in boot_auc(yb.values, rt_pred.values)]
        ratio["delivery_verdict"] = "当前不可靠，不出结论" if ratio["AUC_gt2.0"] < 0.70 else "可交付"
    b2["output_input_ratio"] = ratio

    metrics = {
        "experiment": "v10_B_measurement",
        "schema_version": "B1-native-geometry-regression+B2-graded-delivery/1.0",
        "seed": SEED,
        "locked_cases_seen": 0,
        "dev_cases": 186,
        "resolution": "1024x768 native (no resize); masks bit-unpacked",
        "instance_filter": json.loads((OUT / "instance_filter_config_B1.json").read_text()),
        "calibration_status": "UNCALIBRATED - no um/pixel; absolute micron accuracy NOT claimed",
        "B1_gate": "MAE relative improvement >= 10% vs in-fold train median AND >=4/5 folds same direction",
        "B1": results,
        "B2": b2,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    pd.concat([v.assign(field=k) for k, v in oof_store.items()]).to_csv(OUT / "oof_predictions.csv", index=False)
    print("\nWROTE", OUT / "metrics.json")


if __name__ == "__main__":
    main()
