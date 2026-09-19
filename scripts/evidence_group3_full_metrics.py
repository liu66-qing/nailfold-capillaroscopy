#!/usr/bin/env python
"""EVIDENCE GROUP 3: the complete test-result panel for the 5 delivered fields.

Not accuracy. Per field, on locked-47 AND on development OOF:
  confusion matrix, balanced accuracy, macro-F1, sensitivity, specificity, PPV,
  NPV, AUROC, AUPRC, bootstrap 95% CI on every one of them, Brier score and a
  calibration table, results stratified by archive, and three explicit baselines:
  majority-class, random guessing at the observed prevalence, and a simple
  human-measurable-feature model (logistic regression on image statistics only).

WHAT THIS SCRIPT DOES AND DOES NOT DO
-------------------------------------
It RE-USES the delivered configuration exactly as scripts/eval_locked_delivered.py
defines it, refits it once on development, and predicts locked once -- so the
locked predictions are identical to the ones already reported. It then computes
more metrics on those same predictions. It does NOT tune anything, does not select
anything on locked, and does not change the configuration. No metric computed here
may be used to pick a configuration; locked-47's evaluation budget is already
overspent and this run adds nothing to what it spent.

The simple-feature baseline is a genuine control, not a formality: if grey-level
statistics alone match DINOv2, the backbone is not the bottleneck.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
FIELDS = ["clarity", "subpapillary_venous_plexus", "exudation", "blood_color",
          "malformation_ratio"]

BINARY_MAP = {
    "clarity": {0: ["清晰"], 1: ["不清", "模糊"]},
    "subpapillary_venous_plexus": {0: ["不见"],
                                   1: ["可见1排", "可见2排", ">2排,扩张"]},
    "exudation": {0: ["无"], 1: ["+", "++", "+++"]},
    "blood_color": {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]},
    "malformation_ratio": {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]},
}


def binarise(series):
    out = {}
    for case, raw in series.items():
        s = str(raw).strip()
        for lab, vals in BINARY_MAP[series.name].items():
            if s in vals:
                out[case] = float(lab)
    return pd.Series(out, dtype=float)


def load_store(path):
    ix = pd.read_csv(os.path.join(path, "index.csv"))
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    feats = {p: np.load(os.path.join(path, f"features_{p}.npy")).astype(np.float32)
             for p in POOLINGS}
    return ix, feats


def fit_predict(Xtr, ytr, Xts):
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(np.mean(ytr)))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def case_mean(prob, cases, order):
    s = pd.Series(prob, index=cases).groupby(level=0).mean()
    return s.reindex(order).to_numpy()


def panel(yt, prob, thr=0.5):
    """Every requested metric on one set of predictions."""
    yp = (prob >= thr).astype(int)
    yt = yt.astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    two_class = len(set(yt)) == 2
    out = {
        "n": int(len(yt)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_layout": "rows=truth(0,1), cols=pred(0,1) -> [[tn,fp],[fn,tp]]",
        "accuracy": round(float((yp == yt).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4)
                             if two_class else None,
        "macro_f1": round(float(f1_score(yt, yp, average="macro",
                                         zero_division=0)), 4),
        "sensitivity_recall_abnormal": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity_recall_normal": round(tn / (tn + fp), 4) if tn + fp else None,
        "ppv_precision_abnormal": round(tp / (tp + fp), 4) if tp + fp else None,
        "npv": round(tn / (tn + fn), 4) if tn + fn else None,
        "auroc": round(float(roc_auc_score(yt, prob)), 4) if two_class else None,
        "auprc": round(float(average_precision_score(yt, prob)), 4)
                 if two_class else None,
        "prevalence_abnormal": round(float(yt.mean()), 4),
        "predicted_abnormal_share": round(float(yp.mean()), 4),
        "brier_score": round(float(brier_score_loss(yt, prob)), 4)
                       if two_class else None,
    }
    return out


def boot_ci(yt, prob, rng, thr=0.5):
    """Bootstrap CI for the panel metrics that the user asked for CIs on."""
    keys = ["accuracy", "balanced_accuracy", "macro_f1",
            "sensitivity_recall_abnormal", "specificity_recall_normal",
            "ppv_precision_abnormal", "npv", "auroc", "auprc", "brier_score"]
    acc = {k: [] for k in keys}
    n = len(yt)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(set(yt[i])) < 2:
            continue
        p = panel(yt[i], prob[i], thr)
        for k in keys:
            if p[k] is not None:
                acc[k].append(p[k])
    return {k: ([round(float(np.percentile(v, 2.5)), 4),
                 round(float(np.percentile(v, 97.5)), 4)] if len(v) > 50
                else "CI_UNAVAILABLE_too_few_valid_resamples")
            for k, v in acc.items()}


def calibration(yt, prob, n_bins=5):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if m.sum() == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f}]", "n": 0})
            continue
        rows.append({"bin": f"[{lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                     "mean_predicted": round(float(prob[m].mean()), 4),
                     "observed_abnormal_rate": round(float(yt[m].mean()), 4)})
    return rows


def simple_feature_table(files_csv, ids):
    """Human-measurable image statistics only: grey mean/std per case.

    These are the only numbers in files.csv a person could read off without a
    model. If this matches DINOv2, the backbone is not the bottleneck.
    """
    f = pd.read_csv(files_csv)
    f["exam_case_id"] = f.exam_case_id.astype(str)
    cap = f[(f.role == "cap_image") & (~f.is_black_placeholder.astype(bool))]
    g = cap.groupby("exam_case_id").agg(
        gm_mean=("gray_mean", "mean"), gm_std=("gray_mean", "std"),
        gs_mean=("gray_std", "mean"), gs_std=("gray_std", "std"),
        n_img=("sha256", "size"))
    g = g.fillna(0.0)
    return g.reindex([i for i in ids if i in g.index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-store", default="artifacts/features_spatial/native")
    ap.add_argument("--locked-store",
                    default="artifacts/features_spatial/native_locked")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-understand-the-cohort-was-already-seen",
                    action="store_true")
    a = ap.parse_args()
    if not getattr(a, "i_understand_the_cohort_was_already_seen"):
        raise SystemExit(
            "Refusing to run without --i-understand-the-cohort-was-already-seen. "
            "This re-scores locked-47. It computes NO new configuration choice, "
            "but the cohort's budget is already overspent and that must be "
            "acknowledged explicitly.")

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    m = m.set_index("exam_case_id")
    dev_ids = m.index[m.evaluation_role == "development"]
    lk_ids = m.index[m.evaluation_role == "locked_test"]

    dix, dfe = load_store(a.dev_store)
    lix, lfe = load_store(a.locked_store)
    assert set(lix.exam_case_id) <= set(lk_ids), "locked store leaked dev cases"

    rng = np.random.default_rng(SEED)
    out = {
        "what_this_is": "the full metric panel for the delivered configuration; "
                        "predictions are identical to locked_delivered.json",
        "config": {"features": a.dev_store, "poolings": POOLINGS,
                   "estimator": f"StandardScaler->PCA({DIM_FIXED})->"
                                f"LogisticRegression(C={C_FIXED})",
                   "threshold": 0.5, "n_boot": N_BOOT, "seed": SEED},
        "locked_cases_seen_in_training": 0,
        "fields": {},
    }

    for field in FIELDS:
        raw = m[field]
        raw.name = field
        y = binarise(raw)
        y_dev = y.reindex(dev_ids).dropna()
        y_lk = y.reindex(lk_ids).dropna()

        # image rows with a label
        dm = dix.exam_case_id.isin(y_dev.index).to_numpy()
        lm = lix.exam_case_id.isin(y_lk.index).to_numpy()
        ytr_img = y_dev.reindex(dix.exam_case_id[dm]).to_numpy()
        dev_order = sorted(y_dev.index)
        lk_order = sorted(y_lk.index)

        # ensemble of the 5 poolings, mean of probabilities
        probs_lk, probs_dev_insample = [], []
        for p in POOLINGS:
            Xtr = dfe[p][dm]
            pr_lk = fit_predict(Xtr, ytr_img, lfe[p][lm])
            probs_lk.append(case_mean(pr_lk, lix.exam_case_id[lm].to_numpy(),
                                      lk_order))
        prob_lk = np.mean(probs_lk, axis=0)
        yt_lk = y_dev.reindex(lk_order) if False else y_lk.reindex(lk_order).to_numpy()

        # ---- development OOF, case-level folds, for comparison ---------------
        folds = m.development_fold.reindex(dev_order)
        oof = np.full(len(dev_order), np.nan)
        pos = {c: i for i, c in enumerate(dev_order)}
        for fo in sorted(folds.dropna().unique()):
            te_cases = set(folds.index[folds == fo])
            tr_cases = set(dev_order) - te_cases
            tr_m = dm & dix.exam_case_id.isin(tr_cases).to_numpy()
            te_m = dm & dix.exam_case_id.isin(te_cases).to_numpy()
            if tr_m.sum() == 0 or te_m.sum() == 0:
                continue
            ps = []
            for p in POOLINGS:
                pr = fit_predict(dfe[p][tr_m],
                                 y_dev.reindex(dix.exam_case_id[tr_m]).to_numpy(),
                                 dfe[p][te_m])
                ps.append(pr)
            cm = case_mean(np.mean(ps, axis=0),
                           dix.exam_case_id[te_m].to_numpy(), sorted(te_cases))
            for c, v in zip(sorted(te_cases), cm):
                oof[pos[c]] = v
        ok = ~np.isnan(oof)
        yt_dev = y_dev.reindex(dev_order).to_numpy()

        entry = {
            "labelled_n": {"development": int(len(y_dev)), "locked": int(len(y_lk))},
            "binary_scheme": {str(k): v for k, v in BINARY_MAP[field].items()},
            "locked": panel(yt_lk, prob_lk),
            "locked_ci95": boot_ci(yt_lk, prob_lk, np.random.default_rng(SEED)),
            "locked_calibration": calibration(yt_lk, prob_lk),
            "development_oof": panel(yt_dev[ok], oof[ok]),
            "development_oof_ci95": boot_ci(yt_dev[ok], oof[ok],
                                            np.random.default_rng(SEED)),
            "development_oof_calibration": calibration(yt_dev[ok], oof[ok]),
        }

        # ---- three baselines on locked --------------------------------------
        dev_maj = int(np.bincount(yt_dev.astype(int)).argmax())
        lk_maj = int(np.bincount(yt_lk.astype(int)).argmax())
        prev = float(yt_lk.mean())
        rand_rng = np.random.default_rng(SEED)
        rand_acc, rand_ba = [], []
        for _ in range(N_BOOT):
            rp = (rand_rng.random(len(yt_lk)) < prev).astype(int)
            rand_acc.append(float((rp == yt_lk).mean()))
            if len(set(yt_lk)) == 2:
                rand_ba.append(float(balanced_accuracy_score(yt_lk, rp)))
        entry["baselines_on_locked"] = {
            "majority_class_frozen_development_answer": {
                "answer": dev_maj,
                "accuracy": round(float((yt_lk == dev_maj).mean()), 4),
                "balanced_accuracy": 0.5,
                "macro_f1": round(float(f1_score(
                    yt_lk, np.full(len(yt_lk), dev_maj), average="macro",
                    zero_division=0)), 4),
            },
            "majority_class_locked_own_answer": {
                "answer": lk_maj,
                "accuracy": round(float((yt_lk == lk_maj).mean()), 4),
                "balanced_accuracy": 0.5,
                "note": "the STRICTER ruler when the two answers differ; using it "
                        "is not allowed for selection, only for reporting",
                "answers_differ": bool(lk_maj != dev_maj),
            },
            "random_guessing_at_observed_prevalence": {
                "accuracy_mean": round(float(np.mean(rand_acc)), 4),
                "accuracy_ci95": [round(float(np.percentile(rand_acc, 2.5)), 4),
                                  round(float(np.percentile(rand_acc, 97.5)), 4)],
                "balanced_accuracy_mean": round(float(np.mean(rand_ba)), 4)
                                          if rand_ba else None,
                "auroc_expected": 0.5,
            },
        }

        # ---- simple human-measurable-feature baseline ------------------------
        g_dev = simple_feature_table(a.files, dev_order)
        g_lk = simple_feature_table(a.files, lk_order)
        common = [c for c in dev_order if c in g_dev.index]
        common_lk = [c for c in lk_order if c in g_lk.index]
        ysd = y_dev.reindex(common).to_numpy()
        yslk = y_lk.reindex(common_lk).to_numpy()
        sp = fit_predict(g_dev.loc[common].to_numpy(), ysd,
                         g_lk.loc[common_lk].to_numpy())
        entry["simple_feature_baseline_on_locked"] = {
            "features": list(g_dev.columns),
            "what": "per-case grey-level statistics from files.csv only; no "
                    "learned representation",
            "n": int(len(yslk)),
            **panel(yslk, sp),
            "delta_vs_frozen_dev_majority": round(
                float((sp >= 0.5).astype(int).__eq__(yslk).mean()
                      - (yslk == dev_maj).mean()), 4),
        }

        # ---- stratification by archive ---------------------------------------
        arch = m.archive_id.reindex(lk_order)
        strat = {}
        for av in sorted(arch.dropna().unique()):
            sel = (arch == av).to_numpy()
            if sel.sum() < 5:
                strat[str(av)] = {"n": int(sel.sum()),
                                  "status": "N_TOO_SMALL_not_reported"}
                continue
            sub = panel(yt_lk[sel], prob_lk[sel])
            sub["baseline_frozen_dev_majority"] = round(
                float((yt_lk[sel] == dev_maj).mean()), 4)
            strat[str(av)] = sub
        entry["locked_stratified_by_archive"] = strat
        entry["stratification_by_device_or_magnification"] = (
            "IMPOSSIBLE -- no device or magnification is recorded anywhere; see "
            "group 2 acquisition_conditions")
        entry["stratification_by_patient"] = (
            "IMPOSSIBLE -- patient_id is null for all 233 rows")

        out["fields"][field] = entry
        print(field, "locked acc", entry["locked"]["accuracy"],
              "BA", entry["locked"]["balanced_accuracy"],
              "AUROC", entry["locked"]["auroc"],
              "| simple-feat acc",
              entry["simple_feature_baseline_on_locked"]["accuracy"])

    out["limitations"] = [
        "n=46-47 on locked: every CI here is wide, roughly +-0.13 to +-0.25. A "
        "metric whose CI covers its baseline is UNRESOLVED, not 'close'.",
        "locked-47 is the SAME sites and the SAME capture pipeline as "
        "development. It is an internal held-out set, NOT external validation. "
        "This project has zero external validation.",
        "the cohort's one-shot budget was already spent at least five times with "
        "model selection performed on it in three runs; these numbers inherit "
        "that contamination and cannot be read as a clean first look",
        "AUROC and AUPRC use the ensemble mean probability, which was never "
        "calibrated; Brier and the calibration table show the miscalibration "
        "rather than correcting it",
        "the simple-feature baseline uses only grey mean/std, which is a weak "
        "proxy for 'human-measurable features'. It bounds the trivial-signal "
        "level from below; it does not prove a hand-crafted clinical feature set "
        "would do as badly.",
    ]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
