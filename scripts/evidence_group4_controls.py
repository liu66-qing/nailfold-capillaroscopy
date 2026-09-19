#!/usr/bin/env python
"""EVIDENCE GROUP 4: the model/training configuration, and the six control experiments.

Part 1 is a factual dump of the configuration, read from the artifacts themselves
(feature metadata, encoder build code, estimator constants) -- nothing asserted.

Part 2 addresses the six controls the user named. Three already exist in the
repository and are cited with their numbers. Three do not, and this script RUNS
them on DEVELOPMENT ONLY:
  (2) a conventional supervised CNN/ViT trained from ImageNet init -- run here as
      the closest cheap equivalent: an ImageNet-pretrained ResNet18 feature probe,
      same estimator and same folds, so the comparison isolates the backbone
  (3) a model on human-measurable features only
  (5) a small-sample overfit test: can the pipeline memorise 20 cases?
All three are development-only. locked-47 is NOT touched by this script.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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

CONFIG_FACTS = {
    "encoder": {
        "model": "timm vit_base_patch14_dinov2.lvd142m",
        "family": "DINOv2 ViT-B/14, LVD-142M pretraining",
        "parameters": 86579712,
        "parameters_m": 86.6,
        "embed_dim": 768,
        "depth": 12,
        "patch_size": [14, 14],
        "weights_sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
        "weights_identical_dev_and_locked": True,
        "frozen_or_finetuned": "FROZEN. model.eval(), no gradient, no LoRA, no "
                               "unfrozen blocks. The only trained parameters in "
                               "the delivered configuration are the PCA basis and "
                               "one logistic regression per field.",
        "trainable_parameter_count_in_delivery": "PCA(64) basis + LogisticRegression "
                                                 "on 64 dims = 65 trainable numbers "
                                                 "per field (64 coefficients + 1 "
                                                 "intercept), plus the unsupervised "
                                                 "PCA/scaler fit",
    },
    "preprocessing": {
        "resize": "bicubic to 518x686 (NOT square; square_center_crop=false)",
        "patch_grid": [37, 49],
        "normalisation": "ImageNet mean/std",
        "pos_embed": "bicubically resampled from the 37x37 pretrain grid",
        "augmentation": "NONE. No flip, no rotation, no colour jitter, no crop "
                        "jitter. Features are extracted once per image and reused.",
        "tta": "NONE in the delivered configuration",
    },
    "input_unit": {
        "what_the_model_sees": "ONE STILL IMAGE at a time",
        "training_rows": "image rows; the exam-level label is broadcast to every "
                         "image of that exam",
        "inference_aggregation": "mean of the 5 pooling probabilities, then mean "
                                 "over a case's images -> one probability per case",
        "multi_crop": "NO",
        "video_frames": "NOT USED. 242 videos exist and are unused by the "
                        "delivered configuration.",
        "patient_level_aggregation": "IMPOSSIBLE -- patient_id is null for all rows; "
                                     "aggregation is at exam level",
    },
    "poolings": {
        "members": POOLINGS,
        "topk": 16,
        "what_they_are": "statistics over the 37x49 patch tokens of a frozen "
                         "encoder, plus the CLS token. They are not learned "
                         "detectors and imply no lesion localisation.",
    },
    "estimator": {
        "pipeline": "StandardScaler -> PCA(64, random_state=20260917) -> "
                    "LogisticRegression(C=0.03, max_iter=4000)",
        "loss": "L2-regularised logistic loss (sklearn default lbfgs), one "
                "BINARY problem per field",
        "class_weights": "NONE. class_weight=None deliberately -- balanced weights "
                         "were tested and refuted "
                         "(nailfold-perfield-selection-refuted)",
        "sampling": "NONE. No oversampling, no undersampling, no focal loss in the "
                    "delivered configuration.",
        "multi_task_loss_weighting": "NOT APPLICABLE -- there is no multi-task "
                                     "model. Each field is an independent "
                                     "logistic regression on the same frozen "
                                     "features. No shared head, no shared trunk "
                                     "beyond the frozen encoder, no loss balancing.",
        "one_head_per_field": "Each field has its own independent estimator. They "
                              "share the features, nothing else.",
        "threshold": 0.5,
        "per_field_selection": "NONE -- one fixed configuration for all fields. "
                               "Per-field selection was tested and shown to be "
                               "negative-value (nailfold-perfield-selection-refuted).",
    },
    "cross_validation": {
        "unit": "exam (exam_case_id)",
        "column": "development_fold ONLY; the `split` column must never be used "
                  "(it yields the opposite conclusion)",
        "locked_cases_in_any_training_fold": 0,
    },
}

EXISTING_CONTROLS = {
    "1_dinov2_frozen_plus_linear_classifier": {
        "status": "THIS IS THE DELIVERED CONFIGURATION ITSELF -- no separate run "
                  "is needed. The delivery is a frozen DINOv2 + PCA + linear "
                  "(logistic) classifier. There is no finetuned model to compare "
                  "it against in the delivered line.",
        "numbers": "development OOF: clarity +0.3297, SVP +0.2663, exudation "
                   "+0.2404, blood_color +0.1326, malformation +0.0802; locked: "
                   "see group 3",
        "source": "artifacts/experiments/variance_20260918/imagelevel_ensemble.json",
    },
    "4_label_randomization_sanity_check": {
        "status": "ALREADY RUN -- 20 label shuffles per field, same pipeline",
        "source": "artifacts/experiments/variance_20260918/imagelevel_ensemble.json"
                  " keys shuffled_null_mean / shuffled_null_p95",
        "results": {
            "clarity": {"null_mean": -0.0043, "null_p95": 0.0649,
                        "real_delta": 0.3297, "verdict": "real >> null"},
            "subpapillary_venous_plexus": {"null_mean": -0.0448,
                                           "null_p95": 0.0223,
                                           "real_delta": 0.2663,
                                           "verdict": "real >> null"},
            "exudation": {"null_mean": -0.0134, "null_p95": 0.0721,
                          "real_delta": 0.2404, "verdict": "real >> null"},
            "blood_color": {"null_mean": -0.0478, "null_p95": 0.0235,
                            "real_delta": 0.1326, "verdict": "real > null"},
            "malformation_ratio": {"null_mean": -0.0540, "null_p95": -0.0056,
                                   "real_delta": 0.0802,
                                   "verdict": "real > null, but the real delta's "
                                              "own CI covers zero"},
        },
        "interpretation": "The pipeline does NOT manufacture signal from noise: "
                          "under shuffled labels it lands at or below zero. So "
                          "the development deltas are not a leakage artifact. "
                          "This says nothing about whether they generalise.",
    },
    "6_case_level_cross_validation": {
        "status": "ALREADY THE DEFAULT -- every number in this project uses "
                  "case-level (exam-level) folds from development_fold. There is "
                  "no image-level-random-split number anywhere in the delivered "
                  "line.",
        "verified": "artifacts/experiments/split_integrity_20260919/"
                    "split_integrity.json: 0 exams and 0 duplicate_groups span "
                    "the split; 0 of 2089 image hashes shared across it",
        "caveat": "case-level, NOT patient-level. patient_id is null for all 233 "
                  "rows, so if one person was examined twice their exams could "
                  "straddle the split undetected. UNQUANTIFIED.",
    },
}


def binarise(series, field):
    out = {}
    for case, raw in series.items():
        s = str(raw).strip()
        for lab, vals in BINARY_MAP[field].items():
            if s in vals:
                out[case] = float(lab)
    return pd.Series(out, dtype=float)


def fit_predict(Xtr, ytr, Xts, C=C_FIXED, dim=DIM_FIXED):
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(np.mean(ytr)))
    d = max(2, min(dim, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=d, random_state=SEED),
                         LogisticRegression(C=C, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def boot_delta(yt, yp, maj, rng):
    n = len(yt)
    acc = float((yp == yt).mean())
    base = float((yt == maj).mean())
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = (yp[i] == yt[i]).mean() - (yt[i] == maj).mean()
    return {"n": n, "accuracy": round(acc, 4), "baseline": round(base, 4),
            "delta": round(acc - base, 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "balanced_accuracy": round(float(balanced_accuracy_score(
                yt.astype(int), yp.astype(int))), 4) if len(set(yt)) == 2 else None,
            "passes": bool(np.percentile(d, 2.5) > 0)}


def oof(feat_by_pool, cases, folds_map, y_case, order):
    """Case-level OOF with the delivered ensemble. feat_by_pool may hold one key."""
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    fseries = pd.Series([folds_map.get(c, np.nan) for c in order], index=order)
    for fo in sorted(pd.Series(fseries).dropna().unique()):
        te = set(fseries.index[fseries == fo])
        tr = set(order) - te
        tr_m = np.isin(cases, list(tr))
        te_m = np.isin(cases, list(te))
        if tr_m.sum() == 0 or te_m.sum() == 0:
            continue
        ytr = y_case.reindex(cases[tr_m]).to_numpy()
        ps = [fit_predict(F[tr_m], ytr, F[te_m]) for F in feat_by_pool.values()]
        p = np.mean(ps, axis=0)
        s = pd.Series(p, index=cases[te_m]).groupby(level=0).mean()
        for c, v in s.items():
            if c in pos:
                out[pos[c]] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-store", default="artifacts/features_spatial/native")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--resnet-store", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    m = m.set_index("exam_case_id")
    dev_ids = m.index[m.evaluation_role == "development"]
    folds_map = m.development_fold.to_dict()

    ix = pd.read_csv(os.path.join(a.dev_store, "index.csv"))
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    feats = {p: np.load(os.path.join(a.dev_store, f"features_{p}.npy"))
                 .astype(np.float32) for p in POOLINGS}
    assert not set(ix.exam_case_id) & set(m.index[m.evaluation_role == "locked_test"]), \
        "locked case in the development store"

    # human-measurable features
    f = pd.read_csv(a.files)
    f["exam_case_id"] = f.exam_case_id.astype(str)
    cap = f[(f.role == "cap_image") & (~f.is_black_placeholder.astype(bool))]
    human = cap.groupby("exam_case_id").agg(
        gray_mean_mean=("gray_mean", "mean"), gray_mean_std=("gray_mean", "std"),
        gray_std_mean=("gray_std", "mean"), gray_std_std=("gray_std", "std"),
        n_images=("sha256", "size")).fillna(0.0)

    out = {"part1_configuration": CONFIG_FACTS,
           "part2_controls": {"already_in_repository": EXISTING_CONTROLS,
                              "run_by_this_script": {}},
           "locked_cases_seen": 0}

    ran = out["part2_controls"]["run_by_this_script"]
    ran["_scope"] = ("DEVELOPMENT ONLY. locked-47 is not read by this script. "
                     "All numbers below are case-level OOF on the 186 "
                     "development exams.")

    for field in FIELDS:
        raw = m[field]
        y = binarise(raw, field)
        y_dev = y.reindex(dev_ids).dropna()
        order = sorted(y_dev.index)
        dm = ix.exam_case_id.isin(y_dev.index).to_numpy()
        cases = ix.exam_case_id[dm].to_numpy()
        yt = y_dev.reindex(order).to_numpy()
        maj = float(np.bincount(yt.astype(int)).argmax())
        rng = np.random.default_rng(SEED)
        entry = {}

        # ---- reference: the delivered configuration -------------------------
        fbp = {p: feats[p][dm] for p in POOLINGS}
        ref = oof(fbp, cases, folds_map, y_dev, order)
        ok = ~np.isnan(ref)
        entry["reference_delivered_config"] = boot_delta(
            yt[ok], (ref[ok] >= 0.5).astype(float), maj, rng)

        # ---- control 3: human-measurable features only ----------------------
        comm = [c for c in order if c in human.index]
        Hc = human.loc[comm].to_numpy()
        yh = y_dev.reindex(comm).to_numpy()
        hf = {"human": Hc}
        hp = oof(hf, np.array(comm), folds_map, y_dev, comm)
        okh = ~np.isnan(hp)
        entry["control3_human_features_logreg"] = boot_delta(
            yh[okh], (hp[okh] >= 0.5).astype(float), maj,
            np.random.default_rng(SEED))
        entry["control3_human_features_logreg"]["features"] = list(human.columns)

        # random forest on the same human features
        fseries = pd.Series([folds_map.get(c, np.nan) for c in comm], index=comm)
        rfp = np.full(len(comm), np.nan)
        posc = {c: i for i, c in enumerate(comm)}
        for fo in sorted(fseries.dropna().unique()):
            te = fseries.index[fseries == fo]
            tr = [c for c in comm if c not in set(te)]
            if len(set(y_dev.reindex(tr))) < 2:
                continue
            rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                        min_samples_leaf=3)
            rf.fit(human.loc[tr].to_numpy(), y_dev.reindex(tr).to_numpy())
            for c, v in zip(te, rf.predict_proba(
                    human.loc[te].to_numpy())[:, 1]):
                rfp[posc[c]] = v
        okr = ~np.isnan(rfp)
        entry["control3_human_features_randomforest"] = boot_delta(
            yh[okr], (rfp[okr] >= 0.5).astype(float), maj,
            np.random.default_rng(SEED))

        # ---- control 2: conventional ImageNet CNN backbone ------------------
        if a.resnet_store and os.path.exists(
                os.path.join(a.resnet_store, "index.csv")):
            rix = pd.read_csv(os.path.join(a.resnet_store, "index.csv"))
            rix["exam_case_id"] = rix.exam_case_id.astype(str)
            R = np.load(os.path.join(a.resnet_store, "features_mean.npy"))\
                  .astype(np.float32)
            rm = rix.exam_case_id.isin(y_dev.index).to_numpy()
            rp = oof({"r": R[rm]}, rix.exam_case_id[rm].to_numpy(), folds_map,
                     y_dev, order)
            okrn = ~np.isnan(rp)
            entry["control2_imagenet_cnn_backbone"] = boot_delta(
                yt[okrn], (rp[okrn] >= 0.5).astype(float), maj,
                np.random.default_rng(SEED))
            entry["control2_imagenet_cnn_backbone"]["backbone"] = a.resnet_store
        else:
            entry["control2_imagenet_cnn_backbone"] = {
                "status": "NOT RUN -- no ImageNet backbone feature store was "
                          "provided (--resnet-store). This control is MISSING.",
            }

        # ---- control 5: small-sample overfit test ---------------------------
        # can the pipeline memorise 20 cases when trained and tested on them?
        sub = order[:20]
        subm = np.isin(cases, sub)
        ysub = y_dev.reindex(cases[subm]).to_numpy()
        fits = []
        for p in POOLINGS:
            X = feats[p][dm][subm]
            fits.append(fit_predict(X, ysub, X))
        pin = np.mean(fits, axis=0)
        s = pd.Series(pin, index=cases[subm]).groupby(level=0).mean()
        ysc = y_dev.reindex(s.index).to_numpy()
        entry["control5_small_sample_overfit"] = {
            "n_cases": int(len(s)),
            "train_equals_test": True,
            "in_sample_accuracy": round(float(
                ((s.to_numpy() >= 0.5).astype(float) == ysc).mean()), 4),
            "class_balance": {"n_abnormal": int(ysc.sum()),
                              "n_normal": int(len(ysc) - ysc.sum())},
            "interpretation_rule": "if this is not near 1.0, the failure is in "
                                   "labels/preprocessing/implementation, not in "
                                   "sample size. If it IS near 1.0, the pipeline "
                                   "can fit the data and the problem is "
                                   "generalisation.",
        }
        # a harder version: unregularised, full dimensionality
        fits2 = []
        for p in POOLINGS:
            X = feats[p][dm][subm]
            fits2.append(fit_predict(X, ysub, X, C=1e6, dim=min(64, subm.sum() - 1)))
        pin2 = np.mean(fits2, axis=0)
        s2 = pd.Series(pin2, index=cases[subm]).groupby(level=0).mean()
        entry["control5_small_sample_overfit"]["in_sample_accuracy_unregularised"] = \
            round(float(((s2.to_numpy() >= 0.5).astype(float)
                         == y_dev.reindex(s2.index).to_numpy()).mean()), 4)

        ran[field] = entry
        print(field,
              "| delivered", entry["reference_delivered_config"]["delta"],
              "| human-LR", entry["control3_human_features_logreg"]["delta"],
              "| human-RF", entry["control3_human_features_randomforest"]["delta"],
              "| overfit20", entry["control5_small_sample_overfit"]
                                  ["in_sample_accuracy"],
              "unreg", entry["control5_small_sample_overfit"]
                            ["in_sample_accuracy_unregularised"])

    out["missing_controls"] = [
        "control 2 (conventional supervised CNN/ViT baseline) is MISSING unless "
        "--resnet-store was supplied. Without it I cannot say whether DINOv2 "
        "beats an ordinary ImageNet backbone on this data.",
    ]
    out["limitations"] = [
        "every number here is DEVELOPMENT OOF and must never be reported as "
        "product capability",
        "the human-measurable-feature control uses grey mean/std only, which is "
        "a weak stand-in for a clinical hand-crafted feature set; it bounds the "
        "trivial-signal floor and no more",
        "the overfit test trains and tests on the same 20 cases on purpose; the "
        "number is diagnostic only and is not a performance claim",
        "the reference row recomputes the delivered development OOF; small "
        "differences from the frozen numbers would indicate a reimplementation "
        "drift and should be checked before reading anything else",
    ]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
