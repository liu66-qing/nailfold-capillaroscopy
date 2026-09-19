#!/usr/bin/env python
"""Locked-47 evaluation of THE DELIVERED CONFIGURATION, and nothing else.

Why this replaces eval_locked_final.py
--------------------------------------
That script implemented a geometry+DINOv2 'concat' logistic regression, and took
its frozen development numbers from selective_c05_fixlabels.json. Its own
recompute-and-assert guard failed on malformation_ratio, and chasing that failure
exposed a worse problem: the delivery form's A1 table mixes two configurations.
clarity/exudation/blood_color/malformation_ratio came from the concat run, while
subpapillary_venous_plexus came from the 5-pooling ensemble -- SVP is not even a
field in the concat run. Testing a mixture on locked would measure nothing.

SPEC_delivery_v2_20260918.md declares ONE delivered configuration, so that is what
is tested here:
  features   artifacts/features_spatial/native, poolings [mean, topk_mean, max, cls, std]
  estimator  StandardScaler -> PCA(64) -> LogisticRegression(C=0.03)
  training   image rows (case label broadcast to its images)
  aggregation mean of the 5 pooling probabilities, then mean over a case's images
  threshold  0.5, no class_weight, no per-field selection
Everything is copied from scripts/eval_imagelevel_ensemble.py, which produced
artifacts/experiments/variance_20260918/imagelevel_ensemble.json.

FROZEN, written down before locked labels were touched
-----------------------------------------------------
Development results of that single configuration (imagelevel_ensemble.json):
  clarity                    d=+0.3297 CI[+0.2324,+0.3730] acc=0.8378 base=0.5081 n=185  5/5 members pass
  subpapillary_venous_plexus d=+0.2663 CI[+0.1793,+0.3424] acc=0.8370 base=0.5707 n=184  5/5
  exudation                  d=+0.2404 CI[+0.1366,+0.2896] acc=0.7486 base=0.5082 n=183  5/5
  blood_color                d=+0.1326 CI[+0.0331,+0.2210] acc=0.6851 base=0.5525 n=181  5/5
  malformation_ratio         d=+0.0802 CI[-0.0062,+0.1667] acc=0.6481 base=0.5679 n=162  2/5  DOES NOT PASS
So under the delivered configuration the honest development count is FOUR, not
five. malformation_ratio is carried here as a prespecified negative control: it is
expected to fail, and if it passes on locked that is luck, not a fifth field.

Baseline is the DEVELOPMENT majority answer, frozen and passed in, never the
locked majority -- otherwise the test set would define its own bar.

THE HELD-OUT COHORT WAS ALREADY LOOKED AT
-----------------------------------------
Six artifacts in this repository carry mode="development_train_locked_test"
(geometry_v2_locked, geometry_deploy_v2_locked, geometry_deploy_v3_locked,
labelv2_locked_metrics, hybrid_base_locked_metrics, hybrid_locked_base_server_metrics),
and three of them recorded a per-field `selected` estimator family that differs
between runs, i.e. model selection was performed on locked-47. blood_color's delta
moved +0.163 -> +0.233 -> +0.341 across those runs on that selection alone.
This run does not restore virginity. It is the only locked measurement of the
delivered configuration, which is why it is worth doing, but every report of it
must say the cohort has been seen before.

n=47 resolves almost nothing: bootstrap CI half-width is about +-0.14 to +-0.21.
A field whose CI covers zero here is UNRESOLVED, not "nearly passing".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# every constant below is copied from eval_imagelevel_ensemble.py
C_FIXED, DIM_FIXED, N_BOOT, SEED = 0.03, 64, 2000, 20260917
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]

# frozen development results of the delivered configuration; 'dev_baseline' is
# recomputed from development labels and asserted to match, which is the guard
# that caught the mixed-configuration problem in the first place
FROZEN = {
    "clarity": {"dev_acc": 0.8378, "dev_delta": 0.3297, "dev_baseline": 0.5081,
                "dev_n": 185, "members_pass": 5, "dev_passes": True},
    "subpapillary_venous_plexus": {"dev_acc": 0.8370, "dev_delta": 0.2663,
                                   "dev_baseline": 0.5707, "dev_n": 184,
                                   "members_pass": 5, "dev_passes": True},
    "exudation": {"dev_acc": 0.7486, "dev_delta": 0.2404, "dev_baseline": 0.5082,
                  "dev_n": 183, "members_pass": 5, "dev_passes": True},
    "blood_color": {"dev_acc": 0.6851, "dev_delta": 0.1326, "dev_baseline": 0.5525,
                    "dev_n": 181, "members_pass": 5, "dev_passes": True},
    # prespecified negative control: failed on development, expected to fail here
    "malformation_ratio": {"dev_acc": 0.6481, "dev_delta": 0.0802,
                           "dev_baseline": 0.5679, "dev_n": 162,
                           "members_pass": 2, "dev_passes": False},
}


def load_labels(oof_dir):
    """Binary truth per case, exactly as the delivered run read it."""
    out = {}
    for fn in sorted(os.listdir(oof_dir)):
        if fn.endswith("__binary_oof.csv"):
            d = pd.read_csv(os.path.join(oof_dir, fn)).drop_duplicates("case_id")
            out[fn.split("__")[0]] = d.set_index("case_id").truth.astype(float)
    return out


def fit_predict(Xtr, ytr, Xts):
    """Identical estimator to eval_imagelevel_ensemble.fit_predict."""
    if len(set(ytr)) < 2:
        return np.full(len(Xts), float(np.mean(ytr)))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def score(yt, yp, dev_majority, rng):
    """Accuracy minus the frozen DEVELOPMENT majority answer."""
    n = len(yt)
    if n < 10:
        return {"n": n, "status": "N_TOO_SMALL"}
    acc = float((yp == yt).mean())
    base = float((yt == dev_majority).mean())
    d = np.empty(N_BOOT)
    a = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        ys, ps = yt[i], yp[i]
        a[b] = (ps == ys).mean()
        d[b] = a[b] - (ys == dev_majority).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4),
            "acc_ci95": [round(float(np.percentile(a, 2.5)), 4),
                         round(float(np.percentile(a, 97.5)), 4)],
            "baseline_dev_majority_answer": round(base, 4),
            "delta": round(acc - base, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "passes": bool(lo > 0),
            "abnormal_share": round(float(yt.mean()), 4)}


# The locked truth must be binarised with the SAME rule the delivered run used.
# That rule was not written as a table anywhere; it is implicit in the
# *__binary_oof.csv files. It was recovered by cross-tabulating each OOF file's
# truth column against the raw manifest value, and every field agreed exactly:
#   clarity        清晰=0                         | 不清, 模糊 = 1
#   SVP            不见=0                         | 可见1排, 可见2排, >2排,扩张 = 1
#   exudation      无=0                           | +, ++, +++ = 1
#   blood_color    浅红, 淡红 = 0                  | 暗红, 暗紫 = 1
#   malformation   <=10% = 0                      | 10--30%, 30--60%, >60% = 1
# A value in neither list becomes NaN and is EXCLUDED, never guessed -- and the
# excluded values are printed, because a silent drop of that kind already cost
# blood_color 65 development cases once.
BINARY_MAP = {
    "clarity": {"normal": ["清晰"], "abnormal": ["不清", "模糊"]},
    "subpapillary_venous_plexus": {"normal": ["不见"],
                                   "abnormal": ["可见1排", "可见2排", ">2排,扩张"]},
    "exudation": {"normal": ["无", "０", "0"], "abnormal": ["+", "++", "+++"]},
    "blood_color": {"normal": ["浅红", "淡红"], "abnormal": ["暗红", "暗紫", "紫红"]},
    "malformation_ratio": {"normal": ["<=10%", "[<10%]"],
                           "abnormal": ["10--30%", "30--60%", ">60%", "60--80%", ">80%"]},
}


def binarise(s, cfg):
    v = s.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[v.isin(cfg["normal"])] = 0.0
    out[v.isin(cfg["abnormal"])] = 1.0
    return out


def unmapped(s, cfg):
    v = s.astype(str).str.strip()
    known = set(cfg["normal"]) | set(cfg["abnormal"])
    bad = v[~v.isin(known) & ~v.isin(["nan", "", "None"])]
    return {str(k): int(n) for k, n in bad.value_counts().items()}


def ensemble_locked(feat_dev, feat_lk, case_dev, case_lk, y_dev):
    """Delivered configuration: fit on ALL development images, predict locked.

    Development is entirely training data here, so there are no folds -- the
    out-of-fold machinery of the development run exists to estimate generalisation
    from development alone, and is replaced by the actual held-out cohort.
    Aggregation order is preserved from the delivered run: average the 5 pooling
    probabilities per image, then average over a case's images.
    """
    yi = y_dev.reindex(case_dev).to_numpy()
    keep = ~pd.isna(yi)
    probs = []
    for p in POOLINGS:
        pr = fit_predict(feat_dev[p][keep], yi[keep].astype(int), feat_lk[p])
        probs.append(pr)
    mean_prob = np.mean(probs, axis=0)
    per_member = {p: pd.Series(pr, index=case_lk).groupby(level=0).mean()
                  for p, pr in zip(POOLINGS, probs)}
    return pd.Series(mean_prob, index=case_lk).groupby(level=0).mean(), per_member


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="artifacts/features_spatial/native")
    ap.add_argument("--feat-dir-locked",
                    default="artifacts/features_spatial/native_locked")
    ap.add_argument("--oof-dir", default="artifacts/experiments/threshold_tuned_20260916")
    ap.add_argument("--roles",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-understand-the-cohort-was-already-seen", action="store_true")
    a = ap.parse_args()
    if not a.i_understand_the_cohort_was_already_seen:
        raise SystemExit("refusing to run: pass "
                         "--i-understand-the-cohort-was-already-seen")

    roles = pd.read_csv(a.roles)
    roles["exam_case_id"] = roles["exam_case_id"].astype(str)
    dev_ids = set(roles.loc[roles.evaluation_role == "development", "exam_case_id"])
    lk_ids = set(roles.loc[roles.evaluation_role == "locked_test", "exam_case_id"])
    assert len(dev_ids) == 186 and len(lk_ids) == 47, (len(dev_ids), len(lk_ids))
    assert not (dev_ids & lk_ids), "development and locked overlap"
    lkman = roles[roles.evaluation_role == "locked_test"].set_index("exam_case_id")
    devman = roles[roles.evaluation_role == "development"].set_index("exam_case_id")

    idx_d = pd.read_csv(os.path.join(a.feat_dir, "index.csv"))
    idx_l = pd.read_csv(os.path.join(a.feat_dir_locked, "index.csv"))
    idx_d["exam_case_id"] = idx_d["exam_case_id"].astype(str)
    idx_l["exam_case_id"] = idx_l["exam_case_id"].astype(str)
    assert not (set(idx_d.exam_case_id) & lk_ids), "locked case in development features"
    assert not (set(idx_l.exam_case_id) & dev_ids), "development case in locked features"
    case_d = pd.Index(idx_d.exam_case_id)
    case_l = pd.Index(idx_l.exam_case_id)

    feat_d, feat_l = {}, {}
    for p in POOLINGS:
        feat_d[p] = np.load(os.path.join(a.feat_dir, "features_%s.npy" % p)).astype(np.float32)
        feat_l[p] = np.load(os.path.join(a.feat_dir_locked, "features_%s.npy" % p)).astype(np.float32)
        assert len(feat_d[p]) == len(idx_d), f"{p}: dev features/index mismatch"
        assert len(feat_l[p]) == len(idx_l), f"{p}: locked features/index mismatch"
        assert feat_d[p].shape[1] == feat_l[p].shape[1], f"{p}: dim mismatch"

    labels = load_labels(a.oof_dir)
    out = {"config": {"delivered_spec": "artifacts/annotations/v2/SPEC_delivery_v2_20260918.md",
                      "members": POOLINGS, "weights": "equal", "C": C_FIXED,
                      "pca_dim": DIM_FIXED, "threshold": 0.5, "class_weight": None,
                      "selection": "none", "level": "image",
                      "fit": "all 186 development cases / %d images" % len(idx_d),
                      "test": "47 locked cases / %d images" % len(idx_l),
                      "baseline": "development majority answer, frozen",
                      "n_boot": N_BOOT, "seed": SEED},
           "cohort_caveat": ("locked-47 carries SIX prior "
                             "development_train_locked_test artifacts and three of "
                             "them selected an estimator family on it; this is a "
                             "previously-seen held-out cohort, not a virgin test set"),
           "fields": {}}

    for field, cfg in FROZEN.items():
        y_oof = labels.get(field)
        assert y_oof is not None, f"{field} has no OOF label file in {a.oof_dir}"
        y_oof.index = y_oof.index.astype(str)
        y_dev = y_oof[y_oof.index.isin(dev_ids)]

        # guard: the frozen development baseline must reproduce from the same
        # labels the delivered run used, else the protocol has drifted
        maj = float(np.bincount(y_dev.to_numpy().astype(int)).argmax())
        base_chk = float((y_dev.to_numpy() == maj).mean())
        assert abs(base_chk - cfg["dev_baseline"]) < 0.002, (
            f"{field}: recomputed dev baseline {base_chk:.4f} != frozen "
            f"{cfg['dev_baseline']}; refusing to proceed")
        assert len(y_dev) == cfg["dev_n"], (
            f"{field}: dev n {len(y_dev)} != frozen {cfg['dev_n']}")

        bm = BINARY_MAP[field]
        y_lk = binarise(lkman[field], bm).dropna()

        prob, members = ensemble_locked(feat_d, feat_l, case_d, case_l, y_dev)
        common = y_lk.index.intersection(prob.index)
        yt = y_lk.loc[common].to_numpy()
        yp = (prob.loc[common] > 0.5).astype(float).to_numpy()

        entry = {
            "dev_frozen": {k: cfg[k] for k in
                           ("dev_acc", "dev_delta", "dev_baseline", "dev_n",
                            "members_pass", "dev_passes")},
            "dev_majority_answer": int(maj),
            "dev_baseline_recomputed": round(base_chk, 4),
            "locked_labelled": int(len(y_lk)),
            "locked_scored": int(len(common)),
            "locked_unmapped_values": unmapped(lkman[field], bm),
            "dev_unmapped_values": unmapped(devman[field], bm),
            "locked": score(yt, yp, maj, np.random.default_rng(SEED)),
        }

        # SECOND RULER, diagnostic only -- the protocol baseline above stays the
        # frozen development majority answer. But when prevalence shifts between
        # cohorts, that frozen answer can be the WRONG answer on locked, which
        # lowers the baseline and inflates the delta without the model improving.
        # clarity is exactly that case: dev majority answer is abnormal, locked is
        # 65% normal. So the locked-majority delta is also recorded. It is the
        # harsher number and it is the one that answers "did this beat guessing on
        # this cohort". Neither is used to choose anything.
        lk_maj = float(np.bincount(yt.astype(int)).argmax())
        entry["locked_majority_answer"] = int(lk_maj)
        entry["prevalence_shift"] = {
            "dev_abnormal_share": round(float(y_dev.mean()), 4),
            "locked_abnormal_share": round(float(yt.mean()), 4),
            "dev_majority_answer_is_wrong_on_locked": bool(lk_maj != maj),
        }
        entry["locked_vs_locked_majority"] = score(
            yt, yp, lk_maj, np.random.default_rng(SEED))
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tp = int(((yp == 1) & (yt == 1)).sum())
        entry["confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        entry["predicted_abnormal_share"] = round(float(yp.mean()), 4)
        entry["collapsed_to_one_class"] = bool(yp.min() == yp.max())
        entry["recall_abnormal"] = round(tp / max(1, tp + fn), 4)
        entry["recall_normal"] = round(tn / max(1, tn + fp), 4)
        # each member alone, so a locked result cannot rest on one lucky pooling
        entry["locked_members"] = {}
        for p, mp in members.items():
            ypm = (mp.loc[common] > 0.5).astype(float).to_numpy()
            s = score(yt, ypm, maj, np.random.default_rng(SEED))
            entry["locked_members"][p] = {"delta": s.get("delta"),
                                          "ci95": s.get("ci95"),
                                          "passes": s.get("passes")}
        entry["locked_members_passing"] = sum(
            1 for v in entry["locked_members"].values() if v.get("passes"))
        out["fields"][field] = entry

        L = entry["locked"]
        tag = "" if cfg["dev_passes"] else "  [NEGATIVE CONTROL, failed on dev]"
        print(f"{field:28s} locked n={L['n']:3d} acc={L['accuracy']:.4f} "
              f"base={L['baseline_dev_majority_answer']:.4f} "
              f"d={L['delta']:+.4f} CI[{L['ci95'][0]:+.4f},{L['ci95'][1]:+.4f}] "
              f"pass={L['passes']} members={entry['locked_members_passing']}/5"
              f"{tag}", flush=True)
        print(f"{'':28s} dev was d={cfg['dev_delta']:+.4f} acc={cfg['dev_acc']:.4f}"
              f"  -> locked change {L['delta'] - cfg['dev_delta']:+.4f}", flush=True)
        M = entry["locked_vs_locked_majority"]
        flip = " <-- frozen dev answer is WRONG on locked" if \
            entry["prevalence_shift"]["dev_majority_answer_is_wrong_on_locked"] else ""
        print(f"{'':28s} vs locked majority: base={M['baseline_dev_majority_answer']:.4f} "
              f"d={M['delta']:+.4f} CI[{M['ci95'][0]:+.4f},{M['ci95'][1]:+.4f}] "
              f"pass={M['passes']}{flip}", flush=True)
        print(f"{'':28s} truth abn={entry['prevalence_shift']['locked_abnormal_share']:.3f} "
              f"(dev {entry['prevalence_shift']['dev_abnormal_share']:.3f})  "
              f"pred abn={entry['predicted_abnormal_share']:.3f}  "
              f"recall abn={entry['recall_abnormal']:.3f} norm={entry['recall_normal']:.3f}"
              f"{'  COLLAPSED' if entry['collapsed_to_one_class'] else ''}\n", flush=True)

    out["limitations"] = [
        "the held-out cohort was looked at before this run (6 prior locked "
        "artifacts, 3 with model selection on it); this is not a clean one-shot",
        "n=47: bootstrap CI half-width ~+-0.14 to +-0.21. A CI covering zero is "
        "UNRESOLVED, not 'nearly passing'. This run cannot distinguish +0.08 "
        "from +0.29",
        "single cohort, single site: this is NOT external validation",
        "malformation_ratio was carried as a prespecified negative control "
        "(2/5 members, CI covering zero on development); if it passes here that "
        "is not evidence of a fifth deliverable field",
        "no micron values claimed; calibration_factors.json unused",
        "a locked result that disagrees with development IS the finding; it must "
        "not be used to choose a different configuration",
        "clarity's protocol delta is inflated by prevalence shift: the frozen "
        "development majority answer is abnormal, but locked is majority normal, "
        "so the protocol baseline drops to ~0.35. Read locked_vs_locked_majority "
        "for clarity, not the headline delta",
    ]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
