#!/usr/bin/env python
"""SUPERSEDED -- DO NOT RUN. Kept only as the record of a wrong protocol.

Superseded by scripts/eval_locked_delivered.py on 2026-09-19, for two reasons:

1. It implements a MIXED configuration. Its field list and frozen development
   numbers come from the geometry+DINOv2 concat run (selective_c05_fixlabels.json),
   but the delivery form's A1 table it was built to test also contained
   subpapillary_venous_plexus, which came from the 5-pooling image-level ensemble
   and is not even a field in the concat run. Testing a mixture measures nothing.
   SPEC_delivery_v2_20260918.md declares ONE delivered configuration; that is what
   eval_locked_delivered.py tests instead.

2. Its own recompute-and-assert guard rejected it. malformation_ratio's recorded
   dev_baseline 0.561 is the baseline INSIDE the coverage-0.80 retained subset, not
   the baseline of the full labelled set (0.5706). The guard fired, and chasing that
   failure is what exposed problem 1. The guard worked; the protocol was wrong.

The title below ("THE ONE-SHOT") was also false when written: the locked-47 budget
had already been spent at least five times before this file existed, with model
selection performed on the cohort in three of those runs. See the budget paragraph
further down, which was added once that was discovered.

Original docstring follows, unedited, because the wrong protocol is the record.
--------------------------------------------------------------------------------

THE ONE-SHOT locked-47 evaluation. Every choice is frozen before locked loads.

locked-47 may be evaluated ONCE. That rule only means something if nothing about
this run can be tuned after seeing the result, so this script hard-codes the
protocol below and takes no tuning arguments at all. There is no model selection,
no threshold search, no arm choice, no coverage search inside this file.

FROZEN PROTOCOL (all values copied from the development runs, cited inline)
--------------------------------------------------------------------------
1. Fields: exactly the list in FIELDS_FROZEN. Nothing is added after the fact.
2. Classifier: LogisticRegression(C=0.03, liblinear), image-level fit on ALL 186
   development cases, predict each locked image, average to the case. Identical to
   the delivered configuration; only the test set changes.
3. Features: geometry (conf=0.05 instance geometry) concatenated with DINOv2 --
   the prespecified 'concat' arm, never a per-field winner.
4. Abstention: malformation_ratio uses conf_threshold = 0.0669, the value read off
   the DEVELOPMENT coverage-0.80 point (selective_c05_fixlabels.json). The locked
   coverage is therefore whatever this fixed threshold yields; it is NOT tuned to
   land on 80%. If locked coverage comes out at 0.62 or 0.93, that is the result.
5. Baseline: the SAME single fixed answer chosen on development (the development
   majority class, hard-coded per field below), NOT the locked majority. Using the
   locked majority would let the test set define its own baseline.
6. Segmenter weights: locked-47 was held out of every seg_vessel_fold{k} training
   set, so all five are clean here. We use the fold-ensemble median-count rule only
   for locked; no fold matching is possible (development_fold is NaN on locked).

THE ONE-SHOT BUDGET WAS ALREADY SPENT BEFORE THIS FILE EXISTED
--------------------------------------------------------------
Honesty requires recording this, because it weakens every number below. A scan of
the repository finds SIX artifacts carrying mode="development_train_locked_test":
  artifacts/geometry_v2_locked/metrics.json
  artifacts/geometry_deploy_v2_locked/metrics.json
  artifacts/geometry_deploy_v3_locked/metrics.json
  artifacts/labelv2_locked_metrics.json
  artifacts/hybrid_base_locked_metrics.json
  artifacts/hybrid_locked_base_server_metrics.json
Three of the geometry runs additionally recorded a per-field `selected` family
(logistic vs extra_trees) that DIFFERS between runs, i.e. model selection was
performed on locked-47. blood_color's delta moved +0.163 -> +0.233 -> +0.341
across those three runs purely with that selection.

So locked-47 is no longer a virgin test set, and this run does not restore that.
Running it once more with a frozen protocol is still the best available evidence --
it is the only locked measurement of the DELIVERED configuration -- but it must be
reported as "a held-out cohort that has been looked at before", never as a clean
one-shot test. Any launch document must carry this paragraph.

WHAT THIS RUN CANNOT DO
-----------------------
It cannot be repeated. If it disagrees with development, that disagreement IS the
finding and must be reported as such -- not used to pick a better configuration.
No number from this run may be fed back into any development decision.

n=47 resolves almost nothing: at a near-even split the bootstrap CI half-width is
about +-0.21. This run can detect a gross failure or a large effect; it cannot
distinguish +0.08 from +0.29. A field whose CI covers zero here is UNRESOLVED,
not "nearly passing".
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

N_BOOT = 2000
SEED = 20260918
C_FIXED = 0.03

# frozen from artifacts/experiments/selective_20260918/selective_c05_fixlabels.json
# 'dev_delta' and 'dev_acc' are recorded so the locked result can be compared
# against a number that was written down BEFORE locked was loaded.
#
# 'dev_baseline' is the development majority-answer accuracy as printed by the
# development run. The script RECOMPUTES the majority answer from development
# labels and asserts the resulting baseline matches this recorded value to within
# 0.002. That assert is the guard: it caught two hand-entry errors of mine
# (clarity's and blood_color's majority class were transposed) before locked was
# ever loaded, and it will catch any label-vocabulary drift too.
FIELDS_FROZEN = {
    "clarity": {
        "normal": ["清晰"],
        "abstain_conf": None,
        "dev_acc": 0.800, "dev_delta": 0.2919, "dev_baseline": 0.508,
    },
    "subpapillary_venous_plexus": {
        # 不见 = plexus not visible = normal; every 可见* value is abnormal and
        # none of them appear in ABNORMAL_STR, so they must be named explicitly
        "normal": ["不见"],
        "abnormal": ["可见1排", "可见2排", ">2排,扩张"],
        "abstain_conf": None,
        "dev_acc": 0.837, "dev_delta": 0.2663, "dev_baseline": 0.571,
    },
    "exudation": {
        "normal": ["无", "０", "0"],
        "abstain_conf": None,
        "dev_acc": 0.743, "dev_delta": 0.2350, "dev_baseline": 0.508,
    },
    "blood_color": {
        "normal": ["浅红", "淡红"], "abnormal": ["暗红", "暗紫", "紫红"],
        "abstain_conf": None,
        "dev_acc": 0.740, "dev_delta": 0.1878, "dev_baseline": 0.552,
    },
    "malformation_ratio": {
        "normal": ["<=10%", "[<10%]"],
        # development coverage-0.80 confidence threshold, frozen
        "abstain_conf": 0.0669,
        "dev_acc": 0.700, "dev_delta": 0.1385, "dev_baseline": 0.561,
    },
}

ABNORMAL_STR = ["不清", "模糊", "30--60%", "60--80%", ">80%", "10--30%", ">60%",
                "5--6", "3--4", "1--2", "<1", "有", "少量", "中量", "大量",
                "轻度", "中度", "重度", "淡", "暗红", "紫红", "淡紫", "紫",
                "１", "２", "1", "2", "3", "＋", "+", "++", "+++"]


def binarise(s, cfg):
    """1 = abnormal. Unparseable text stays NaN, never becomes a class.

    A value present in neither list becomes NaN and is EXCLUDED, never guessed.
    That silent exclusion is exactly what cost blood_color 65 cases on
    development, so this script also reports the excluded count per field.
    """
    v = s.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[v.isin(cfg["normal"])] = 0.0
    out[v.isin(cfg.get("abnormal", ABNORMAL_STR))] = 1.0
    return out


def unmapped_values(s, cfg):
    """Label values that fall through both lists, so they cannot hide."""
    v = s.astype(str).str.strip()
    known = set(cfg["normal"]) | set(cfg.get("abnormal", ABNORMAL_STR))
    bad = v[~v.isin(known) & ~v.isin(["nan", "", "None"])]
    return {str(k): int(n) for k, n in bad.value_counts().items()}


def fit_dev_predict_locked(Xd, dev_case, yd, Xl, locked_case):
    """Fit once on ALL development image rows, predict every locked image.

    No folds here: development is entirely training data for this run, and the
    test set is locked-47. Predictions are averaged to the case, matching the
    delivered aggregation.
    """
    yi = yd.reindex(dev_case).to_numpy()
    keep = ~pd.isna(yi)
    mdl = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C_FIXED, max_iter=5000, solver="liblinear"),
    )
    mdl.fit(Xd[keep], yi[keep].astype(int))
    p = mdl.predict_proba(Xl)[:, 1]
    return pd.Series(p, index=locked_case).groupby(level=0).mean()


def score_locked(yt, yp, dev_majority, rng):
    """Accuracy vs the DEVELOPMENT majority answer, not the locked majority.

    Letting the locked set supply its own baseline would allow the test set to
    define the bar it is measured against, so dev_majority is passed in frozen.
    """
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
    alo, ahi = np.percentile(a, [2.5, 97.5])
    return {"n": n, "accuracy": round(acc, 4),
            "acc_ci95": [round(float(alo), 4), round(float(ahi), 4)],
            "baseline_dev_majority": round(base, 4),
            "delta": round(acc - base, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "usable": bool(lo > 0),
            "abnormal_share": round(float(yt.mean()), 4)}


def load_side(geom_path, dino_feat, dino_index, case_ids, label):
    """Load image-level features for one case set, aligned geometry to DINOv2."""
    G = pd.read_csv(geom_path)
    G["exam_case_id"] = G["exam_case_id"].astype(str)
    G["image_path"] = G["image_path"].astype(str)
    G = G[G.exam_case_id.isin(case_ids)]
    idx = pd.read_csv(dino_index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    F = np.load(dino_feat).astype(np.float32)
    assert len(idx) == len(F), f"{label}: index {len(idx)} != features {len(F)}"
    D = pd.DataFrame(F)
    D["image_path"] = idx["image_path"].astype(str)
    D["exam_case_id"] = idx["exam_case_id"]
    D = D[D.exam_case_id.isin(case_ids)]
    mrg = G.merge(D, on=["exam_case_id", "image_path"], how="inner",
                  suffixes=("", "_d"))
    gcols = [c for c in G.columns
             if c not in ("exam_case_id", "image_path", "fold")
             and pd.api.types.is_numeric_dtype(G[c])]
    X = np.hstack([mrg[gcols].to_numpy(dtype=np.float32),
                   mrg[list(range(F.shape[1]))].to_numpy(dtype=np.float32)])
    print(f"{label}: {len(mrg)} images / {mrg.exam_case_id.nunique()} cases "
          f"/ {X.shape[1]} dims", flush=True)
    return X, pd.Index(mrg["exam_case_id"]), gcols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-geom",
                    default="artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--dev-dino", default="artifacts/features/dinov2/features.npy")
    ap.add_argument("--dev-index", default="artifacts/features/dinov2/index.csv")
    ap.add_argument("--locked-geom", required=True)
    ap.add_argument("--locked-dino", required=True)
    ap.add_argument("--locked-index", required=True)
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-understand-this-is-one-shot", action="store_true",
                    help="required: locked-47 may be evaluated exactly once")
    args = ap.parse_args()
    raise SystemExit(
        "SUPERSEDED -- refusing to run. This script implements a MIXED "
        "configuration (concat fields + an SVP number from the 5-pooling ensemble) "
        "and its own dev_baseline assert rejects malformation_ratio. "
        "Use scripts/eval_locked_delivered.py, which tests the single configuration "
        "declared in SPEC_delivery_v2_20260918.md. See this file's docstring.")
    if not args.i_understand_this_is_one_shot:
        raise SystemExit("refusing to run: pass --i-understand-this-is-one-shot")

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    dev_ids = set(man.loc[man.evaluation_role == "development", "exam_case_id"])
    lk_ids = set(man.loc[man.evaluation_role == "locked_test", "exam_case_id"])
    assert not (dev_ids & lk_ids), "development and locked overlap"
    dev = man[man.evaluation_role == "development"].set_index("exam_case_id")
    lk = man[man.evaluation_role == "locked_test"].set_index("exam_case_id")
    print(f"development {len(dev_ids)} cases / locked {len(lk_ids)} cases",
          flush=True)

    Xd, dev_case, gcols_d = load_side(args.dev_geom, args.dev_dino,
                                      args.dev_index, dev_ids, "development")
    Xl, lk_case, gcols_l = load_side(args.locked_geom, args.locked_dino,
                                     args.locked_index, lk_ids, "locked")
    assert gcols_d == gcols_l, "geometry columns differ between dev and locked"
    assert Xd.shape[1] == Xl.shape[1], "feature dims differ"
    assert not (set(dev_case) & set(lk_case)), "a case appears in both sides"

    out = {"config": {"protocol": "frozen before locked was loaded",
                      "C": C_FIXED, "n_boot": N_BOOT, "seed": SEED,
                      "arm": "concat (geometry + DINOv2), prespecified",
                      "fit": "all development image rows, no folds",
                      "baseline": "development majority answer, frozen",
                      "n_dev_cases": len(dev_ids),
                      "n_locked_cases": len(lk_ids),
                      "dims": int(Xd.shape[1]),
                      "one_shot": True},
           "fields": {}}

    for field, cfg in FIELDS_FROZEN.items():
        yd = binarise(dev[field], cfg).dropna()
        yl = binarise(lk[field], cfg).dropna()
        # the majority answer is derived from DEVELOPMENT and checked against the
        # recorded baseline before any locked number is computed
        dev_majority = float(np.bincount(yd.to_numpy().astype(int)).argmax())
        dev_base_check = float((yd.to_numpy() == dev_majority).mean())
        assert abs(dev_base_check - cfg["dev_baseline"]) < 0.002, (
            f"{field}: recomputed dev baseline {dev_base_check:.4f} != recorded "
            f"{cfg['dev_baseline']}; the protocol drifted, refusing to proceed")

        prob = fit_dev_predict_locked(Xd, dev_case, yd, Xl, lk_case)
        yl = yl[yl.index.isin(prob.index)]
        prob = prob.reindex(yl.index)
        lab = (prob > 0.5).astype(float)

        entry = {"dev_recorded": {"accuracy": cfg["dev_acc"],
                                  "delta": cfg["dev_delta"],
                                  "baseline": cfg["dev_baseline"]},
                 "dev_majority_answer": dev_majority,
                 "dev_baseline_recomputed": round(dev_base_check, 4),
                 "locked_labelled": int(len(yl)),
                 "locked_unmapped_values": unmapped_values(lk[field], cfg),
                 "dev_unmapped_values": unmapped_values(dev[field], cfg)}

        if cfg["abstain_conf"] is None:
            s = score_locked(yl.to_numpy(), lab.to_numpy(), dev_majority,
                             np.random.default_rng(SEED))
            entry["full_coverage"] = s
            entry["abstention"] = None
        else:
            conf = (prob - 0.5).abs()
            keep = conf >= cfg["abstain_conf"]
            entry["full_coverage"] = score_locked(
                yl.to_numpy(), lab.to_numpy(), dev_majority,
                np.random.default_rng(SEED))
            entry["abstention"] = {
                "frozen_conf_threshold": cfg["abstain_conf"],
                "locked_coverage": round(float(keep.mean()), 4),
                "n_answered": int(keep.sum()),
                "n_abstained": int((~keep).sum()),
                "note": ("coverage is whatever the frozen threshold yields on "
                         "locked; it was NOT tuned to reproduce 0.80"),
                "scored": score_locked(yl[keep].to_numpy(), lab[keep].to_numpy(),
                                       dev_majority,
                                       np.random.default_rng(SEED)),
            }
        out["fields"][field] = entry
        fc = entry["full_coverage"]
        line = (f"{field:28s} locked n={fc.get('n')} acc={fc.get('accuracy')} "
                f"base={fc.get('baseline_dev_majority')} d={fc.get('delta')} "
                f"{fc.get('ci95')} pass={fc.get('usable')} "
                f"| dev was acc={cfg['dev_acc']} d={cfg['dev_delta']}")
        if entry["abstention"]:
            ab = entry["abstention"]["scored"]
            line += (f"\n{'':28s} abstain cov={entry['abstention']['locked_coverage']} "
                     f"n={ab.get('n')} acc={ab.get('accuracy')} d={ab.get('delta')} "
                     f"{ab.get('ci95')} pass={ab.get('usable')}")
        print(line, flush=True)

    out["limitations"] = [
        "locked-47 is n=47: CI half-widths are roughly +-0.14 at a near-even "
        "split, so this run can confirm a gross failure but cannot resolve "
        "differences of 0.05 from development",
        "this evaluation may not be repeated; no number here may be fed back "
        "into any development decision",
        "still a single cohort: this is NOT external validation",
        "no micron values are claimed; calibration_factors.json unused",
        "a locked result that disagrees with development IS the finding and must "
        "be reported as such, not used to select a different configuration",
    ]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
