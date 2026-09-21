"""Why do SVP and blood_color fail leave-one-archive-out? Acquisition or labelling?

THE QUESTION AND WHY IT DECIDES WHETHER TO KEEP INVESTING
---------------------------------------------------------
clarity and exudation pass all three LOAO directions. SVP and blood_color score
0.78-0.92 pooled but are archive-dependent. Two very different causes:

  ACQUISITION DRIFT -- the archives were captured differently (illumination,
  white balance, magnification). The same physical finding lands at a different
  pixel value per archive, so a decision boundary learned on one archive is
  wrong on another. We have NO acquisition metadata (established earlier), so
  we cannot correct this from records -- but if the drift is monotone in a
  measurable statistic, per-archive standardisation may still close it, which
  would make the field recoverable.

  LABELLING HABIT -- the archives were read by people using different internal
  thresholds. The image is the same; the word written down differs. Unified
  re-annotation fixes this; no amount of new data does.

HOW THEY ARE TOLD APART
-----------------------
1. Does the IMAGE differ across archives? Train nothing; just test whether
   simple colour/intensity statistics separate archive pairs. High separability
   = the archives are physically different.
2. Does the LABEL rate differ across archives?
3. THE DISCRIMINATOR: within each archive separately, how well does a simple
   statistic rank the label, and where does the boundary sit?
     - similar within-archive ranking + shifted feature level  => acquisition
       drift, potentially fixable by per-archive standardisation
     - within-archive ranking itself differs                   => the feature
       means different things per archive: labelling habit or a real population
       difference, NOT fixable by normalisation
4. Direct test of the fix: does per-archive z-scoring of the feature raise the
   cross-archive transfer of that feature? Measured, not assumed.

blood_color is the cleanest case to run this on: the label IS a colour word, so
the physically corresponding statistic (hue/saturation/value of the frame) is
known in advance rather than learned, which removes the usual circularity.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It trains no model and touches no locked case. It cannot prove acquisition drift
absent metadata; it can only show whether the observable statistics behave the
way each hypothesis predicts. A real population difference between archives
(different patient mix) mimics labelling habit in test 3 and is not separable
here -- that ambiguity is reported, not hidden.

Run:  PYTHONIOENCODING=utf-8 python scripts/diagnose_archive_dependence.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "evidence", "archive_dependence_20260920")
MANIFEST = os.path.join(ROOT, "artifacts", "manifest", "locked_evaluation_v1.csv")
CANON = os.path.join(ROOT, "artifacts", "audits", "canonical_labels.csv")
CASE_ROOT = os.path.join(ROOT, "data")
SEED = 20260920

# abnormal definitions, taken from the label scheme already on file
ABN = {
    "blood_color": lambda s: s.isin(["暗红", "暗紫"]),
    "subpapillary_venous_plexus": lambda s: s.isin([">2排,扩张"]),
    "clarity": lambda s: s.isin(["不清", "模糊"]),
    "exudation": lambda s: ~s.isin(["无"]),
}


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    p, n = s[y == 1], s[y == 0]
    if len(p) < 3 or len(n) < 3:
        return None
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def case_images(case_id: str) -> list[str]:
    archive, num = case_id.split("/")
    d = os.path.join(CASE_ROOT, archive, num)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith("CAPorg") and f.lower().endswith(".jpg"))


def frame_stats(path: str) -> dict | None:
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if float(g.std()) < 8.0:
        return None
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(float), hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    b, gr, r = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    # vessel-ish pixels: the reddest decile, which is what a colour word describes
    redness = r - (b + gr) / 2
    thr = np.percentile(redness, 90)
    m = redness >= thr
    return {
        "mean_v": float(v.mean()), "mean_s": float(s.mean()), "mean_h": float(h.mean()),
        "gray_std": float(g.std()), "mean_r": float(r.mean()),
        "redness_p90": float(thr),
        "vessel_s": float(s[m].mean()), "vessel_v": float(v[m].mean()),
        "vessel_h": float(h[m].mean()),
        "height": int(arr.shape[0]), "width": int(arr.shape[1]),
    }


FEATS = ["mean_v", "mean_s", "mean_h", "gray_std", "mean_r", "redness_p90",
         "vessel_s", "vessel_v", "vessel_h"]


def build_case_features() -> pd.DataFrame:
    man = pd.read_csv(MANIFEST)
    dev = man.loc[man.development_fold.notna(), "exam_case_id"].tolist()
    rows = []
    for cid in dev:
        st = [frame_stats(p) for p in case_images(cid)]
        st = [s for s in st if s is not None]
        if not st:
            continue
        d = {k: float(np.mean([s[k] for s in st])) for k in FEATS}
        d.update(exam_case_id=cid, n_img=len(st),
                 archive=cid.split("/")[0],
                 portrait=float(np.mean([s["height"] > s["width"] for s in st])))
        rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "case_frame_stats.csv")
    if os.path.exists(cache):
        F = pd.read_csv(cache)
        print(f"reusing cached frame stats for {len(F)} cases")
    else:
        F = build_case_features()
        F.to_csv(cache, index=False)
        print(f"computed frame stats for {len(F)} development cases")

    man = pd.read_csv(MANIFEST)
    assert not set(F.exam_case_id) & set(
        man.loc[man.development_fold.isna(), "exam_case_id"]), "locked case present"

    canon = pd.read_csv(CANON).set_index("exam_case_id")
    res = {"n_development_cases": int(len(F)),
           "governance": {"locked_cases_seen": 0, "models_trained": 0,
                          "images_transmitted": 0}}

    # --- 1. do the IMAGES differ by archive? ---------------------------------
    res["image_level_archive_shift"] = {}
    for f in FEATS + ["portrait", "n_img"]:
        g = F.groupby("archive")[f]
        mu = g.mean().round(3).to_dict()
        pooled_sd = float(F[f].std())
        spread = (max(mu.values()) - min(mu.values())) / pooled_sd if pooled_sd > 0 else 0.0
        res["image_level_archive_shift"][f] = {
            "per_archive_mean": mu,
            "range_in_pooled_sd": round(float(spread), 3),
        }

    # pairwise archive separability by a single feature, rank-based (no model)
    res["archive_pair_separability_auroc"] = {}
    archives = sorted(F.archive.unique())
    for i in range(len(archives)):
        for j in range(i + 1, len(archives)):
            a, b = archives[i], archives[j]
            sub = F[F.archive.isin([a, b])]
            y = (sub.archive == b).astype(int).values
            best = max(((abs((auroc(y, sub[f].values) or 0.5) - 0.5) + 0.5, f)
                        for f in FEATS))
            res["archive_pair_separability_auroc"][f"{a}|{b}"] = {
                "best_single_feature": best[1],
                "auroc": round(float(best[0]), 3),
                "per_feature": {f: round(float(auroc(y, sub[f].values) or 0.5), 3)
                                for f in FEATS},
            }
    # --- 2. do the LABEL rates differ by archive? ----------------------------
    res["label_rate_by_archive"] = {}
    for field, fn in ABN.items():
        lab = canon.reindex(F.exam_case_id)[field].astype(str)
        ok = canon.reindex(F.exam_case_id)[field].notna().values
        y = fn(lab).values.astype(int)
        d = {}
        for a in archives:
            m = (F.archive.values == a) & ok
            d[a] = {"n": int(m.sum()), "abnormal_rate": round(float(y[m].mean()), 3)}
        rates = [d[a]["abnormal_rate"] for a in archives]
        res["label_rate_by_archive"][field] = {
            "per_archive": d,
            "max_minus_min": round(float(max(rates) - min(rates)), 3),
        }

    # --- 3. THE DISCRIMINATOR: within-archive ranking vs feature level -------
    # For each field pick, ONCE, the single most plausible physical statistic and
    # keep it fixed across archives. Choosing per archive would guarantee a
    # difference and prove nothing.
    PHYS = {"blood_color": "vessel_v",        # dark red = lower value
            "subpapillary_venous_plexus": "mean_v",
            "clarity": "gray_std",
            "exudation": "mean_s"}
    res["within_archive_ranking"] = {}
    for field, feat in PHYS.items():
        lab = canon.reindex(F.exam_case_id)[field].astype(str)
        ok = canon.reindex(F.exam_case_id)[field].notna().values
        y = ABN[field](lab).values.astype(int)
        x = F[feat].values
        per = {}
        for a in archives:
            m = (F.archive.values == a) & ok
            au = auroc(y[m], x[m])
            per[a] = {
                "n": int(m.sum()),
                "auroc_within_archive": None if au is None else round(au, 3),
                "feature_mean_abnormal": round(float(x[m & (y == 1)].mean()), 2)
                if (m & (y == 1)).sum() else None,
                "feature_mean_normal": round(float(x[m & (y == 0)].mean()), 2)
                if (m & (y == 0)).sum() else None,
            }
        aus = [v["auroc_within_archive"] for v in per.values()
               if v["auroc_within_archive"] is not None]
        # where does the archive's own boundary sit, vs its neighbours'
        levels = [v["feature_mean_abnormal"] for v in per.values()
                  if v["feature_mean_abnormal"] is not None]
        res["within_archive_ranking"][field] = {
            "physical_feature_fixed_in_advance": feat,
            "per_archive": per,
            "ranking_consistent": bool(len(aus) == len(archives)
                                       and (all(a > 0.5 for a in aus)
                                            or all(a < 0.5 for a in aus))),
            "ranking_spread": round(float(max(aus) - min(aus)), 3) if aus else None,
            "abnormal_level_spread_in_pooled_sd": round(
                float((max(levels) - min(levels)) / F[feat].std()), 3) if levels else None,
        }

    # --- 4. does per-archive z-scoring help the feature transfer? ------------
    res["per_archive_standardisation_effect"] = {}
    for field, feat in PHYS.items():
        lab = canon.reindex(F.exam_case_id)[field].astype(str)
        ok = canon.reindex(F.exam_case_id)[field].notna().values
        y = ABN[field](lab).values.astype(int)
        raw = F[feat].values.astype(float)
        z = np.zeros_like(raw)
        for a in archives:
            m = F.archive.values == a
            sd = raw[m].std()
            z[m] = (raw[m] - raw[m].mean()) / (sd if sd > 0 else 1.0)
        res["per_archive_standardisation_effect"][field] = {
            "feature": feat,
            "pooled_auroc_raw": round(float(auroc(y[ok], raw[ok]) or 0.5), 3),
            "pooled_auroc_archive_zscored": round(float(auroc(y[ok], z[ok]) or 0.5), 3),
        }

    res["verdict_rules"] = {
        "acquisition_drift_signature": "ranking_consistent true AND "
            "abnormal_level_spread_in_pooled_sd large AND z-scoring raises pooled AUROC",
        "labelling_or_population_signature": "ranking_consistent false OR "
            "ranking_spread large, with z-scoring not helping",
        "not_separable_here": "a genuine per-archive patient-mix difference "
            "produces the same pattern as labelling habit; no acquisition "
            "metadata exists in this project to break that tie",
    }
    json.dump(res, open(os.path.join(OUT, "diagnosis.json"), "w"),
              indent=2, ensure_ascii=False)
    for k in ("label_rate_by_archive", "within_archive_ranking",
              "per_archive_standardisation_effect"):
        print(json.dumps({k: res[k]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
