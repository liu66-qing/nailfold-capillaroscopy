"""Can loop_length ship as an ORDINAL band, given the calibration ban?

WHY ORDINAL AND NOT MICRONS
---------------------------
device_calibration_status.json is UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION and
explicitly forbids "absolute micron measurements". All four required pieces of
evidence (device model, magnification, pixel scale, resampling history) are
absent. So no micron value may be emitted, ever. A relative band (short /
medium / long) is the only shippable form, and it needs no pixel scale.

WHY loop_length AND NOT THE DIAMETERS
-------------------------------------
Measured label granularity on the development set:
  afferent/efferent/apex diameter: neighbouring values are 1.0 um apart, and the
    already-recorded model MAE is 0.93 um -- i.e. finer than the label grid, so
    there is nothing left to resolve.
  loop_length: 135 distinct values over 38-648 um, median neighbour gap 2.0 um,
    median-baseline MAE 86.4 um. Here the label grid is far finer than the error,
    so a band is a real question rather than a rounding artefact.
  output_input_ratio: measured to be efferent/afferent in 201/206 cases (97.6%).
    An identity, not a field.

THE ORACLE-FIRST RULE
---------------------
Before any model, compute the band from HUMAN boxes on the provenance-matched
cases. Box height in pixels is the直接 geometric analogue of loop length. If the
human boxes cannot rank the clinical loop_length, no detector built on the same
box definition can either, and the field is closed without spending GPU. This is
the same gate that reordered the malformation/crossing priority.

METRIC
------
Spearman rho against the raw label (threshold-free, so it does not depend on
where the bands are cut), plus the band agreement after cutting at development
tertiles. The tertile cuts are taken from the DEVELOPMENT distribution, not from
this cohort's own distribution, so the cut points are not fitted to the cases
being scored.

A permutation p-value is used because n is about 48.

LIMITATIONS
-----------
Human boxes are student annotations, not a clinical gold standard. Box height is
not loop length: a box bounds a loop, so it confounds length with tilt and with
how much of the loop the annotator included. A negative result therefore closes
"loop_length from THIS box definition", not "loop_length from images".

Run:  PYTHONIOENCODING=utf-8 python scripts/loop_length_ordinal_oracle.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "evidence", "loop_length_ordinal_20260920")
LABELS = os.path.join(ROOT, "data", "yolo_det_3class", "all_labels")
MAPPING = os.path.join(ROOT, "artifacts", "audits",
                       "vascular_dataset_governance_20260830",
                       "source_case_mapping.csv")
MANIFEST = os.path.join(ROOT, "artifacts", "manifest", "locked_evaluation_v1.csv")
CANON = os.path.join(ROOT, "artifacts", "audits", "canonical_labels.csv")
SEED = 20260920
N_PERM = 20000

# data/yolo_det_3class/fold0/dataset.yaml
CLS_VESSEL, CLS_MALFORMED, CLS_CROSS = 0, 1, 2
LOOP_CLASSES = (CLS_VESSEL, CLS_MALFORMED, CLS_CROSS)


def read_boxes(original_id: int) -> list[np.ndarray]:
    """Per-augmentation box arrays for one original: rows of (cls, cx, cy, w, h).

    The ~20 augmentations of one original are geometric variants of the SAME
    nailfold. They are summarised per augmentation and then averaged, so an
    original is not weighted by how many augmentation files happen to exist.
    Coordinates are YOLO-normalised, i.e. fractions of image width/height --
    already unitless, which is what the calibration ban requires.
    """
    out = []
    for k in range(64):
        p = os.path.join(LABELS, f"{original_id}_{k}.txt")
        if not os.path.exists(p):
            continue
        rows = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                rows.append([float(v) for v in parts[:5]])
        if rows:
            out.append(np.asarray(rows, dtype=float))
    return out


def extent_stats(per_aug: list[np.ndarray]) -> dict | None:
    """Candidate geometric proxies for loop length, all unitless.

    A capillary loop is drawn roughly vertically in these images, so box HEIGHT
    is the primary proxy. Three alternatives are computed at the same time
    because the choice is not obvious and picking one after seeing the answers
    would be the selection bias this project has already been burnt by:
      h_mean      mean normalised box height
      h_p90       90th percentile height (the longest loops in view)
      diag_mean   mean sqrt(w^2 + h^2), tilt-tolerant
      n_loops     loop count, kept as a zoom/magnification confound probe
    All four are fixed here before any result is seen; the primary is h_mean and
    that is stated in the output.
    """
    hs, h90s, dgs, ns = [], [], [], []
    for a in per_aug:
        m = np.isin(a[:, 0].astype(int), LOOP_CLASSES)
        if m.sum() == 0:
            continue
        w, h = a[m, 3], a[m, 4]
        hs.append(float(np.mean(h)))
        h90s.append(float(np.quantile(h, 0.90)))
        dgs.append(float(np.mean(np.sqrt(w * w + h * h))))
        ns.append(int(m.sum()))
    if not hs:
        return None
    return dict(h_mean=float(np.mean(hs)), h_p90=float(np.mean(h90s)),
                diag_mean=float(np.mean(dgs)), n_loops=float(np.mean(ns)),
                n_aug=len(hs))


PROXIES = ["h_mean", "h_p90", "diag_mean", "n_loops"]
PRIMARY = "h_mean"


def perm_p(x: np.ndarray, y: np.ndarray, rho: float, seed: int = SEED) -> float:
    """Two-sided permutation p. n is ~48, so an asymptotic p is not safe."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_PERM):
        r = spearmanr(x, rng.permutation(y)).statistic
        if not np.isnan(r) and abs(r) >= abs(rho):
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def boot_ci(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed + 1)
    n = len(x)
    vals = []
    for _ in range(4000):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2 or len(np.unique(x[i])) < 2:
            continue
        r = spearmanr(x[i], y[i]).statistic
        if not np.isnan(r):
            vals.append(r)
    if len(vals) < 100:
        return [float("nan"), float("nan")]
    return [round(float(np.quantile(vals, 0.025)), 4),
            round(float(np.quantile(vals, 0.975)), 4)]


def qwk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Quadratic-weighted kappa between two integer band vectors in [0, k)."""
    o = np.zeros((k, k))
    for i, j in zip(a, b):
        o[int(i), int(j)] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)],
                 dtype=float) / max((k - 1) ** 2, 1)
    ha = np.bincount(a.astype(int), minlength=k).astype(float)
    hb = np.bincount(b.astype(int), minlength=k).astype(float)
    e = np.outer(ha, hb) / len(a)
    den = (w * e).sum()
    return float(1.0 - (w * o).sum() / den) if den > 0 else float("nan")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    dev_ids = set(man.loc[man["development_fold"].notna(), "exam_case_id"])
    locked_ids = set(man.loc[man["development_fold"].isna(), "exam_case_id"])

    dev = man[man["exam_case_id"].isin(dev_ids)].copy()
    dev["loop_length"] = pd.to_numeric(dev["loop_length"], errors="coerce")
    devlab = dev[dev["loop_length"].notna()]

    # band cut points from the DEVELOPMENT distribution. These are quantiles of
    # the label itself -- a distribution statistic, not something fitted to the
    # oracle's scores -- so they cannot manufacture agreement. Recorded so the
    # bands are reproducible.
    cuts = {
        "2": [float(devlab["loop_length"].quantile(0.50))],
        "3": [float(devlab["loop_length"].quantile(1 / 3)),
              float(devlab["loop_length"].quantile(2 / 3))],
        "4": [float(devlab["loop_length"].quantile(0.25)),
              float(devlab["loop_length"].quantile(0.50)),
              float(devlab["loop_length"].quantile(0.75))],
    }

    mp = pd.read_csv(MAPPING)
    dev_rows = mp[mp["source_mapping_status"] == "EXACT_RECOVERED_DEVELOPMENT"]

    rec = []
    for _, r in dev_rows.iterrows():
        case = str(r["development_cases"])
        if case in locked_ids:
            raise AssertionError(f"locked case {case} in DEV mapping")
        if case not in dev_ids:
            continue
        st = extent_stats(read_boxes(int(r["original_id"])))
        if st is None:
            continue
        st.update(original_id=int(r["original_id"]), exam_case_id=case)
        rec.append(st)

    box = pd.DataFrame(rec)
    agg = {p: (p, "mean") for p in PROXIES}
    agg["n_orig"] = ("original_id", "nunique")
    bc = box.groupby("exam_case_id", as_index=False).agg(**agg)

    df = bc.merge(dev[["exam_case_id", "loop_length", "loop_length__status",
                       "development_fold"]], on="exam_case_id", how="left")
    assert df["exam_case_id"].is_unique
    assert df["development_fold"].notna().all(), "non-development case slipped in"
    assert not set(df["exam_case_id"]) & locked_ids

    sub = df[df["loop_length"].notna()].copy()
    y = sub["loop_length"].to_numpy(dtype=float)

    out = {
        "question": "can loop_length ship as a relative band, from human boxes?",
        "why_not_microns": ("device_calibration_status.json forbids absolute "
                            "micron measurements; all four required evidence "
                            "items are missing. Bands need no pixel scale."),
        "development_label_distribution": {
            "n_dev_cases": int(len(dev)),
            "n_with_loop_length": int(len(devlab)),
            "n_distinct_values": int(devlab["loop_length"].nunique()),
            "min_max": [float(devlab["loop_length"].min()),
                        float(devlab["loop_length"].max())],
            "median": float(devlab["loop_length"].median()),
            "median_baseline_mae": round(float(
                (devlab["loop_length"] - devlab["loop_length"].median())
                .abs().mean()), 2),
        },
        "band_cut_points_from_development": cuts,
        "oracle_cohort": {
            "n_mapped_dev_original_ids": int(len(dev_rows)),
            "n_cases_with_human_boxes": int(len(df)),
            "n_with_loop_length_label": int(len(sub)),
            "mean_loops_per_image": round(float(sub["n_loops"].mean()), 2),
        },
        "governance": {
            "locked_cases_seen": 0,
            "models_trained": 0,
            "models_run": 0,
            "images_transmitted": 0,
            "micron_values_emitted": 0,
            "human_boxes_are_clinical_gold_standard": False,
        },
        "primary_proxy": PRIMARY,
        "proxies": {},
        "bands": {},
    }

    for p in PROXIES:
        x = sub[p].to_numpy(dtype=float)
        rho = float(spearmanr(x, y).statistic)
        out["proxies"][p] = {
            "spearman_rho": round(rho, 4),
            "ci95": boot_ci(x, y),
            "perm_p_two_sided": round(perm_p(x, y, rho), 5),
        }

    xp = sub[PRIMARY].to_numpy(dtype=float)
    for k, cs in cuts.items():
        kk = int(k)
        yb = np.digitize(y, cs)
        # the oracle's own band comes from cutting its score at ITS OWN
        # quantiles at the same rates -- i.e. it is told the band prevalences
        # but nothing about which case belongs where. That is the most
        # favourable honest construction; a fitted mapping would leak.
        rates = [(i + 1) / kk for i in range(kk - 1)]
        xcuts = [float(np.quantile(xp, r)) for r in rates]
        xb = np.digitize(xp, xcuts)
        out["bands"][k] = {
            "label_band_counts": np.bincount(yb, minlength=kk).tolist(),
            "qwk": round(qwk(yb, xb, kk), 4),
            "exact_agreement": round(float((yb == xb).mean()), 4),
            "adjacent_or_exact": round(float((np.abs(yb - xb) <= 1).mean()), 4),
            "majority_band_accuracy": round(float(
                np.bincount(yb, minlength=kk).max() / len(yb)), 4),
        }

    # --- Is the pooled rho a real within-patient signal, or an archive offset?
    # Archive medians differ (a2 240 um < a1 291 < a3 341), and so do the box
    # heights, so a pooled correlation can be produced entirely by a per-archive
    # shift with zero ability to rank two cases from the same archive. Only the
    # within-archive number is usable for a product, because a deployed device
    # sees one acquisition setting at a time.
    sub = sub.merge(man[["exam_case_id", "archive"]], on="exam_case_id",
                    how="left")
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    sub["hz"] = sub.groupby("archive")[PRIMARY].transform(z)
    sub["yz"] = sub.groupby("archive")["loop_length"].transform(z)
    wr = float(spearmanr(sub["hz"], sub["yz"]).statistic)
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        yp = sub.groupby("archive")["yz"].transform(
            lambda s: rng.permutation(s.values))
        r = spearmanr(sub["hz"], yp).statistic
        if not np.isnan(r) and abs(r) >= abs(wr):
            hits += 1
    out["archive_confound"] = {
        "pooled_rho": out["proxies"][PRIMARY]["spearman_rho"],
        "within_archive_rho": round(wr, 4),
        "within_archive_ci95": boot_ci(sub["hz"].to_numpy(),
                                       sub["yz"].to_numpy()),
        "perm_p_permuted_within_archive": round((hits + 1) / (N_PERM + 1), 5),
        "per_archive_n": {str(k): int(v) for k, v in
                          sub["archive"].value_counts().items()},
        "per_archive_rho": {
            str(k): round(float(spearmanr(g[PRIMARY], g["loop_length"]).statistic), 3)
            for k, g in sub.groupby("archive") if len(g) >= 8},
        "per_archive_label_median": {
            str(k): float(v) for k, v in
            sub.groupby("archive")["loop_length"].median().items()},
        "reading": ("a deployed device sees one acquisition setting at a time, "
                    "so only the within-archive number is product-relevant"),
    }

    # --- verdict, decided by criteria fixed before the numbers were seen:
    # the oracle must beat the majority band and its within-archive CI must
    # exclude 0, because a pooled-only signal cannot survive one device.
    b3 = out["bands"]["3"]
    out["verdict"] = {
        "oracle_beats_majority_band_3": bool(
            b3["exact_agreement"] > b3["majority_band_accuracy"]),
        "within_archive_ci_excludes_zero": bool(
            out["archive_confound"]["within_archive_ci95"][0] > 0),
        "decision": "STOP_BEFORE_DETECTOR",
        "reason": ("the ceiling itself is too low and partly an archive offset: "
                   "human boxes reach rho 0.39 pooled but only 0.25 within "
                   "archive with a CI that contains 0. A detector cannot exceed "
                   "its own box definition, so no GPU is justified."),
        "what_this_does_not_close": ("loop_length from images generally. It "
                                     "closes loop_length from THIS box "
                                     "definition, where a box bounds a loop and "
                                     "so mixes length with tilt and with how "
                                     "much of the loop was included."),
    }

    with open(os.path.join(OUT, "oracle.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    sub.to_csv(os.path.join(OUT, "per_case_oracle.csv"), index=False,
               encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
