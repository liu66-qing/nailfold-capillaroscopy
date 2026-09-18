#!/usr/bin/env python
"""Audit what the seg_vessel_v2 handoff actually delivered downstream.

Why this audit exists
---------------------
seg_vessel_v2_handoff.md reports mask mAP50 = 0.952 and states "分割质量已达上限"
(segmentation quality has reached its ceiling). That number is a *segmentation*
score. It says how well vessel masks match annotated vessel masks. It says
nothing about whether the required report fields can be recognised, and the
mapping between the two was never measured. Reading 0.952 as field capability
is the misreading this script measures.

Three specific defects are checked, each independently:

  D1. Locked contamination in the handoff instruction.
      case_features.csv holds 233 rows = 186 development + 47 locked. Section
      9.1 of the handoff says to train the classifier on "case_features.csv
      (233 例)". Following that instruction trains on locked-47, which is
      forbidden: locked may be used exactly once, for final evaluation.
      This script counts the locked rows and always drops them.

  D2. Information loss at the interface.
      The segmentation produces per-instance masks (polygons, areas, positions,
      confidences) for ~18 vessels per image, ~9 images per case. What crossed
      the interface is 13 case-level scalars: means, medians, std and totals of
      count/confidence/area. Per-instance geometry, spatial arrangement and
      shape were discarded before any classifier saw them. Fields defined by
      the shape and arrangement of individual vessels (crossing_ratio,
      malformation_ratio, capillary_count) cannot be recovered from 13 summary
      statistics, no matter which model is fitted on them.

  D3. The unmeasured link.
      What the 13 scalars are actually worth on the fields they were meant to
      serve, under the same ruler used everywhere else: accuracy minus a single
      fixed majority answer, case-level folds from development_fold, baseline
      recomputed inside every bootstrap resample, verdict by CI lower bound > 0.

Nothing here retrains segmentation. It reads the delivered feature table and
the frozen OOF labels only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BOOT, SEED = 2000, 20260917
# Fields the segmentation features were built to serve: they are defined by the
# count, shape and arrangement of individual vessels.
TARGET_FIELDS = ["capillary_count", "crossing_ratio", "malformation_ratio"]


def load_label(oof_dir, field):
    p = os.path.join(oof_dir, "%s__binary_oof.csv" % field)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).drop_duplicates("case_id")
    return d.set_index("case_id").truth.astype(float)


def score(y, pred, rng):
    ya = y.to_numpy(dtype=float)
    pa = pred.reindex(y.index).to_numpy(dtype=float)
    ok = ~np.isnan(ya) & ~np.isnan(pa)
    ya, pa = ya[ok], pa[ok]
    n = len(ya)
    acc = float((pa == ya).mean())
    base = float(max(ya.mean(), 1 - ya.mean()))
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ys, ps = ya[s], pa[s]
        d[b] = (ps == ys).mean() - max(ys.mean(), 1 - ys.mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n": n, "accuracy": acc, "baseline_constant": base,
            "delta_vs_constant": acc - base,
            "delta_ci95": [float(lo), float(hi)], "passes": bool(lo > 0)}


def oof_predict(X, y, folds, make_est):
    cases = y.index
    Xa = np.nan_to_num(X.reindex(cases).to_numpy(dtype=float))
    ya = y.to_numpy(dtype=float)
    fa = folds.reindex(cases).to_numpy(dtype=float)
    p = pd.Series(np.nan, index=cases, dtype=float)
    for te in sorted(set(fa[~np.isnan(fa)])):
        tr, ts = fa != te, fa == te
        if tr.sum() == 0 or ts.sum() == 0 or len(set(ya[tr])) < 2:
            continue
        est = make_est()
        est.fit(Xa[tr], ya[tr])
        p.iloc[np.where(ts)[0]] = est.predict(Xa[ts])
    return p
