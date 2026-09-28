"""Implementation check for the final release: on the saved locked-47 A0
features, the release predictor must reproduce the predictions that were
already scored (no new evaluation, nothing is selected here).

  v1 heads   vs artifacts/experiments/test_locked47_rag_heads_v1/per_case_predictions.csv
  supp heads vs artifacts/experiments/test_locked47_all_fields/per_case.csv

    python scripts/check_final_reproduction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from nailfold_report.final_inference import fields_from_features, load_bundles  # noqa: E402
from nailfold_report.final_registry import FIELDS  # noqa: E402

FEAT = ROOT / "artifacts/experiments/locked_consumed_20260924/features_locked"
POOL = ["mean", "topk_mean", "max", "cls", "std"]


def main() -> int:
    v1, supp = load_bundles(ROOT)
    ix = pd.read_csv(FEAT / "index.csv", dtype={"exam_case_id": str})
    F = {p: np.load(FEAT / ("features_%s.npy" % p)).astype(np.float32) for p in POOL}
    ref_v1 = pd.read_csv(ROOT / "artifacts/experiments/test_locked47_rag_heads_v1/"
                         "per_case_predictions.csv", dtype={"exam_case_id": str})
    ref_all = pd.read_csv(ROOT / "artifacts/experiments/test_locked47_all_fields/"
                          "per_case.csv", dtype={"exam_case_id": str})
    ours = {}
    for case, g in ix.groupby("exam_case_id"):
        rows = g.index.to_numpy()
        fields, audit = fields_from_features(v1, supp, {p: F[p][rows] for p in POOL})
        assert len(fields) == 20 and all(v["value"] for v in fields.values())
        ours[case] = (fields, audit)
    res, bad = {}, 0
    for f, _n, src, _w, _l in FIELDS:
        if src == "model_v1":
            r = ref_v1[ref_v1.field == f]
            want = (r.probability >= 0.5).astype(int).to_numpy()
            p_ref = r.probability.to_numpy()
            p_our = np.array([ours[c][1][f]["scores"][1] for c in r.exam_case_id])
            extra = dict(max_abs_prob_diff=float(np.abs(p_ref - p_our).max()))
        elif src == "model_supp":
            r = ref_all[ref_all.field == f]
            want = r.pred.astype(int).to_numpy()
            extra = {}
        else:
            continue
        got = np.array([ours[c][0][f]["class_id"] for c in r.exam_case_id])
        n_diff = int((got != want).sum())
        bad += n_diff
        res[f] = dict(source=src, n=int(len(r)), mismatches=n_diff, **extra)
    print(json.dumps(res, indent=1))
    print("cases", len(ours), "total mismatches", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
