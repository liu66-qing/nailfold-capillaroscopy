"""Fit the final field-expert router on every labelled case of the 233.

Same recipe as eval_calibrated_router.py, nothing re-chosen:
  p = mean of four expert heads (A0 image-level DINOv2 x5 poolings; COL colour /
      sharpness; SEG S0 segmenter geometry; DET Capillary-Dataset detector),
      each LogReg C=0.03, one head per binary target
  q = 1-D Platt calibration of logit(p), fitted on the 5-fold OOF p already
      written by eval_fixed_router.py (fixed_router_oof.csv, column p_all), so the
      calibrator never sees an in-sample p
The report decision is q > 0.5; the confidence wording uses |q - 0.5|.

Output: artifacts/models/expert_router_v1/{bundle.joblib,metadata.json}

    python scripts/fit_expert_router_final.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_calibrated_router as K  # noqa: E402
import eval_field_experts as E  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

OUT = os.path.join(ROOT, "artifacts/models/expert_router_v1")
OOF = os.path.join(E.OUT, "fixed_router_oof.csv")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_case_head(X, cases, y):
    Xtr = X.reindex(sorted(cases))
    keep = list(Xtr.columns[Xtr.notna().mean() > 0.9])
    Xtr = Xtr[keep]
    med = Xtr.median()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=R.C_FIXED, max_iter=4000))
    pipe.fit(Xtr.fillna(med).to_numpy(), y.reindex(Xtr.index).to_numpy())
    return dict(columns=keep, median=med.to_numpy(), pipe=pipe)


def fit_a0(Fa, img_case, y):
    m = np.isin(img_case, y.index.to_numpy())
    yi = y.reindex(img_case[m]).to_numpy().astype(int)
    heads = {}
    for p in R.POOLINGS:
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=R.DIM_FIXED, random_state=R.SEED),
                             LogisticRegression(C=R.C_FIXED, max_iter=4000))
        heads[p] = pipe.fit(Fa[p][m], yi)
    return heads


def main():
    man, ixa, Fa, source, fold = C.load()
    routes = E.load_case_routes(ixa)
    cases = set(ixa.exam_case_id)
    img_case = ixa.exam_case_id.to_numpy()
    T = E.build_targets(man)
    oof = pd.read_csv(OOF, dtype={"exam_case_id": str})
    bundle = dict(recipe="mean(A0,COL,SEG,DET) + Platt(logit p)", poolings=list(R.POOLINGS),
                  route_columns={k: list(v.columns) for k, v in routes.items()},
                  targets={})
    meta = {}
    for t in K.BIN:
        y = T[t][T[t].index.isin(cases)].astype(int)
        heads = dict(A0=fit_a0(Fa, img_case, y))
        for k in ("COL", "SEG", "DET"):
            heads[k] = fit_case_head(routes[k], set(y.index), y)
        o = oof[oof.target == t].set_index("exam_case_id")
        assert set(o.index) == set(y.index), t
        cal = LogisticRegression(C=1e4, max_iter=1000).fit(
            K.logit(o.p_all.to_numpy())[:, None], y.reindex(o.index).to_numpy())
        bundle["targets"][t] = dict(heads=heads, cal_coef=float(cal.coef_[0, 0]),
                                    cal_intercept=float(cal.intercept_[0]),
                                    prevalence=float(y.mean()), n=int(len(y)))
        meta[t] = dict(n=int(len(y)), positives=int(y.sum()),
                       cal=[round(float(cal.coef_[0, 0]), 4), round(float(cal.intercept_[0]), 4)])
        print(t, meta[t], flush=True)
    os.makedirs(OUT, exist_ok=True)
    bp = os.path.join(OUT, "bundle.joblib")
    joblib.dump(bundle, bp, compress=3)
    extract = dict(
        A0=dict(encoder_sha256="55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
                note="rag_heads_v1 encoder/preprocessing, 5 poolings"),
        COL=dict(recipe="extract_roi_quality_background_features.features (512x384, 6 regions)"),
        SEG=dict(weights="E:/nailfold_tmp/seg_runs/S0/weights/best.pt",
                 sha256=sha("E:/nailfold_tmp/seg_runs/S0/weights/best.pt"), conf=0.05, imgsz=1024,
                 recipe="extract_instance_geometry.polygon_features/pairwise_topology/image_row, case mean"),
        DET=dict(weights="artifacts/experiments/rescue_external_20260922/detectors/external_pretrain/weights/best.pt",
                 sha256=sha(os.path.join(ROOT, "artifacts/experiments/rescue_external_20260922/detectors/external_pretrain/weights/best.pt")),
                 conf=0.25, imgsz=640, class_map={0: "bushy", 1: "crossing", 2: "hairpin", 3: "tortuous"},
                 recipe="extract_detector_local_features.per_image_stats + aggregate"))
    json.dump(dict(bundle_sha256=sha(bp), cases=len(cases), images=int(len(ixa)),
                   extractors=extract, targets=meta, protocol=__doc__),
              open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("bundle", sha(bp))


if __name__ == "__main__":
    main()
