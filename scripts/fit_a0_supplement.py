"""Fit the eight supplementary A0 heads the final report needs beyond rag_heads_v1.

rag_heads_v1 (frozen, sha 8c66fbb9...) holds 7 binary heads. The final report
prints every one of its 20 rows, so the remaining image-modelled fields get
heads here, with the exact recipe of scripts/score_all_fields_locked47.py:
A0 features (DINOv2-B/14, 518x686), 5 poolings, StandardScaler -> PCA64 ->
LogReg C=0.03, fitted on the 186 development cases only, class set = the
declared development classes, probabilities averaged over poolings and then
over the case's images, argmax. No threshold, no abstention, no selection:
this is packaging of an already-evaluated fixed method, not a new experiment.

Output: artifacts/models/a0_supplement_v1/bundle.joblib (+ fit_report.json)

    python scripts/fit_a0_supplement.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_field_matrix_three_arms as R  # noqa: E402

OUT = ROOT / "artifacts" / "models" / "a0_supplement_v1"
V1_SHA = "8c66fbb940217697ae3cb3b1b4473c384edf178428450e33add8d1665e3dae72"
FIELDS = ["capillary_count", "afferent_diameter", "efferent_diameter",
          "apex_diameter", "crossing_ratio", "flow_state", "rbc_aggregation",
          "papilla"]


def main():
    v1 = ROOT / "artifacts" / "models" / "rag_heads_v1" / "bundle.joblib"
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == V1_SHA
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    assert len(dev) == 186 and len(locked) == 47
    ix, feats, _ = R.load_arm(R.ANCHOR)
    assert not (set(ix.exam_case_id) & locked)
    img = ix.exam_case_id.to_numpy()
    fields, report = {}, {}
    for f in FIELDS:
        y, k, _ = R.target(dev, f)
        y = y.astype(int)
        y = y[y.index.isin(set(img))]
        classes = sorted(int(c) for c in y.unique())
        m = np.isin(img, y.index.to_numpy())
        ytr = y.reindex(img[m]).to_numpy()
        heads = {}
        for p in R.POOLINGS:
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components=R.DIM_FIXED, random_state=R.SEED),
                                 LogisticRegression(C=R.C_FIXED, max_iter=4000))
            heads[p] = pipe.fit(feats[p][m], ytr)
            assert [int(c) for c in pipe.classes_] == classes, (f, p)
        mode = int(y.mode()[0])
        fields[f] = dict(heads=heads, classes=classes, dev_mode=mode)
        report[f] = dict(n=int(len(y)), classes=classes, dev_mode=mode,
                         class_counts={int(c): int(v) for c, v in
                                       y.value_counts().sort_index().items()})
        print(f, report[f], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = dict(version="a0_supplement_v1", base_bundle_sha256=V1_SHA,
                  poolings=list(R.POOLINGS), fields=fields,
                  recipe="A0: 5 poolings, StandardScaler->PCA64->LogReg C=0.03, "
                         "dev 186 only, mean prob over poolings then images, argmax")
    joblib.dump(bundle, OUT / "bundle.joblib")
    sha = hashlib.sha256((OUT / "bundle.joblib").read_bytes()).hexdigest()
    (OUT / "fit_report.json").write_text(json.dumps(dict(
        fields=report, sha256=sha, locked_cases_seen=0,
        note="development-only fit of the recipe evaluated in "
             "artifacts/experiments/test_locked47_all_fields"), indent=1,
        ensure_ascii=False), encoding="utf-8")
    print("sha256", sha)


if __name__ == "__main__":
    main()
