"""Fit the shipped per-field heads for the RAG input contract on all 186
development cases, and freeze everything inference needs into one bundle.

Fields (7), all binary, all on the A0 anchor features (DINOv2-B/14, 518x686,
5 poolings, StandardScaler -> PCA64 -> LogReg C=0.03, mean over poolings, then
mean over the case's images):
  answered always : clarity, subpapillary_venous_plexus, exudation
  with abstention : loop_length (>250, the report's printed upper normal),
                    blood_color, microthrombus, malformation_ratio

Abstention threshold per field = 40th percentile of |p - 0.5| on the 5-fold
development OOF (so ~60% of development cases are answered). This is the same
rule that was validated leave-one-archive-out with thresholds fitted on the
training archives only.

Device statistics: per-pooling per-dimension mean/std of the development image
features, used by inference to re-centre a new device's features, plus the
null distribution of the batch-centroid distance on development batches, used
to decide whether a batch comes from a different device at all.

Default: locked-47 is never loaded. Output: artifacts/models/rag_heads_v1/bundle.joblib

--include-locked (user-authorised 2026-09-28, after the one frozen locked read in
scripts/score_loop_length_locked.py): fits on dev 186 + locked 47 using the
locked features extracted 2026-09-24 with the same encoder, and writes
rag_heads_v2. loop_length is fitted but withheld from the RAG output because
it failed preregistered locked gate 3; its prediction is kept in the shadow log
only. After this there is no untouched internal test set; the next
newly collected cases are the test set (see the fit report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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

OUT = ROOT / "artifacts" / "models" / "rag_heads_v1"
LOCKED_FEAT = (ROOT / "artifacts" / "experiments" / "locked_consumed_20260924"
               / "features_locked")
ALWAYS = ["clarity", "subpapillary_venous_plexus", "exudation"]
ABSTAIN = ["loop_length", "blood_color", "microthrombus", "malformation_ratio"]
KEEP_FRACTION = 0.60
LOOP_UPPER = 250.0
DEVICE_CASES = 12          # whole cases per batch for the device-distance null
N_NULL = 500
# the product-facing wording; never a number, never a micrometre
LABELS = {
    "clarity": ("清晰", "不清/模糊"),
    "subpapillary_venous_plexus": ("不见", "可见"),
    "exudation": ("无", "有"),
    "loop_length": ("不偏长", "偏长(相对本系统参考人群)"),
    "blood_color": ("浅红/淡红", "暗红/暗紫"),
    "microthrombus": ("无", "有"),
    "malformation_ratio": ("畸形不多(<=10%档)", "畸形偏多(>10%档)"),
}
# what the RAG layer may do with each field (user's field-role table, 2026-09-28)
ROLES = {
    "clarity": "quality_gate",
    "subpapillary_venous_plexus": "research_candidate",
    "exudation": "research_candidate",
    "blood_color": "research_candidate",
    "malformation_ratio": "research_candidate_locked_ci_crosses_zero",
    "microthrombus": "static_correlation_only_not_for_advice",
    "loop_length": "report_band_correlation_hint",
}


def target(dev, field):
    if field == "loop_length":
        v = pd.to_numeric(R.clean_column(dev, field), errors="coerce").dropna()
        return (v > LOOP_UPPER).astype(float)
    y, k, _ = R.target(dev, field)
    assert k == 2, field
    return y


def fit_heads(feats, ix, y):
    m = ix.exam_case_id.isin(y.index).to_numpy()
    ytr = y.reindex(ix.exam_case_id[m]).to_numpy()
    heads = {}
    for p in R.POOLINGS:
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=R.DIM_FIXED, random_state=R.SEED),
                             LogisticRegression(C=R.C_FIXED, max_iter=4000))
        heads[p] = pipe.fit(feats[p][m], ytr)
    return heads


def main():
    include_locked = "--include-locked" in sys.argv
    out_dir = OUT.with_name("rag_heads_v2") if include_locked else OUT
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    assert len(dev) == 186 and len(locked) == 47
    ix, feats, meta = R.load_arm(R.ANCHOR)
    assert not (set(ix.exam_case_id) & locked)
    folds = dev.development_fold.to_dict()
    fit_fields = ALWAYS + ABSTAIN
    if include_locked:
        lix = pd.read_csv(LOCKED_FEAT / "index.csv", dtype={"exam_case_id": str})
        assert set(lix.exam_case_id) == locked
        lf = {p: np.load(LOCKED_FEAT / ("features_%s.npy" % p)).astype(np.float32)
              for p in R.POOLINGS}
        ix = pd.concat([ix, lix], ignore_index=True)
        feats = {p: np.concatenate([feats[p], lf[p]]) for p in R.POOLINGS}
        # locked cases keep their own seeded 5-fold assignment for the inner OOF
        folds.update({c: float(v) for c, v in man.loc[sorted(locked), "fold"].items()})
        dev = man
        # loop_length is still fitted so the shadow log can score it on future
        # cases, but it is withheld from the RAG output (see bundle["withheld"])

    fields, report = {}, {}
    for f in fit_fields:
        y = target(dev, f)
        y = y[y.index.isin(set(ix.exam_case_id))]
        order = sorted(y.index)
        p, _ = R.oof(feats, ix, y, folds, order, False, [0.0, 1.0])
        thr = (float(np.quantile(np.abs(p - 0.5), 1 - KEEP_FRACTION))
               if f in ABSTAIN else 0.0)
        yy = y.reindex(order).to_numpy()
        keep = np.abs(p - 0.5) >= thr
        h = p >= 0.5
        report[f] = dict(n=len(order), prevalence=round(float(yy.mean()), 4),
                         abstain_margin=round(thr, 4),
                         oof_coverage=round(float(keep.mean()), 4),
                         oof_answered_accuracy=round(float((h[keep] == yy[keep]).mean()), 4))
        fields[f] = dict(heads=fit_heads(feats, ix, y), abstain_margin=thr,
                         labels=LABELS[f], role=ROLES[f])
        print(f, report[f], flush=True)

    ref = {p: dict(mean=feats[p].mean(0), std=feats[p].std(0) + 1e-6)
           for p in R.POOLINGS}
    # null distribution of the standardised batch-centroid distance on
    # development batches of whole cases (a real device delivers whole cases,
    # and images inside a case are correlated, so a random-image null is too
    # tight and flags the development device itself as foreign)
    rng = np.random.default_rng(R.SEED)
    X = feats["cls"]
    cases = ix.exam_case_id.unique()
    null = []
    for _ in range(N_NULL):
        m = ix.exam_case_id.isin(rng.choice(cases, DEVICE_CASES, replace=False)).to_numpy()
        null.append(float(np.linalg.norm((X[m].mean(0) - ref["cls"]["mean"])
                                         / ref["cls"]["std"]) / np.sqrt(X.shape[1])))
    device = dict(cases=DEVICE_CASES, null_q99=float(np.quantile(null, 0.99)),
                  null_median=float(np.median(null)))
    print("device null", device)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = dict(version=out_dir.name, encoder=dict(
        model="vit_base_patch14_dinov2.lvd142m",
        weights="weights/dinov2/b/model.safetensors", resize=(518, 686),
        norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), topk=16),
        poolings=list(R.POOLINGS), fields=fields, reference=ref, device=device)
    if include_locked:
        bundle["withheld"] = {"loop_length": "failed preregistered locked gate 3 "
                              "(covered recall_0 0.615 < 0.65), 2026-09-28"}
    joblib.dump(bundle, out_dir / "bundle.joblib")
    note = ("dev 186 + locked 47 fit; no untouched internal test set remains. "
            "The next newly collected cases are the test set; oof numbers here "
            "are internal estimates, not a test result." if include_locked
            else "development-only fit; not a launch claim")
    (out_dir / "fit_report.json").write_text(json.dumps(dict(
        fields=report, device=device, n_cases=int(ix.exam_case_id.nunique()),
        locked_cases_seen=47 if include_locked else 0,
        keep_fraction=KEEP_FRACTION, loop_upper=LOOP_UPPER,
        withheld=bundle.get("withheld", {}), note=note), indent=1,
        ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
