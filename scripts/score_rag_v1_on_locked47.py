"""Test-set scoring of the RAG field heads on locked-47 (user request: "现在就需要测试集，污染就污染吧").

Only rag_heads_v1 can be scored here: it was fitted on the 186 development
cases and never on locked-47. rag_heads_v2 trained on locked-47, so it has no
internal test set, and this script refuses to load it.

This is a contaminated test set. Locked-47 was read seven times before, and
model selection was done on it in three runs. Every number is therefore
descriptive and optimistic. This is not external or clinical validation, and
it is not the preregistered prospective evaluation, which stays untouched.

    python scripts/score_rag_v1_on_locked47.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fit_rag_heads as F  # noqa: E402
import run_field_matrix_three_arms as R  # noqa: E402

BUNDLE = ROOT / "artifacts/models/rag_heads_v1/bundle.joblib"
BUNDLE_SHA = "8c66fbb940217697ae3cb3b1b4473c384edf178428450e33add8d1665e3dae72"
LOCKED_FEAT = F.LOCKED_FEAT
OUT = ROOT / "artifacts/experiments/test_locked47_rag_heads_v1"
N_BOOT, N_PERM = 2000, 2000


def rec(y, h, c):
    m = y == c
    return float((h[m] == c).mean()) if m.any() else float("nan")


def block(y, h, p):
    out = dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    if len(y) == 0:
        return out
    out.update(accuracy=round(float((h == y).mean()), 4),
               recall_0=round(rec(y, h, 0), 4), recall_1=round(rec(y, h, 1), 4))
    if len(set(y)) == 2:
        out["balanced_accuracy"] = round(float(balanced_accuracy_score(y, h)), 4)
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def main():
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert sha == BUNDLE_SHA, "rag_heads_v1 bundle changed"
    b = joblib.load(BUNDLE)
    assert b["version"] == "rag_heads_v1"
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev, lk = man[man.development_fold.notna()], man[man.development_fold.isna()]
    assert len(dev) == 186 and len(lk) == 47
    ix = pd.read_csv(LOCKED_FEAT / "index.csv", dtype={"exam_case_id": str})
    assert set(ix.exam_case_id) == set(lk.index)
    feats = {p: np.load(LOCKED_FEAT / ("features_%s.npy" % p)).astype(np.float32)
             for p in b["poolings"]}
    img_case = ix.exam_case_id.to_numpy()
    rng = np.random.default_rng(R.SEED)

    res, rows = {}, []
    for f, spec in b["fields"].items():
        y = F.target(lk, f).astype(int)
        ydev = F.target(dev, f).astype(int)
        dev_mode = int(ydev.mean() >= 0.5)
        # same order as rag_inference: mean over the case's images per pooling, then mean
        per = np.stack([spec["heads"][p].predict_proba(feats[p])[:, 1]
                        for p in b["poolings"]], 1)
        prob = pd.DataFrame(per, index=img_case).groupby(level=0).mean().mean(axis=1)
        prob = prob.reindex(y.index)
        yy, pp = y.to_numpy(), prob.to_numpy()
        hh = (pp >= 0.5).astype(int)
        keep = np.abs(pp - 0.5) >= spec["abstain_margin"]
        full = block(yy, hh, pp)
        sel = block(yy[keep], hh[keep], pp[keep])
        sel["coverage"] = round(float(keep.mean()), 4)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(yy), len(yy))
            if len(set(yy[i])) == 2:
                bs.append(balanced_accuracy_score(yy[i], hh[i]))
        perm = np.array([balanced_accuracy_score(rng.permutation(yy), hh)
                         for _ in range(N_PERM)])
        obs = full.get("balanced_accuracy", np.nan)
        arch = {}
        for a in sorted(lk.archive.unique()):
            m = (lk.archive.reindex(y.index) == a).to_numpy()
            arch[a] = block(yy[m], hh[m], pp[m])
        res[f] = dict(
            role=spec.get("role"), labels=list(spec["labels"]),
            prevalence_locked=round(float(yy.mean()), 4),
            prevalence_dev=round(float(ydev.mean()), 4),
            baseline_dev_mode=dict(class_=dev_mode,
                                   accuracy=round(float((yy == dev_mode).mean()), 4),
                                   balanced_accuracy=0.5),
            full_coverage=full,
            full_ba_ci95=[round(float(np.percentile(bs, 2.5)), 4),
                          round(float(np.percentile(bs, 97.5)), 4)],
            permutation_p=round(float((1 + (perm >= obs).sum()) / (1 + N_PERM)), 5),
            selective=sel, abstain_margin=round(float(spec["abstain_margin"]), 4),
            per_archive=arch)
        for c, pr, t, k in zip(y.index, pp, yy, keep):
            rows.append(dict(exam_case_id=c, field=f, y_true=int(t),
                             probability=round(float(pr), 4),
                             prediction=int(pr >= 0.5), answered=bool(k)))
        print(f, json.dumps(dict(full=full, ci=res[f]["full_ba_ci95"],
                                 p=res[f]["permutation_p"], sel=sel),
                            ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "per_case_predictions.csv", index=False)
    (OUT / "test_metrics.json").write_text(json.dumps(dict(
        evaluation_name="contaminated internal holdout test (locked-47)",
        model="rag_heads_v1", bundle_sha256=sha, trained_on="dev 186 only",
        test_cases=47,
        contamination=("locked-47 read 7 times before; model selection on it in 3 runs; "
                       "numbers are descriptive and optimistic"),
        not_this=["external validation", "clinical validation",
                  "the preregistered prospective evaluation (untouched)",
                  "a test of rag_heads_v2 (v2 trained on these cases)"],
        fields=res), indent=1, ensure_ascii=False, default=bool), encoding="utf-8")


if __name__ == "__main__":
    main()
