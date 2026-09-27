"""Score the frozen rag_heads_v1 loop_length head on locked-47 once, then
(separately) measure the pooled 186+45 LOAO the retrained model would carry.

  python scripts/score_loop_length_locked.py score     # the locked read, frozen bundle
  python scripts/score_loop_length_locked.py pooled    # LOAO on dev+locked, for the refit

Preregistration: artifacts/experiments/loop_length_audit_20260928/locked_preregistration.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import audit_loop_length_loao as A  # noqa: E402

LOCKED_FEAT = ROOT / "artifacts/experiments/locked_consumed_20260924/features_locked"
BUNDLE = ROOT / "artifacts/models/rag_heads_v1/bundle.joblib"
BUNDLE_SHA = "8c66fbb940217697ae3cb3b1b4473c384edf178428450e33add8d1665e3dae72"
OUT = A.OUT


def locked_data():
    man = pd.read_csv(A.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    lk = man[man.development_fold.isna()]
    assert len(lk) == 47
    ix = pd.read_csv(LOCKED_FEAT / "index.csv", dtype={"exam_case_id": str})
    assert set(ix.exam_case_id) == set(lk.index)
    F = {p: np.load(LOCKED_FEAT / ("features_%s.npy" % p)).astype(np.float32)
         for p in A.POOLINGS}
    num = pd.to_numeric(lk.loop_length, errors="coerce").dropna()
    y = (num > A.UPPER).astype(int)
    return lk, ix, F, y, num


def cmd_score():
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert sha == BUNDLE_SHA, "bundle changed since preregistration"
    b = joblib.load(BUNDLE)
    spec = b["fields"]["loop_length"]
    lk, ix, F, y, num = locked_data()
    img_case = ix.exam_case_id.to_numpy()
    per_pool = [spec["heads"][p].predict_proba(F[p])[:, 1] for p in b["poolings"]]
    # same order as rag_inference: per pooling mean over the case's images, then mean
    prob = pd.DataFrame(np.stack(per_pool, 1), index=img_case).groupby(level=0).mean().mean(1)
    margin = float(spec["abstain_margin"])
    df = pd.DataFrame(dict(archive=lk.archive.reindex(y.index), y_true=y,
                           loop_value=num, probability=prob.reindex(y.index)))
    df["prediction"] = (df.probability >= 0.5).astype(int)
    df["abstain_margin"] = margin
    df["abstain"] = (df.probability - 0.5).abs() < margin
    df.index.name = "exam_case_id"
    df.to_csv(OUT / "locked_per_case_predictions.csv")
    back = pd.read_csv(OUT / "locked_per_case_predictions.csv",
                       dtype={"exam_case_id": str}).set_index("exam_case_id")
    m = A.metrics(back)
    yy, hh = back.y_true.to_numpy(), back.prediction.to_numpy()
    rng = np.random.default_rng(A.SEED)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yy), len(yy))
        if len(set(yy[i])) == 2:
            bs.append(balanced_accuracy_score(yy[i], hh[i]))
    obs = m["full_coverage"]["balanced_accuracy"]
    perm = np.array([balanced_accuracy_score(rng.permutation(yy), hh)
                     for _ in range(1000)])
    ab = back.abstain.to_numpy()
    sel_obs = m["selective"].get("balanced_accuracy", np.nan)
    perm_sel = []
    for _ in range(1000):
        yp = rng.permutation(yy)
        if len(set(yp[~ab])) == 2:
            perm_sel.append(balanced_accuracy_score(yp[~ab], hh[~ab]))
    perm_sel = np.array(perm_sel)
    # device distance, informational only
    ref = b["reference"]["cls"]
    dist = float(np.linalg.norm((F["cls"].mean(0) - ref["mean"]) / ref["std"])
                 / np.sqrt(F["cls"].shape[1]))
    fc, sel, r = m["full_coverage"], m["selective"], m["abstain_rate_by_true_class"]
    g = {"1_full_BA>=0.65_recalls>=0.50": fc["balanced_accuracy"] >= .65
         and min(fc["recall_0"], fc["recall_1"]) >= .5,
         "2_perm_p<0.05": float((1 + (perm >= obs).sum()) / 1001) < .05,
         "3_sel_cov>=0.50_BA>=0.75_recalls>=0.65": sel["coverage"] >= .5
         and sel.get("balanced_accuracy", 0) >= .75
         and min(sel.get("recall_0", 0), sel.get("recall_1", 0)) >= .65,
         "4_abstain_class_gap<=0.15": abs(r["0"] - r["1"]) <= .15}
    out = dict(bundle_sha256=sha, n_cases=int(len(back)), prevalence=round(float(yy.mean()), 4),
               metrics=m,
               full_ba_ci95=[round(float(np.percentile(bs, 2.5)), 4),
                             round(float(np.percentile(bs, 97.5)), 4)],
               permutation_full=dict(mean=round(float(perm.mean()), 4),
                                     q95=round(float(np.quantile(perm, .95)), 4),
                                     p=round(float((1 + (perm >= obs).sum()) / 1001), 5)),
               permutation_selective=dict(mean=round(float(perm_sel.mean()), 4),
                                          q95=round(float(np.quantile(perm_sel, .95)), 4),
                                          p=round(float((1 + (perm_sel >= sel_obs).sum())
                                                        / (1 + len(perm_sel))), 5)),
               device_distance_informational=round(dist, 4),
               device_null_q99=round(float(b["device"]["null_q99"]), 4),
               gates=g, all_gates_pass=all(g.values()),
               this_run_read_locked=True, refitted_anything=False)
    (OUT / "locked_metrics.json").write_text(json.dumps(out, indent=1, default=bool),
                                             encoding="utf-8")
    print(json.dumps(out, indent=1, default=bool))


def cmd_pooled():
    """LOAO over dev+locked pooled (231 cases). This is the internal estimate the
    refitted model carries once locked is folded into training."""
    cases, ix, F = A.load()
    lk, lix, LF, ly, lnum = locked_data()
    lcases = pd.DataFrame(dict(y=ly, loop_value=lnum,
                               archive=lk.archive.reindex(ly.index),
                               dev_fold=lk.fold.reindex(ly.index)))
    keep = lix.exam_case_id.isin(ly.index).to_numpy()
    allc = pd.concat([cases, lcases]).sort_index()
    ixa = pd.concat([ix, lix[keep]], ignore_index=True)
    Fa = {p: np.concatenate([F[p], LF[p][keep]]) for p in A.POOLINGS}
    # inner folds for the abstain margin: dev keeps development_fold; locked
    # cases are spread over the same 5 folds by a fixed seeded draw
    rng = np.random.default_rng(A.SEED)
    lf = pd.Series(rng.integers(0, 5, len(lcases)), index=lcases.index)
    folds = sorted(cases.dev_fold.unique())
    allc.loc[lcases.index, "dev_fold"] = [folds[i] for i in lf]
    df = A.loao(allc, ixa, Fa, allc.y)
    df = df.join(allc[["loop_value"]])
    df["source"] = np.where(df.index.isin(lcases.index), "locked47", "dev186")
    df.to_csv(OUT / "pooled231_per_case_predictions.csv")
    back = pd.read_csv(OUT / "pooled231_per_case_predictions.csv",
                       dtype={"exam_case_id": str}).set_index("exam_case_id")
    m = A.metrics(back)
    yy, hh = back.y_true.to_numpy(), back.prediction.to_numpy()
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yy), len(yy))
        bs.append(balanced_accuracy_score(yy[i], hh[i]))
    out = dict(n_cases=int(len(back)), metrics=m,
               full_ba_ci95=[round(float(np.percentile(bs, 2.5)), 4),
                             round(float(np.percentile(bs, 97.5)), 4)],
               by_source={s: A.metrics(g) for s, g in back.groupby("source")})
    (OUT / "pooled231_metrics.json").write_text(json.dumps(out, indent=1, default=bool),
                                                encoding="utf-8")
    print(json.dumps(dict(n=out["n_cases"], full=m["full_coverage"], sel=m["selective"],
                          outcome=m["all_cases_outcome"], ci=out["full_ba_ci95"],
                          abstain=m["abstain_rate_by_true_class"],
                          per_archive={a: (v["full_coverage"]["balanced_accuracy"],
                                           v["coverage"],
                                           v["selective"].get("balanced_accuracy"))
                                       for a, v in m["per_archive"].items()}),
                     indent=1, default=bool))


if __name__ == "__main__":
    {"score": cmd_score, "pooled": cmd_pooled}[sys.argv[1]]()
