"""Auditable leave-one-archive-out for loop_length > 250, written without the
shared oof() helper so every fit, threshold and assignment is visible here.

Protocol (preregistration.md in the output folder, written before first run):
  per direction: the held-out archive is removed whole; scaler, PCA64 and
  LogReg C=0.03 (one per pooling) are fitted on the other two archives'
  images only; the abstention margin is the 40th percentile of |p-0.5| on an
  inner 5-fold case-level OOF inside those two archives; the held-out archive
  is predicted once at the end. No recentring, no pooled statistics.

Every metric is recomputed from the single per-case prediction file.

  python scripts/audit_loop_length_loao.py run          # main + sensitivities
  python scripts/audit_loop_length_loao.py perm         # 2 x 1000 permutations
  python scripts/audit_loop_length_loao.py reconcile    # the 0.733 / 0.528 audit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "artifacts/experiments/medical_encoder_transfer_20260921/features/anchor_dinov2b_deployed"
LABELS = ROOT / "server_code_audit/locked_evaluation_v1_reviewed.csv"
INTEGRITY = ROOT / ".tmp_probe/image_integrity.csv"
OUT = ROOT / "artifacts/experiments/loop_length_audit_20260928"
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
C, DIM, SEED = 0.03, 64, 20260917
KEEP = 0.60
UPPER = 250.0
N_PERM = 1000


# ---------------------------------------------------------------- data
def load(threshold=UPPER):
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    assert len(dev) == 186 and len(locked) == 47
    ix = pd.read_csv(FEAT / "index.csv", dtype={"exam_case_id": str})
    assert not (set(ix.exam_case_id) & locked), "locked case in features"
    F = {p: np.load(FEAT / ("features_%s.npy" % p)).astype(np.float32)
         for p in POOLINGS}
    raw = dev.loop_length.map(lambda s: "" if pd.isna(s) else str(s).strip())
    num = pd.to_numeric(raw, errors="coerce").dropna()
    num = num[num.index.isin(set(ix.exam_case_id))]
    y = (num > threshold).astype(int)
    cases = pd.DataFrame(dict(y=y, loop_value=num,
                              archive=dev.archive.reindex(y.index),
                              dev_fold=dev.development_fold.reindex(y.index)))
    cases = cases.sort_index()
    keep = ix.exam_case_id.isin(cases.index).to_numpy()
    ix = ix[keep].reset_index(drop=True)
    F = {p: v[keep] for p, v in F.items()}
    return cases, ix, F


# ---------------------------------------------------------------- model
def fit_predict(F, img_case, y_img, train_mask, test_mask):
    """Fit one pipeline per pooling on train images, predict test images,
    mean over poolings, then mean over each case's images."""
    probs = []
    for p in POOLINGS:
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=DIM, random_state=SEED),
                             LogisticRegression(C=C, max_iter=4000))
        pipe.fit(F[p][train_mask], y_img[train_mask])
        probs.append(pipe.predict_proba(F[p][test_mask])[:, 1])
    return pd.Series(np.mean(probs, axis=0),
                     index=img_case[test_mask]).groupby(level=0).mean()


def loao(cases, ix, F, y_case, image_keep=None):
    """y_case: Series case -> label (the real one or a permutation).
    Returns the per-case prediction frame."""
    img_case = ix.exam_case_id.to_numpy()
    y_img = y_case.reindex(img_case).to_numpy()
    ok = np.ones(len(ix), bool) if image_keep is None else image_keep
    rows = []
    for a in sorted(cases.archive.unique()):
        test_cases = set(cases.index[cases.archive == a])
        train_cases = set(cases.index[cases.archive != a])
        tr = ok & np.isin(img_case, list(train_cases))
        te = ok & np.isin(img_case, list(test_cases))
        assert not (set(img_case[tr]) & set(img_case[te]))
        p_out = fit_predict(F, img_case, y_img, tr, te)
        # inner OOF inside the training archives only, by development_fold
        inner = pd.Series(np.nan, index=sorted(train_cases))
        tf = cases.dev_fold.reindex(sorted(train_cases))
        for fo in sorted(tf.unique()):
            itest = set(tf.index[tf == fo])
            itr = ok & np.isin(img_case, list(train_cases - itest))
            ite = ok & np.isin(img_case, list(itest))
            if ite.sum() == 0:
                continue
            s = fit_predict(F, img_case, y_img, itr, ite)
            inner.loc[s.index] = s.to_numpy()
        inner = inner.dropna()
        margin = float(np.quantile(np.abs(inner.to_numpy() - 0.5), 1 - KEEP))
        for c, p in p_out.items():
            rows.append(dict(exam_case_id=c, archive=a, y_true=int(y_case[c]),
                             probability=float(p), prediction=int(p >= 0.5),
                             abstain_margin=margin,
                             abstain=bool(abs(p - 0.5) < margin),
                             n_train_cases=len(train_cases),
                             n_train_images=int(tr.sum()),
                             n_test_images_this_case=int((te & (img_case == c)).sum())))
    return pd.DataFrame(rows).set_index("exam_case_id").sort_index()


# ---------------------------------------------------------------- metrics
def _block(y, h, p):
    out = dict(n=int(len(y)), n_pos=int(y.sum()))
    if len(y) == 0:
        return out
    out["accuracy"] = round(float((h == y).mean()), 4)
    if len(set(y)) == 2:
        out["balanced_accuracy"] = round(float(balanced_accuracy_score(y, h)), 4)
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
        out["recall_0"] = round(float((h[y == 0] == 0).mean()), 4)
        out["recall_1"] = round(float((h[y == 1] == 1).mean()), 4)
        out["majority_baseline_accuracy"] = round(float(max(y.mean(), 1 - y.mean())), 4)
    cm = confusion_matrix(y, h, labels=[0, 1])
    out["confusion_rows_true_cols_pred"] = cm.tolist()
    return out


def metrics(df):
    """Computed only from the per-case frame."""
    y, h, p, ab = (df.y_true.to_numpy(), df.prediction.to_numpy(),
                   df.probability.to_numpy(), df.abstain.to_numpy())
    res = dict(full_coverage=_block(y, h, p),
               selective=_block(y[~ab], h[~ab], p[~ab]))
    res["selective"]["coverage"] = round(float((~ab).mean()), 4)
    res["all_cases_outcome"] = dict(
        answered_correct=round(float(((~ab) & (h == y)).mean()), 4),
        answered_wrong=round(float(((~ab) & (h != y)).mean()), 4),
        abstained=round(float(ab.mean()), 4))
    res["abstain_rate_by_true_class"] = {
        str(k): round(float(ab[y == k].mean()), 4) for k in (0, 1)}
    per = {}
    for a, g in df.groupby("archive"):
        gy, gh, gp, ga = (g.y_true.to_numpy(), g.prediction.to_numpy(),
                          g.probability.to_numpy(), g.abstain.to_numpy())
        per[a] = dict(full_coverage=_block(gy, gh, gp),
                      selective=_block(gy[~ga], gh[~ga], gp[~ga]),
                      coverage=round(float((~ga).mean()), 4),
                      abstain_margin=round(float(g.abstain_margin.iloc[0]), 4),
                      prevalence=round(float(gy.mean()), 4))
    res["per_archive"] = per
    return res


def gates(m, m_excl):
    fc, sel, per = m["full_coverage"], m["selective"], m["per_archive"]
    g = {}
    g["1_each_direction_full_BA>=0.65_and_recalls>=0.50"] = all(
        v["full_coverage"]["balanced_accuracy"] >= 0.65
        and min(v["full_coverage"]["recall_0"], v["full_coverage"]["recall_1"]) >= 0.50
        for v in per.values())
    g["3_selective_cov>=0.50_BA>=0.75_recalls>=0.65_each_dir_BA>=0.65"] = (
        sel["coverage"] >= 0.50 and sel["balanced_accuracy"] >= 0.75
        and min(sel["recall_0"], sel["recall_1"]) >= 0.65
        and all(v["selective"].get("balanced_accuracy", 0) >= 0.65
                for v in per.values()))
    r = m["abstain_rate_by_true_class"]
    g["4_abstain_class_gap<=0.15_and_archive_cov_in_[0.35,0.80]"] = (
        abs(r["0"] - r["1"]) <= 0.15
        and all(0.35 <= v["coverage"] <= 0.80 for v in per.values()))
    g["5_blank_image_exclusion_changes_full_BA<=0.03"] = abs(
        m_excl["full_coverage"]["balanced_accuracy"]
        - fc["balanced_accuracy"]) <= 0.03
    return g


# ---------------------------------------------------------------- commands
def cmd_run():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, ix, F = load()
    df = loao(cases, ix, F, cases.y)
    df = df.join(cases[["loop_value"]])
    df.to_csv(OUT / "per_case_predictions.csv")
    # metrics strictly from the file just written
    back = pd.read_csv(OUT / "per_case_predictions.csv",
                       dtype={"exam_case_id": str}).set_index("exam_case_id")
    m = metrics(back)
    integ = pd.read_csv(INTEGRITY, dtype={"exam_case_id": str})
    blank = ((integ.gstd < 3) | (integ.gmean < 8)).to_numpy()
    assert (integ.image_path.to_numpy() ==
            pd.read_csv(FEAT / "index.csv").image_path.to_numpy()).all()
    keep_img = ~blank[pd.read_csv(FEAT / "index.csv", dtype={"exam_case_id": str})
                      .exam_case_id.isin(cases.index).to_numpy()]
    df_ex = loao(cases, ix, F, cases.y, image_keep=keep_img)
    df_ex.to_csv(OUT / "per_case_predictions_blank_excluded.csv")
    m_ex = metrics(df_ex)
    c2, ix2, F2 = load(255.5)
    df_med = loao(c2, ix2, F2, c2.y)
    df_med.to_csv(OUT / "per_case_predictions_threshold_255.5.csv")
    out = dict(main=m, gates=gates(m, m_ex),
               sensitivity_blank_images_excluded=dict(
                   n_images_removed=int((~keep_img).sum()), metrics=m_ex),
               sensitivity_threshold_255_5=metrics(df_med),
               locked_cases_seen=0, recentring_used=False)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(main_full=m["full_coverage"], main_sel=m["selective"],
                          outcome=m["all_cases_outcome"],
                          abstain_by_class=m["abstain_rate_by_true_class"],
                          per_archive={a: (v["full_coverage"]["balanced_accuracy"],
                                           v["full_coverage"]["recall_0"],
                                           v["full_coverage"]["recall_1"],
                                           v["coverage"],
                                           v["selective"].get("balanced_accuracy"))
                                       for a, v in m["per_archive"].items()},
                          gates=out["gates"],
                          blank_excl_full_ba=m_ex["full_coverage"]["balanced_accuracy"],
                          thr255_full_ba=out["sensitivity_threshold_255_5"]
                          ["full_coverage"]["balanced_accuracy"]), indent=1))


def _perm_one(args):
    kind, k = args
    cases, ix, F = _G
    rng = np.random.default_rng(10_000 + k + (0 if kind == "within" else 500_000))
    y = cases.y.copy()
    if kind == "within":
        for a in cases.archive.unique():
            idx = cases.index[cases.archive == a]
            y[idx] = rng.permutation(y[idx].to_numpy())
    else:
        y[:] = rng.permutation(y.to_numpy())
    df = loao(cases, ix, F, y)
    m = metrics(df)
    return kind, k, m["full_coverage"]["balanced_accuracy"], \
        m["selective"].get("balanced_accuracy", np.nan), m["selective"]["coverage"]


_G = None


def _init():
    global _G
    _G = load()


def cmd_perm():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(kind, k) for kind in ("within", "global") for k in range(N_PERM)]
    rows = []
    with ProcessPoolExecutor(max_workers=28, initializer=_init) as ex:
        for i, r in enumerate(ex.map(_perm_one, jobs, chunksize=4)):
            rows.append(r)
            if i % 200 == 0:
                print(i, r, flush=True)
    d = pd.DataFrame(rows, columns=["kind", "k", "full_ba", "selective_ba",
                                    "selective_coverage"])
    d.to_csv(OUT / "permutations.csv", index=False)
    obs = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))["main"]
    res = {}
    for kind, g in d.groupby("kind"):
        res[kind] = {}
        for col, o in [("full_ba", obs["full_coverage"]["balanced_accuracy"]),
                       ("selective_ba", obs["selective"]["balanced_accuracy"])]:
            v = g[col].to_numpy()
            res[kind][col] = dict(observed=o, n=int(len(v)),
                                  perm_mean=round(float(np.nanmean(v)), 4),
                                  perm_q95=round(float(np.nanquantile(v, .95)), 4),
                                  perm_max=round(float(np.nanmax(v)), 4),
                                  p_empirical=round(float((1 + (v >= o).sum())
                                                          / (1 + len(v))), 5))
    (OUT / "permutation_summary.json").write_text(json.dumps(res, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(res, indent=1))


def cmd_reconcile():
    """Put the four historical numbers on one per-case table."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_field_matrix_three_arms as R
    import eval_measurement_relative as M
    cases, ix, F = load()
    main = pd.read_csv(OUT / "per_case_predictions.csv",
                       dtype={"exam_case_id": str}).set_index("exam_case_id")
    ixR, featsR, _ = R.load_arm(R.ANCHOR)
    y = cases.y.astype(float)
    order = sorted(y.index)
    yy = y.reindex(order).to_numpy()
    arch = cases.archive.reindex(order).to_numpy()
    rep = {}
    # (B) 5-fold development OOF, the source of "0.733 at >250"
    p_oof, _ = R.oof(featsR, ixR, y, cases.dev_fold.to_dict(), order, False, [0., 1.])
    rep["B_devfold_oof_thr250"] = dict(
        protocol="5-fold development_fold OOF across archives (NOT LOAO)",
        ba=round(float(balanced_accuracy_score(yy, p_oof >= .5)), 4))
    # (D) old probe LOAO via R.oof with a 2-fold dict, fixed assignment
    p_d = np.full(len(order), np.nan)
    per_arch_raw = {}
    for a in sorted(set(arch)):
        m = arch == a
        fd = {o: int(mm) for o, mm in zip(order, m)}
        q, _ = R.oof(featsR, ixR, y, fd, order, False, [0., 1.])
        p_d[m] = q[m]
        per_arch_raw[a] = q[m]
    diff_d = float(np.abs(p_d - main.probability.reindex(order).to_numpy()).max())
    rep["D_probe_loao_sorted"] = dict(
        protocol="R.oof with fold dict {held-out:1, rest:0}; refit per archive",
        ba=round(float(balanced_accuracy_score(yy, p_d >= .5)), 4),
        max_abs_prob_diff_vs_audit=diff_d,
        same_predicted_cases=bool(set(order) == set(main.index)))
    # (C) the bug: values of q[m] (sorted-case order) written into list(set) order
    bug_bas = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p_c = pd.Series(np.nan, index=order)
        for a in sorted(set(arch)):
            m = arch == a
            te = list(np.array(order)[m])
            scrambled = list(rng.permutation(te))   # stands in for list(set(te))
            p_c[scrambled] = per_arch_raw[a]
        bug_bas.append(round(float(balanced_accuracy_score(yy, p_c.to_numpy() >= .5)), 4))
    # the actual list(set) order depends on PYTHONHASHSEED; show it differs
    import subprocess
    orders = []
    for hs in ("0", "1", "2"):
        code = ("import pandas as pd;import sys;"
                "c=pd.read_csv(r'%s',dtype={'exam_case_id':str}).exam_case_id;"
                "print(','.join(list(set(c.head(40)))[:5]))") % (OUT / "per_case_predictions.csv")
        env = dict(os.environ, PYTHONHASHSEED=hs)
        orders.append(subprocess.run([sys.executable, "-c", code], env=env,
                                     capture_output=True, text=True).stdout.strip())
    same_multiset = all(
        np.allclose(np.sort(per_arch_raw[a]),
                    np.sort(main.probability[main.archive == a].to_numpy()), atol=1e-4)
        for a in per_arch_raw)
    rep["C_probe_loao_list_set_bug"] = dict(
        mechanism="loao[list(te)] = q[arch==a]: q is in sorted-case order, "
                  "list(set) is in hash order, so each archive's predictions were "
                  "written onto the wrong cases of the same archive",
        historical_ba=0.528,
        ba_under_5_random_within_archive_reassignments=bug_bas,
        within_archive_prediction_multiset_identical_to_audit=bool(same_multiset),
        list_set_first5_under_hashseed_0_1_2=orders,
        reading="same training rows, same features, same fitted models, same "
                "predicted cases; only the case<->prediction assignment differs")
    # (A) eval_measurement_relative at 255.5 (median) LOAO
    c2, _, _ = load(255.5)
    y2 = c2.y.astype(float)
    pa = pd.Series(np.nan, index=sorted(y2.index))
    for a in sorted(c2.archive.unique()):
        te = sorted(c2.index[c2.archive == a]); tr = sorted(c2.index[c2.archive != a])
        pa[te] = M.a0_fit(featsR, ixR, y2, set(tr), set(te)).reindex(te).to_numpy()
    med = pd.read_csv(OUT / "per_case_predictions_threshold_255.5.csv",
                      dtype={"exam_case_id": str}).set_index("exam_case_id")
    rep["A_eval_measurement_relative_loao_thr255.5"] = dict(
        protocol="refit LOAO, threshold 255.5 (median) - a different target",
        ba=round(float(balanced_accuracy_score(y2.reindex(pa.index), pa >= .5)), 4),
        max_abs_prob_diff_vs_audit_255=float(np.abs(
            pa.to_numpy() - med.probability.reindex(pa.index).to_numpy()).max()))
    rep["audit_main_thr250_full_ba"] = round(float(balanced_accuracy_score(
        main.y_true, main.prediction)), 4)
    (OUT / "reconciliation.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"run": cmd_run, "perm": cmd_perm, "reconcile": cmd_reconcile}[sys.argv[1]]()
