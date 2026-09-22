"""All 15 in-scope fields x 3 encoder arms, under the SHIPPED fixed configuration.

Every arm runs the same recipe: five poolings, C fixed at 0.03, PCA 64, seed
20260917, the same development_fold case-level OOF and the same scoring code.
Nothing is selected per field -- not C, not the pooling set, not the threshold,
not the arm. Round one's in-fold C selection is deliberately NOT used, because a
gain that only exists under a tuned readout is not a reason to change deployment.

Targets come from build_field_matrix.ADMISSIBLE, which was written and confirmed
before any of these numbers existed, so no field can be given a target after its
result is seen.

Two reporting notes, neither of which changes the configuration:
  * the shipped recipe decides a multiclass label by majority vote over poolings,
    which yields no score, so macro one-vs-rest AUROC is taken from the mean of
    the pooling probability matrices. The classification metrics still come from
    the shipped vote.
  * flow_state and microthrombus are defined on a time base. They are evaluated
    so that they are not silently written off, but any signal they show is a
    static appearance correlate, not an observation of flow or of events/minute.

The four measurement fields are banded by the report's own printed normal range.
A banded result is NOT a micrometre measurement capability and must never be
reported as one; absolute micron output stays forbidden (device calibration is
UNCALIBRATED). capillary_count is already a printed band, so no 条/mm is emitted.

DEVELOPMENT ONLY. locked-47 is never loaded; the run asserts locked_cases_seen=0.

  PYTHONIOENCODING=utf-8 python scripts/run_field_matrix_three_arms.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_field_matrix import (ADMISSIBLE, DIRTY, IN_SCOPE, NORMAL_RANGE,
                                NUMERIC_FIELDS, UNIT)          # noqa: E402

EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"

# the delivered configuration's own constants, copied verbatim so nothing is tuned
C_FIXED, DIM_FIXED, SEED, N_BOOT = 0.03, 64, 20260917, 2000
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
ARMS = {"anchor_dinov2b_deployed": "A  general vision, ViT-B/14, deployed geometry",
        "dinov2l_deployed_geometry": "A  general vision, ViT-L/14, deployed geometry",
        "biomedclip_medical": "B  medical pretraining, BiomedCLIP ViT-B/16 @224"}
ANCHOR = "anchor_dinov2b_deployed"
NEAR_THRESHOLD_FRAC = 0.10   # within 10% of the band edge counts as borderline
MIN_CLASS_FOR_CI = 10


def clean_column(dev: pd.DataFrame, field: str) -> pd.Series:
    """Same cleaning as the confirmed field matrix: arrow-backed columns leave
    NaN floats behind under astype(str), so coerce explicitly."""
    raw = dev[field].map(lambda s: "" if pd.isna(s) else str(s).strip())
    return raw[~raw.map(lambda s: bool(DIRTY.match(s)))]


def target(dev: pd.DataFrame, field: str):
    """Declared-up-front mapping only. Returns (y, n_classes, extra_report)."""
    clean = clean_column(dev, field)
    _, mapping = ADMISSIBLE[field]
    if mapping is not None:
        m = {v: float(k) for k, vs in mapping.items() for v in vs}
        y = clean.map(m).dropna()
        return y, int(y.nunique()), {}
    # the four measurement fields: band by the report's own printed normal range
    num = pd.to_numeric(clean, errors="coerce").dropna()
    lo, hi = NORMAL_RANGE[field]
    y = pd.Series(np.where(num < lo, 0.0, np.where(num > hi, 2.0, 1.0)),
                  index=num.index)
    span = hi - lo
    near = ((num - lo).abs() <= NEAR_THRESHOLD_FRAC * span) | \
           ((num - hi).abs() <= NEAR_THRESHOLD_FRAC * span)
    extra = dict(
        printed_normal_range=[lo, hi],
        raw_value_distribution=dict(
            n=int(len(num)), distinct=int(num.nunique()),
            min=round(float(num.min()), 3), q25=round(float(num.quantile(.25)), 3),
            median=round(float(num.median()), 3),
            q75=round(float(num.quantile(.75)), 3),
            max=round(float(num.max()), 3),
            sorted_distinct=[round(float(v), 3) for v in sorted(num.unique())]),
        band_sizes={str(int(k)): int(v)
                    for k, v in y.value_counts().sort_index().items()},
        near_threshold_fraction=round(float(near.mean()), 4),
        near_threshold_definition="within %.0f%% of the band width (%.1f) of "
                                  "either printed edge" % (
                                      NEAR_THRESHOLD_FRAC * 100,
                                      NEAR_THRESHOLD_FRAC * span),
        banded_result_is_not_a_micron_measurement=True)
    return y, int(y.nunique()), extra


def load_arm(arm: str):
    d = EXP / "features" / arm
    ix = pd.read_csv(d / "index.csv", dtype={"exam_case_id": str})
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return ix, {p: np.load(d / ("features_%s.npy" % p)).astype(np.float32)
                for p in POOLINGS}, meta


def fit_predict(Xtr, ytr, Xts, multiclass, classes):
    """The shipped head. For multiclass a full probability matrix over the
    declared class set is returned as well, so macro AUROC can be taken without
    changing how the shipped vote decides the label."""
    if len(set(ytr)) < 2:
        lab = float(ytr[0])
        proba = np.zeros((len(Xts), len(classes)))
        proba[:, classes.index(lab)] = 1.0
        return np.full(len(Xts), lab), proba
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    pr = pipe.predict_proba(Xts)
    full = np.zeros((len(Xts), len(classes)))
    for j, c in enumerate(pipe.classes_):
        full[:, classes.index(float(c))] = pr[:, j]
    if multiclass:
        return pipe.predict(Xts).astype(float), full
    return pr[:, 1], full


def oof(feats, ix, y_case, folds, order, multiclass, classes):
    """Exactly the shipped aggregation: probabilities averaged over poolings, or
    a majority vote over poolings for a multiclass field. The averaged pooling
    probability matrix is carried alongside for AUROC only."""
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    prob = np.full((len(order), len(classes)), np.nan)
    fs = pd.Series([folds.get(c, np.nan) for c in order], index=order)
    for fo in sorted(fs.dropna().unique()):
        te = set(fs.index[fs == fo])
        trm = dm & ix.exam_case_id.isin(set(order) - te).to_numpy()
        tem = dm & ix.exam_case_id.isin(te).to_numpy()
        if trm.sum() == 0 or tem.sum() == 0:
            continue
        ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
        ids = ix.exam_case_id[tem].to_numpy()
        got = [fit_predict(feats[p][trm], ytr, feats[p][tem], multiclass, classes)
               for p in POOLINGS]
        pm = pd.DataFrame(np.mean([g[1] for g in got], axis=0),
                          index=ids).groupby(level=0).mean()
        for c, row in pm.iterrows():
            if c in pos:
                prob[pos[c]] = row.to_numpy()
        if multiclass:
            df = pd.DataFrame(np.stack([g[0] for g in got]).T, index=ids)
            for c, grp in df.groupby(level=0):
                vals, cnt = np.unique(grp.to_numpy().ravel(), return_counts=True)
                if c in pos:
                    out[pos[c]] = vals[cnt.argmax()]
        else:
            s = pd.Series(np.mean([g[0] for g in got], axis=0),
                          index=ids).groupby(level=0).mean()
            for c, v in s.items():
                if c in pos:
                    out[pos[c]] = v
    return out, prob


def hard(p, multiclass):
    return p if multiclass else (p >= 0.5).astype(float)


def boot_ci(fn, y, *args, rng=None):
    """Percentile bootstrap over cases. Used for accuracy, delta and BA so every
    headline number carries a CI rather than only the paired contrast."""
    obs = fn(y, *args)
    vals, n = [], len(y)
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(fn(y[s], *[a[s] for a in args]))
    v = np.asarray(vals)
    return dict(value=round(float(obs), 4),
                ci=[round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)])


def score(y, p, prob, multiclass, classes, rng):
    yp = hard(p, multiclass)
    vals, cnt = np.unique(y, return_counts=True)
    const = float(vals[cnt.argmax()])
    labs = [float(v) for v in vals]
    rec = recall_score(y, yp, labels=labs, average=None, zero_division=0)
    acc = boot_ci(lambda t, q: float((hard(q, multiclass) == t).mean()),
                  y, p, rng=rng)
    # the baseline is ONE constant answer -- the development mode -- not a
    # per-fold or per-bootstrap mode, so the ruler cannot drift with the resample
    delta = boot_ci(lambda t, q: float((hard(q, multiclass) == t).mean()
                                       - (t == const).mean()), y, p, rng=rng)
    ba = boot_ci(lambda t, q: float(balanced_accuracy_score(t, hard(q, multiclass))),
                 y, p, rng=rng)
    out = dict(n=int(len(y)),
               class_counts={str(int(k)): int(v) for k, v in zip(labs, cnt)},
               mode_class=str(int(const)),
               baseline_constant=round(float((y == const).mean()), 4),
               accuracy=acc["value"], accuracy_ci=acc["ci"],
               delta=delta["value"], delta_ci=delta["ci"],
               balanced_accuracy=ba["value"], balanced_accuracy_ci=ba["ci"],
               per_class_recall={str(int(k)): round(float(v), 4)
                                 for k, v in zip(labs, rec)},
               collapsed_to_one_class=bool(len(np.unique(yp)) == 1),
               smallest_class=int(cnt.min()),
               minority_class_insufficient=bool(cnt.min() < MIN_CLASS_FOR_CI))
    keep = [i for i, c in enumerate(classes) if c in labs]
    try:
        if multiclass:
            pm = prob[:, keep]
            pm = pm / np.clip(pm.sum(axis=1, keepdims=True), 1e-9, None)
            out["auroc_macro_ovr"] = round(float(roc_auc_score(
                y, pm, multi_class="ovr", average="macro", labels=labs)), 4)
            out["auroc_metric"] = "macro one-vs-rest, from the mean pooling " \
                                  "probability matrix"
        else:
            out["auroc_macro_ovr"] = round(float(roc_auc_score(y, p)), 4)
            out["auroc_metric"] = "binary"
    except Exception as e:                                   # noqa: BLE001
        out["auroc_macro_ovr"] = None
        out["auroc_metric"] = "not estimable: %s" % type(e).__name__
    return out


def severe_cross_band(y, p, multiclass):
    """For a 3-class banded target, the fraction of cases predicted two bands
    away from the truth (low called high, or high called low)."""
    yp = hard(p, multiclass)
    return round(float((np.abs(yp - y) >= 2).mean()), 4)


def paired(y, pa, pb, multiclass, rng):
    """Candidate minus anchor on the same cases and the same OOF folds."""
    ya, yb = hard(pa, multiclass), hard(pb, multiclass)
    f = balanced_accuracy_score
    obs_ba = f(y, ya) - f(y, yb)
    obs_acc = float((ya == y).mean() - (yb == y).mean())
    bb, ab, n = [], [], len(y)
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        bb.append(f(y[s], ya[s]) - f(y[s], yb[s]))
        ab.append(float((ya[s] == y[s]).mean() - (yb[s] == y[s]).mean()))
    bb, ab = np.asarray(bb), np.asarray(ab)
    lo, hi = float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))
    return dict(ba_gain=round(float(obs_ba), 4), ba_ci=[round(lo, 4), round(hi, 4)],
                ba_ci_excludes_zero=bool(lo > 0 or hi < 0),
                accuracy_gain=round(obs_acc, 4),
                accuracy_ci=[round(float(np.percentile(ab, 2.5)), 4),
                             round(float(np.percentile(ab, 97.5)), 4)],
                cases_candidate_right_anchor_wrong=int(((ya == y) & (yb != y)).sum()),
                cases_anchor_right_candidate_wrong=int(((yb == y) & (ya != y)).sum()))


def fold_directions(y, pa, pb, folds_arr, multiclass):
    """Per-fold sign of the candidate-minus-anchor BA gain, so a mean gain that
    rests on one fold is visible as such."""
    ya, yb = hard(pa, multiclass), hard(pb, multiclass)
    signs, per = [], {}
    for fo in sorted(set(folds_arr)):
        m = folds_arr == fo
        if len(np.unique(y[m])) < 2:
            per[str(fo)] = None
            continue
        g = balanced_accuracy_score(y[m], ya[m]) - balanced_accuracy_score(y[m], yb[m])
        per[str(fo)] = round(float(g), 4)
        signs.append(np.sign(g))
    return dict(per_fold_ba_gain=per,
                folds_positive=int(sum(1 for s in signs if s > 0)),
                folds_negative=int(sum(1 for s in signs if s < 0)),
                folds_evaluated=int(len(signs)))


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    locked = set(man.index[man.development_fold.isna()])
    if len(locked) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(locked))
    folds = dev.development_fold.to_dict()
    arch = dev.archive

    arms, metas = {}, {}
    for a in ARMS:
        ix, feats, meta = load_arm(a)
        if set(ix.exam_case_id) & locked:
            raise RuntimeError("locked case present in arm %s" % a)
        if ix.image_path.astype(str).str.contains(r"rep[_0-9]").any():
            raise RuntimeError("report scan reached arm %s" % a)
        arms[a], metas[a] = (ix, feats), meta

    rng = np.random.default_rng(SEED)
    rows, arch_rows, preds_rows = [], [], []
    for field in IN_SCOPE:
        y_case, k, extra = target(dev, field)
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy()
        classes = sorted(set(float(v) for v in y))
        mc = k > 2
        ok = np.ones(len(order), bool)
        preds = {}
        for a, (ix, f) in arms.items():
            p, pr = oof(f, ix, y_case, folds, order, mc, classes)
            preds[a] = (p, pr)
            ok &= ~np.isnan(p)
        unit, static_ok = UNIT[field]
        base = dict(field=field, n_classes=k, multiclass=mc,
                    target=ADMISSIBLE[field][0], unit_of_observation=unit,
                    observable_from_static_image=static_ok)
        if static_ok is False and field in ("flow_state", "microthrombus"):
            base["reading"] = ("STATIC APPEARANCE CORRELATION, NOT FLOW/EVENT "
                              "OBSERVATION: the unit is a time base, so no result "
                              "here supports detecting flow events or thrombi per "
                              "minute from a still frame")
        if field in NUMERIC_FIELDS:
            base["measurement_note"] = ("banded by the report's own printed normal "
                                        "range; this is NOT a micrometre "
                                        "measurement capability")
            base.update(extra)
        if field == "capillary_count":
            base["count_note"] = ("the printed band is the target; no 条/mm figure "
                                  "is produced, and the calibration ban still "
                                  "blocks converting a band into a per-mm count")
        for a in ARMS:
            p, pr = preds[a]
            s = score(y[ok], p[ok], pr[ok], mc, classes, rng)
            r = dict(base, arm=a, arm_role=ARMS[a], readout="shipped_5pool_C0.03",
                     **s)
            if field in NUMERIC_FIELDS or field == "capillary_count":
                r["severe_cross_band_error_rate"] = severe_cross_band(
                    y[ok], p[ok], mc)
            if a != ANCHOR:
                fa = np.array([folds[c] for c in order])[ok]
                r["paired_vs_anchor"] = paired(y[ok], p[ok], preds[ANCHOR][0][ok],
                                               mc, rng)
                r["fold_directions_vs_anchor"] = fold_directions(
                    y[ok], p[ok], preds[ANCHOR][0][ok], fa, mc)
            rows.append(r)
            av = arch.reindex(order).to_numpy()[ok]
            for nm in sorted(set(av)):
                m = av == nm
                if len(np.unique(y[ok][m])) < 2:
                    arch_rows.append(dict(field=field, arm=a, archive=nm,
                                          n=int(m.sum()), ba=None,
                                          note="single class in this archive"))
                    continue
                arch_rows.append(dict(
                    field=field, arm=a, archive=nm, n=int(m.sum()),
                    ba=round(float(balanced_accuracy_score(
                        y[ok][m], hard(p[ok][m], mc))), 4),
                    accuracy=round(float((hard(p[ok][m], mc) == y[ok][m]).mean()), 4)))
            for c, t, q in zip(np.array(order)[ok], y[ok], p[ok]):
                preds_rows.append(dict(exam_case_id=c, field=field, arm=a,
                                       y_true=float(t), pred=float(q),
                                       fold=folds[c]))
        b = [r for r in rows if r["field"] == field and r["arm"] == ANCHOR][0]
        print("  %-28s n=%3d k=%d base %.3f | B d%+.3f ba %.3f | L d%+.3f | Bio d%+.3f"
              % (field, b["n"], k, b["baseline_constant"], b["delta"],
                 b["balanced_accuracy"],
                 [r for r in rows if r["field"] == field
                  and r["arm"] == "dinov2l_deployed_geometry"][0]["delta"],
                 [r for r in rows if r["field"] == field
                  and r["arm"] == "biomedclip_medical"][0]["delta"]), flush=True)

    out = dict(
        run="field_matrix_three_arms",
        question="under the shipped fixed configuration, what does each of the 15 "
                 "in-scope fields score on each of the three available encoder "
                 "arms, with no per-field selection of any kind",
        configuration=dict(poolings=POOLINGS, C_fixed=C_FIXED, pca_dim=DIM_FIXED,
                           seed=SEED, n_bootstrap=N_BOOT,
                           aggregation="mean of pooling probabilities; majority "
                                       "vote over poolings for multiclass",
                           per_field_selection="none: not C, not the pooling set, "
                                               "not the threshold, not the arm",
                           round_one_infold_C_selection_used=False,
                           split="development_fold, case-level OOF, identical "
                                 "across arms"),
        arms={a: dict(role=ARMS[a], input_size=metas[a].get("input_size"),
                      patch=metas[a].get("patch"),
                      embedding_dim=int(np.load(
                          EXP / "features" / a / "features_cls.npy").shape[1]),
                      normalisation=metas[a].get("normalisation"),
                      geometry=metas[a].get("geometry"),
                      pretraining=metas[a].get("pretraining"),
                      readout_vectors=POOLINGS) for a in ARMS},
        locked_cases_seen=0,
        locked_note="locked-47 was not loaded in this run; these are development "
                    "OOF numbers and must not be written as launch capability",
        fields=rows)
    (EXP / "field_matrix_three_arms.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = pd.DataFrame([{k: v for k, v in r.items()
                          if not isinstance(v, (dict, list))} for r in rows])
    flat.to_csv(EXP / "field_matrix_three_arms.csv", index=False,
                encoding="utf-8-sig")
    pd.DataFrame(arch_rows).to_csv(EXP / "field_matrix_three_arms_by_archive.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(preds_rows).to_csv(EXP / "field_matrix_three_arms_predictions.csv",
                                    index=False, encoding="utf-8-sig")
    print("\nwrote %s (%d rows)" % (EXP / "field_matrix_three_arms.json", len(rows)))


if __name__ == "__main__":
    main()
