"""Re-check the ViT-L capacity gain under the SHIPPED readout, not round one's.

Round one used a single cls vector with C tuned inside the training folds. The
delivered configuration is different: five spatial poolings, probabilities
averaged across them, C fixed at 0.03, PCA 64, seed 20260917. A capacity gain
that only exists under round one's readout is not a reason to change deployment,
so this re-runs both arms through the shipped recipe.

Both arms here use the DEPLOYED geometry (direct resize to 518x686), so capacity
is the only factor that varies. Nothing is tuned; there is no selection step to
leak, which is why the fixed-C recipe is the honest one for this question.

DEVELOPMENT ONLY. locked-47 is never loaded.

  PYTHONIOENCODING=utf-8 python scripts/recheck_capacity_deployed_readout.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"

# the delivered configuration's own constants, copied so nothing is re-tuned
C_FIXED, DIM_FIXED, SEED, N_BOOT = 0.03, 64, 20260917, 2000
POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]

import argparse

ARM_B = "anchor_dinov2b_deployed"
ARM_L = "dinov2l_deployed_geometry"
PRIMARY = "malformation_ratio"
OUT_STEM = "capacity_recheck"

BINARY_MAP = {
    "malformation_ratio": {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]},
    "clarity": {0: ["清晰"], 1: ["不清", "模糊"]},
    "exudation": {0: ["无"], 1: ["+", "++", "+++"]},
    "subpapillary_venous_plexus": {0: ["不见"],
                                   1: ["可见1排", "可见2排", ">2排,扩张"]},
    "blood_color": {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]},
    "rbc_aggregation": {0: ["无", "[无]"], 1: ["轻度", "中度", "重度"]},
}
MULTICLASS = {"papilla": ["波纹状", "浅波纹状", "平坦"]}
FIELDS = list(BINARY_MAP) + list(MULTICLASS)


def load_arm(arm: str):
    d = EXP / "features" / arm
    ix = pd.read_csv(d / "index.csv", dtype={"exam_case_id": str})
    return ix, {p: np.load(d / f"features_{p}.npy").astype(np.float32)
                for p in POOLINGS}


def case_labels(man: pd.DataFrame, field: str) -> pd.Series:
    raw = man[field].astype(str).str.strip()
    if field in MULTICLASS:
        keep = MULTICLASS[field]
        y = raw.where(raw.isin(keep))
        return y.dropna().map({v: i for i, v in enumerate(keep)}).astype(float)
    m = {}
    for k, vs in BINARY_MAP[field].items():
        for v in vs:
            m[v] = float(k)
    return raw.map(m).dropna()


def fit_predict(Xtr, ytr, Xts, multiclass):
    if len(set(ytr)) < 2:
        return (np.full(len(Xts), float(ytr[0])) if multiclass
                else np.full(len(Xts), float(np.mean(ytr))))
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=dim, random_state=SEED),
                         LogisticRegression(C=C_FIXED, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return (pipe.predict(Xts).astype(float) if multiclass
            else pipe.predict_proba(Xts)[:, 1])


def oof(feats, ix, y_case, folds, order, multiclass):
    """Exactly the shipped aggregation: probabilities averaged over poolings,
    or a majority vote over poolings for the multiclass field."""
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    fs = pd.Series([folds.get(c, np.nan) for c in order], index=order)
    for fo in sorted(fs.dropna().unique()):
        te = set(fs.index[fs == fo])
        tr = set(order) - te
        trm = dm & ix.exam_case_id.isin(tr).to_numpy()
        tem = dm & ix.exam_case_id.isin(te).to_numpy()
        if trm.sum() == 0 or tem.sum() == 0:
            continue
        ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
        if multiclass:
            preds = np.stack([fit_predict(feats[p][trm], ytr, feats[p][tem], True)
                              for p in POOLINGS])
            df = pd.DataFrame(preds.T, index=ix.exam_case_id[tem].to_numpy())
            for c, grp in df.groupby(level=0):
                vals, cnt = np.unique(grp.to_numpy().ravel(), return_counts=True)
                if c in pos:
                    out[pos[c]] = vals[cnt.argmax()]
        else:
            ps = [fit_predict(feats[p][trm], ytr, feats[p][tem], False)
                  for p in POOLINGS]
            s = pd.Series(np.mean(ps, axis=0),
                          index=ix.exam_case_id[tem].to_numpy()).groupby(level=0).mean()
            for c, v in s.items():
                if c in pos:
                    out[pos[c]] = v
    return out


def score(y, p, multiclass):
    yp = p if multiclass else (p >= 0.5).astype(float)
    vals, cnt = np.unique(y, return_counts=True)
    const = float(vals[cnt.argmax()])
    labs = [float(v) for v in vals]
    rec = recall_score(y, yp, labels=labs, average=None, zero_division=0)
    minority = min(labs, key=lambda k: cnt[labs.index(k)])
    out = dict(n=int(len(y)), accuracy=round(float((yp == y).mean()), 4),
               baseline_constant=round(float((y == const).mean()), 4),
               balanced_accuracy=round(float(balanced_accuracy_score(y, yp)), 4),
               per_class_recall={str(int(k)): round(float(v), 4)
                                 for k, v in zip(labs, rec)},
               minority_class=str(int(minority)),
               minority_recall=round(float(rec[labs.index(minority)]), 4),
               collapsed=bool(len(np.unique(yp)) == 1))
    out["delta"] = round(out["accuracy"] - out["baseline_constant"], 4)
    if not multiclass and len(vals) == 2:
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
        pos_l, neg_l = vals.max(), vals.min()
        out["miss_rate"] = round(float(((y == pos_l) & (yp == neg_l)).sum()
                                       / max(1, int((y == pos_l).sum()))), 4)
    return out


def paired(y, pl, pb, multiclass, rng):
    h = (lambda p: p) if multiclass else (lambda p: (p >= 0.5).astype(float))
    yl, yb = h(pl), h(pb)
    obs = balanced_accuracy_score(y, yl) - balanced_accuracy_score(y, yb)
    boot, n = [], len(y)
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        boot.append(balanced_accuracy_score(y[s], yl[s])
                    - balanced_accuracy_score(y[s], yb[s]))
    boot = np.asarray(boot)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(ba_gain=round(float(obs), 4), ci=[round(lo, 4), round(hi, 4)],
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                cases_L_right_B_wrong=int(((yl == y) & (yb != y)).sum()),
                cases_B_right_L_wrong=int(((yb == y) & (yl != y)).sum()))


def main() -> None:
    global ARM_B, ARM_L, OUT_STEM, POOLINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-b", default=ARM_B)
    ap.add_argument("--arm-l", default=ARM_L)
    ap.add_argument("--poolings", default=",".join(POOLINGS),
                    help="comma separated; use 'cls' to hold the readout at "
                         "round one's single vector while the recipe stays fixed")
    ap.add_argument("--stem", default=OUT_STEM)
    a = ap.parse_args()
    ARM_B, ARM_L, OUT_STEM = a.arm_b, a.arm_l, a.stem
    POOLINGS = [p.strip() for p in a.poolings.split(",") if p.strip()]

    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    locked = set(man.index[man.development_fold.isna()])
    folds = dev.development_fold.to_dict()
    arch = dev.archive

    arms = {}
    for a in (ARM_B, ARM_L):
        ix, feats = load_arm(a)
        if set(ix.exam_case_id) & locked:
            raise RuntimeError("locked case in %s" % a)
        arms[a] = (ix, feats)

    rng = np.random.default_rng(SEED)
    rows, per_arch_rows = [], []
    for field in FIELDS:
        mc = field in MULTICLASS
        y_case = case_labels(dev, field)
        if y_case.nunique() < 2 or len(y_case) < 30:
            continue
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy()
        preds = {a: oof(f, ix, y_case, folds, order, mc)
                 for a, (ix, f) in arms.items()}
        ok = ~(np.isnan(preds[ARM_B]) | np.isnan(preds[ARM_L]))
        for a in arms:
            rows.append(dict(field=field, arm=a, readout="deployed_5pool_C0.03",
                             **score(y[ok], preds[a][ok], mc)))
        pr = paired(y[ok], preds[ARM_L][ok], preds[ARM_B][ok], mc, rng)
        rows[-1]["paired_L_minus_B"] = pr
        # archive stratification on the same OOF predictions
        av = arch.reindex(order).to_numpy()
        for a_name in sorted(set(av[ok])):
            m = ok & (av == a_name)
            if len(np.unique(y[m])) < 2:
                continue
            per_arch_rows.append(dict(
                field=field, archive=a_name, n=int(m.sum()),
                ba_B=round(float(balanced_accuracy_score(
                    y[m], preds[ARM_B][m] if mc else (preds[ARM_B][m] >= .5))), 4),
                ba_L=round(float(balanced_accuracy_score(
                    y[m], preds[ARM_L][m] if mc else (preds[ARM_L][m] >= .5))), 4)))
        print("  %-28s B %.4f  L %.4f  gain %+.4f ci [%+.4f,%+.4f]" % (
            field, rows[-2]["balanced_accuracy"], rows[-1]["balanced_accuracy"],
            pr["ba_gain"], pr["ci"][0], pr["ci"][1]), flush=True)

    pa = pd.DataFrame(per_arch_rows)
    pa["ba_gain"] = (pa.ba_L - pa.ba_B).round(4)
    pa.to_csv(EXP / ("%s_by_archive.csv" % OUT_STEM), index=False,
              encoding="utf-8-sig")
    out = dict(
        question="does the ViT-L gain survive the shipped readout, with geometry "
                 "held at the deployed setting so capacity is the only factor",
        readout=dict(poolings=POOLINGS, C_fixed=C_FIXED, pca_dim=DIM_FIXED,
                     seed=SEED, aggregation="mean of pooling probabilities; "
                     "majority vote over poolings for the multiclass field",
                     nothing_tuned=True),
        arms={ARM_B: "DINOv2 ViT-B/14, deployed geometry",
              ARM_L: "DINOv2 ViT-L/14, deployed geometry"},
        locked_read=False, fields=rows,
        caveats=[
            "development set only; locked-47 was not read",
            "this is the shipped recipe applied to a new encoder, not a new "
            "search; no hyperparameter was selected here",
            "malformation_ratio truth is the doctor's banded report value",
            "a gain here is a research result, not a launch criterion",
        ])
    (EXP / ("%s.json" % OUT_STEM)).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote %s" % (EXP / ("%s.json" % OUT_STEM)))
    print(pa.to_string(index=False))


if __name__ == "__main__":
    main()
