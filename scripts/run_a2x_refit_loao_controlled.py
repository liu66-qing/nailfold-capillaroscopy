"""Refit leave-one-archive-out for malformation_ratio, with A0 refit the same way.

The freeze already had a refit LOAO for A2x, and it looked strong (+0.1961 /
+0.1572 / +0.1219 against the training-archive majority). That number answers the
wrong question on its own: beating the training archives' constant answer is not
the same as beating the whole-image anchor. A0 refit under the identical protocol
is the control that was missing.

So for each held-out archive, BOTH arms are refit from scratch on the other two
archives -- new StandardScaler, new PCA, new logistic head, per pooling -- and the
comparison reported is A2x minus A0 within that held-out archive, with a paired
bootstrap CI over its cases.

Three baselines are printed side by side because they answer different questions:
  * training-archive majority: what a deployed model carries. The honest baseline,
    since the held-out site's prevalence is unknown at deployment.
  * held-out archive's own majority: what a reader with hindsight would use. Shown
    only to make the gap visible, never as the headline.
  * A0 refit: the actual question -- does the local morphology table add anything
    across a site boundary.

development archives only. locked-47 is not read.

  PYTHONIOENCODING=utf-8 PYTHONPATH=src \
    /c/Users/liujunqing/anaconda3/envs/pytorch_gpu/python.exe \
    scripts/run_a2x_refit_loao_controlled.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402
    ANCHOR, LABELS, N_BOOT, POOLINGS, SEED, fit_predict, hard, load_arm, target,
)
from run_rescue_ladder_local import attach, load_local  # noqa: E402

FIELD = "malformation_ratio"
LOCAL_TAG = "external"      # the A2x table: external-pretrained detector
EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
OUT = EXP / "frozen_candidate_malformation_a2x"

# From the freeze. Asserted so a drifted input cannot be compared against the
# recorded candidate.
EXPECTED_A2X_REFIT = {"recovered_archive1": 0.7451,
                      "recovered_archive2": 0.7143,
                      "recovered_archive3": 0.7317}


def fit_arm(feats, ix, y_case, tr_cases, te_cases, classes, multiclass):
    """Refit every pooling on the training archives and predict the held-out one."""
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    trm = dm & ix.exam_case_id.isin(tr_cases).to_numpy()
    tem = dm & ix.exam_case_id.isin(te_cases).to_numpy()
    ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
    ids = ix.exam_case_id[tem].to_numpy()
    got = [fit_predict(feats[p][trm], ytr, feats[p][tem], multiclass, classes)
           for p in POOLINGS]
    order = sorted(te_cases)
    prob = (pd.DataFrame(np.mean([g[1] for g in got], axis=0), index=ids)
            .groupby(level=0).mean().reindex(order).to_numpy())
    if multiclass:
        df = pd.DataFrame(np.stack([g[0] for g in got]).T, index=ids)
        vote = {}
        for c, grp in df.groupby(level=0):
            v, n = np.unique(grp.to_numpy().ravel(), return_counts=True)
            vote[c] = float(v[n.argmax()])
        pred = np.array([vote[c] for c in order], float)
    else:
        s = (pd.Series(np.mean([g[0] for g in got], axis=0), index=ids)
             .groupby(level=0).mean().reindex(order))
        pred = hard(s.to_numpy(), False)
    return order, pred, prob


def metrics(y, pred, prob, classes, multiclass):
    rec = recall_score(y, pred, labels=classes, average=None, zero_division=0)
    try:
        auc = (roc_auc_score(y, prob[:, 1]) if not multiclass
               else roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None      # a held-out archive with one class present
    return dict(
        accuracy=round(float((y == pred).mean()), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y, pred)), 4),
        auroc=None if auc is None else round(float(auc), 4),
        per_class_recall={str(int(c)): round(float(v), 4)
                          for c, v in zip(classes, rec)},
        predicted_classes=sorted(set(float(v) for v in pred)),
        collapsed_to_one_class=bool(len(set(pred)) == 1),
    )


def paired_ci(y, cand, anch, rng):
    """Paired bootstrap over the held-out archive's cases, on the same resample.

    Resampling both arms together is what makes this a paired contrast: the two
    arms saw the same cases, so the CI should reflect only the difference between
    them, not the archive's own sampling noise twice over.
    """
    def d(t, c, a):
        return (float(balanced_accuracy_score(t, c))
                - float(balanced_accuracy_score(t, a)))

    obs_ba = d(y, cand, anch)
    obs_acc = float((y == cand).mean()) - float((y == anch).mean())
    vb, va, n = [], [], len(y)
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        vb.append(d(y[s], cand[s], anch[s]))
        va.append(float((y[s] == cand[s]).mean()) - float((y[s] == anch[s]).mean()))
    vb, va = np.asarray(vb), np.asarray(va)
    lo, hi = float(np.percentile(vb, 2.5)), float(np.percentile(vb, 97.5))
    return dict(
        ba_gain=round(obs_ba, 4), ba_ci=[round(lo, 4), round(hi, 4)],
        ba_ci_excludes_zero=bool(lo > 0 or hi < 0),
        accuracy_gain=round(obs_acc, 4),
        accuracy_ci=[round(float(np.percentile(va, 2.5)), 4),
                     round(float(np.percentile(va, 97.5)), 4)],
        n_bootstrap_kept=int(len(vb)),
    )


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    if len(dev) != 186 or len(locked) != 47:
        raise RuntimeError("expected 186 dev / 47 locked, got %d / %d"
                          % (len(dev), len(locked)))

    ix, feats, meta = load_arm(ANCHOR)
    if meta.get("locked_cases_seen", 0) != 0:
        raise RuntimeError("anchor arm reports locked cases")
    extra = load_local(LOCAL_TAG)
    if extra is None:
        raise SystemExit("local feature table %r missing" % LOCAL_TAG)
    if locked & set(extra.index):
        raise RuntimeError("the local feature table contains locked cases")

    y_case, n_classes, _ = target(dev, FIELD)
    multiclass = n_classes > 2
    if n_classes != 2:
        raise RuntimeError("malformation_ratio must be binary; got %d" % n_classes)
    order = sorted(y_case.index)
    classes = sorted(set(float(v) for v in y_case.reindex(order)))
    arch = dev.archive.reindex(order).to_numpy()
    a2x_feats = attach(feats, ix, extra)

    rng = np.random.default_rng(SEED)
    per, drift = {}, {}
    for a in sorted(set(arch)):
        te = {c for c, ar in zip(order, arch) if ar == a}
        tr = set(order) - te
        o0, p0, pr0 = fit_arm(feats, ix, y_case, tr, te, classes, multiclass)
        oX, pX, prX = fit_arm(a2x_feats, ix, y_case, tr, te, classes, multiclass)
        if o0 != oX:
            raise RuntimeError("the two arms scored different case orders")
        y = y_case.reindex(o0).to_numpy()
        ytr = y_case.reindex(sorted(tr)).to_numpy()
        v, n = np.unique(ytr, return_counts=True)
        const_tr = float(v[n.argmax()])
        v2, n2 = np.unique(y, return_counts=True)
        const_own = float(v2[n2.argmax()])
        m0, mX = (metrics(y, p0, pr0, classes, multiclass),
                  metrics(y, pX, prX, classes, multiclass))
        base_tr = round(float((y == const_tr).mean()), 4)
        per[a] = dict(
            n=int(len(y)), trained_on=sorted(set(arch) - {a}),
            baseline_training_archive_majority=dict(
                class_=const_tr, accuracy=base_tr,
                why="what a deployed model carries; the held-out site's "
                    "prevalence is unknown at deployment"),
            baseline_held_out_own_majority=dict(
                class_=const_own,
                accuracy=round(float((y == const_own).mean()), 4),
                why="hindsight baseline, shown only to make the gap visible"),
            a0_refit=m0, a2x_refit=mX,
            a2x_minus_a0=paired_ci(y, pX, p0, rng),
            a2x_minus_training_majority=round(mX["accuracy"] - base_tr, 4),
            a0_minus_training_majority=round(m0["accuracy"] - base_tr, 4),
            cases=o0, y_true=y.tolist(),
            a0_pred=p0.tolist(), a2x_pred=pX.tolist(),
        )
        if abs(mX["accuracy"] - EXPECTED_A2X_REFIT[a]) > 1e-4:
            drift[a] = (EXPECTED_A2X_REFIT[a], mX["accuracy"])
    if drift:
        raise RuntimeError("A2x refit no longer reproduces the freeze: %s" % drift)

    # The decision the user asked for: a cross-archive MODEL gain requires A2x to
    # hold up against A0 out of archive, not merely against the training majority.
    gains = {a: per[a]["a2x_minus_a0"]["ba_gain"] for a in per}
    worst = min(gains.values())
    verdict = dict(
        directions_evaluated=len(gains),
        ba_gain_vs_a0_per_archive=gains,
        directions_positive=sum(1 for v in gains.values() if v > 0),
        directions_clearly_negative=sum(1 for v in gains.values() if v <= -0.05),
        any_ci_excludes_zero=[a for a in per
                              if per[a]["a2x_minus_a0"]["ba_ci_excludes_zero"]],
        worst_direction=worst,
        no_material_regression_vs_a0=bool(worst > -0.05),
        cross_archive_model_gain=bool(
            all(v > 0 for v in gains.values())
            and any(per[a]["a2x_minus_a0"]["ba_ci_excludes_zero"] for a in per)),
        why_not_the_training_majority_comparison=(
            "beating the training archives' constant answer only shows the model "
            "is not vacuous. The freeze's +0.1961/+0.1572/+0.1219 were that "
            "weaker question; A0 refit under the identical protocol is the "
            "control, and the headline is A2x minus A0 within each held-out "
            "archive."),
    )

    rec = dict(
        run="controlled refit LOAO for malformation_ratio: A0 and A2x both refit",
        field=FIELD, arms=dict(A0=ANCHOR,
                               A2x="%s + local_features/%s" % (ANCHOR, LOCAL_TAG)),
        protocol=("per held-out archive: both arms refit from scratch on the other "
                  "two archives -- new StandardScaler, new PCA, new logistic head, "
                  "per pooling. Nothing is reused across archives."),
        n_classes=n_classes, classes=classes, seed=SEED, n_boot=N_BOOT,
        poolings=list(POOLINGS),
        reproduces_freeze_a2x_refit=True,
        per_archive=per, verdict=verdict,
        locked_cases_seen=0,
        locked_note="development archives only; says nothing about locked-47",
        limitations=[
            "three archives is three comparisons; each held-out n is 41-70, so a "
            "single archive's CI is wide",
            "archives are not independent sites in the regulatory sense; they are "
            "this project's own recovered batches",
            "development data; this is not a product capability claim",
        ],
    )
    (OUT / "refit_loao_controlled.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for a, d in sorted(per.items()):
        for arm, m in (("A0", d["a0_refit"]), ("A2x", d["a2x_refit"])):
            rows.append(dict(
                held_out_archive=a, n=d["n"], arm=arm,
                training_majority_baseline=d[
                    "baseline_training_archive_majority"]["accuracy"],
                own_majority_baseline=d[
                    "baseline_held_out_own_majority"]["accuracy"],
                accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"],
                auroc=m["auroc"], recall_0=m["per_class_recall"].get("0"),
                recall_1=m["per_class_recall"].get("1"),
                collapsed=m["collapsed_to_one_class"],
                ba_gain_a2x_minus_a0=(d["a2x_minus_a0"]["ba_gain"]
                                      if arm == "A2x" else None),
                ba_ci=(str(d["a2x_minus_a0"]["ba_ci"]) if arm == "A2x" else None)))
    pd.DataFrame(rows).to_csv(OUT / "refit_loao_controlled.csv", index=False,
                              encoding="utf-8-sig")

    for a, d in sorted(per.items()):
        print("== %s  n=%d  train-majority base=%.4f  own-majority base=%.4f"
              % (a, d["n"], d["baseline_training_archive_majority"]["accuracy"],
                 d["baseline_held_out_own_majority"]["accuracy"]))
        for arm, m in (("A0 ", d["a0_refit"]), ("A2x", d["a2x_refit"])):
            print("   %s acc=%.4f BA=%.4f AUROC=%s recall=%s collapsed=%s"
                  % (arm, m["accuracy"], m["balanced_accuracy"], m["auroc"],
                     m["per_class_recall"], m["collapsed_to_one_class"]))
        g = d["a2x_minus_a0"]
        print("   A2x-A0 BA %+.4f %s excl0=%s | acc %+.4f"
              % (g["ba_gain"], g["ba_ci"], g["ba_ci_excludes_zero"],
                 g["accuracy_gain"]))
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("wrote", OUT / "refit_loao_controlled.json")


if __name__ == "__main__":
    main()
