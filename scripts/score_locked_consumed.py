"""Score the frozen A0 and A2x heads on locked-47. Descriptive evaluation only.

This is NOT an external validation, a clinical validation, a diagnostic accuracy study
or a final clean test. locked-47 has been consumed seven times already, with model
selection on it in three of those runs, so it is a consumed internal holdout and the
artefact is named for that. The name is enforced below, not just documented.

Nothing is fitted here. The heads, the transform, the threshold, the abstention rule,
the output contract and the RAG routing were all frozen and committed before the
locked features existed. Whatever comes out stands: if locked is worse than
development, the result stays as it is.

  PYTHONIOENCODING=utf-8 PYTHONPATH=scripts python scripts/score_locked_consumed.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import LABELS, N_BOOT, POOLINGS, SEED, target  # noqa: E402
from run_rescue_ladder_local import attach  # noqa: E402

EXP = ROOT / "artifacts" / "experiments" / "locked_consumed_20260924"
FEAT = EXP / "features_locked"
HEADS = (ROOT / "artifacts" / "experiments" / "product_contract_20260924"
         / "final_heads")
FROZEN = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
          / "frozen_candidate_malformation_a2x")
FIELD = "malformation_ratio"

# The only permitted name for this artefact. Anything else is a category error.
EVALUATION_NAME = "consumed internal holdout descriptive evaluation"
FORBIDDEN_NAMES = ["external validation", "clinical validation",
                   "diagnostic accuracy", "final clean test", "clean test set",
                   "independent validation"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_heads() -> dict:
    with open(HEADS / "manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    heads = {}
    for arm, want in man["head_sha256"].items():
        path = HEADS / ("head_%s.pkl" % arm)
        got = sha(path)
        if got != want:
            raise RuntimeError("%s head changed since the freeze: %s" % (arm, got))
        with open(path, "rb") as fh:
            heads[arm] = pickle.load(fh)
    return dict(manifest=man, heads=heads)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, same binning as the development artefacts."""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(float)
    hit = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    tot, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum():
            tot += m.mean() * abs(hit[m].mean() - conf[m].mean())
            rows.append(dict(bin=[round(lo, 2), round(hi, 2)], n=int(m.sum()),
                             mean_confidence=round(float(conf[m].mean()), 4),
                             observed_accuracy=round(float(hit[m].mean()), 4)))
    return float(tot), rows


def metrics(y: np.ndarray, p: np.ndarray, classes: list) -> dict:
    pred = (p >= 0.5).astype(float)
    try:
        auc = round(float(roc_auc_score(y, p)), 4)
    except ValueError:
        auc = None
    e, bins = ece(y, p)
    cm = confusion_matrix(y, pred, labels=classes)
    return dict(
        accuracy=round(float((y == pred).mean()), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y, pred)), 4),
        auroc=auc,
        per_class_recall={str(int(c)): round(float(v), 4) for c, v in
                          zip(classes, recall_score(y, pred, labels=classes,
                                                    average=None,
                                                    zero_division=0))},
        confusion_matrix=dict(labels=[int(c) for c in classes],
                              rows_true_cols_pred=cm.tolist()),
        ece=round(e, 4), calibration_bins=bins,
        predicted_classes=sorted(set(float(v) for v in pred)),
        collapsed_to_one_class=bool(len(set(pred)) == 1),
    )


def coverage_curve(y: np.ndarray, p: np.ndarray, gate: float) -> list:
    """Accuracy as a function of how much the model refuses to answer."""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(float)
    rows = []
    for target_cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(target_cov * len(y))))
        keep = np.argsort(-conf)[:k]
        rows.append(dict(target_coverage=target_cov, n_kept=int(k),
                         coverage=round(k / len(y), 4),
                         accuracy=round(float((y[keep] == pred[keep]).mean()), 4),
                         confidence_threshold=round(float(conf[keep].min()), 4)))
    # The operating point actually shipped: the frozen threshold, not a re-sorted
    # quantile of this set. Choosing a locked quantile would be threshold selection
    # on locked, which is exactly what item 3 forbids.
    m = conf >= gate
    rows.append(dict(at_frozen_threshold=round(gate, 4), n_kept=int(m.sum()),
                     coverage=round(float(m.mean()), 4),
                     accuracy=(round(float((y[m] == pred[m]).mean()), 4)
                               if m.sum() else None),
                     note="the shipped gate applied as frozen; no locked-side tuning"))
    return rows


def paired(y: np.ndarray, cand: np.ndarray, anch: np.ndarray, rng) -> dict:
    """Paired bootstrap over locked cases, both arms on the same resample."""
    def d(t, c, a):
        return (float(balanced_accuracy_score(t, (c >= 0.5).astype(float)))
                - float(balanced_accuracy_score(t, (a >= 0.5).astype(float))))

    obs_ba = d(y, cand, anch)
    obs_acc = (float((y == (cand >= 0.5)).mean())
               - float((y == (anch >= 0.5)).mean()))
    vb, va, n = [], [], len(y)
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        vb.append(d(y[s], cand[s], anch[s]))
        va.append(float((y[s] == (cand[s] >= 0.5)).mean())
                  - float((y[s] == (anch[s] >= 0.5)).mean()))
    vb, va = np.asarray(vb), np.asarray(va)
    lo, hi = float(np.percentile(vb, 2.5)), float(np.percentile(vb, 97.5))
    return dict(ba_gain=round(obs_ba, 4), ba_ci=[round(lo, 4), round(hi, 4)],
                ba_ci_excludes_zero=bool(lo > 0 or hi < 0),
                accuracy_gain=round(obs_acc, 4),
                accuracy_ci=[round(float(np.percentile(va, 2.5)), 4),
                             round(float(np.percentile(va, 97.5)), 4)],
                n_bootstrap_kept=int(len(vb)))


def predict(head: dict, X: dict, ids: np.ndarray, cases: list) -> np.ndarray:
    """Average P(class 1) over the five poolings, then over each case's images."""
    per = [head["pipes"][p].predict_proba(X[p])[:, 1] for p in POOLINGS]
    return (pd.Series(np.mean(per, axis=0), index=ids).groupby(level=0).mean()
            .reindex(cases).to_numpy())


def main() -> None:
    bundle = load_heads()
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    lock = man[man.development_fold.isna()]
    if len(lock) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(lock))

    y_case, n_classes, _ = target(lock, FIELD)
    if n_classes != 2:
        raise RuntimeError("malformation_ratio must be binary on locked too; got %d"
                           % n_classes)
    cases = sorted(y_case.index)
    classes = [0.0, 1.0]
    y = y_case.reindex(cases).to_numpy()

    ix = pd.read_csv(FEAT / "index.csv", dtype={"exam_case_id": str})
    keep = ix.exam_case_id.isin(set(cases)).to_numpy()
    ids = ix.exam_case_id[keep].to_numpy()
    base = {p: np.load(FEAT / ("features_%s.npy" % p)).astype(np.float32)[keep]
            for p in POOLINGS}

    local = pd.read_csv(FEAT / "case_features.csv",
                        dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev_local = pd.read_csv(FROZEN / "case_features.csv",
                            dtype={"exam_case_id": str}).set_index("exam_case_id")
    if list(local.columns) != list(dev_local.columns):
        raise RuntimeError("local columns differ from the development table")
    # attach() is imported rather than reimplemented: it performs the per-image
    # broadcast, the NaN-to-zero-after-scaling rule and the missing-row guard that
    # development used. A local copy that drifted would still produce numbers.
    a2x = attach({p: base[p] for p in POOLINGS},
                 pd.DataFrame(dict(exam_case_id=ids)), local)
    for p in POOLINGS:
        if a2x[p].shape[1] != base[p].shape[1] + local.shape[1]:
            raise RuntimeError("attach produced an unexpected width for %s" % p)

    p0 = predict(bundle["heads"]["A0"], base, ids, cases)
    pX = predict(bundle["heads"]["A2x"], a2x, ids, cases)
    if np.isnan(p0).any() or np.isnan(pX).any():
        raise RuntimeError("a locked case produced no prediction")

    vals, cnt = np.unique(y, return_counts=True)
    own_mode = float(vals[cnt.argmax()])
    with open(FROZEN / "frozen_candidate.json", encoding="utf-8") as fh:
        frozen = json.load(fh)
    dev = frozen["development_oof"]
    gate = bundle["manifest"]["decision_rule"]["abstain_below"]

    rng = np.random.default_rng(SEED)
    mX, m0 = metrics(y, pX, classes), metrics(y, p0, classes)
    report = dict(
        run="locked_consumed_20260924",
        evaluation_name=EVALUATION_NAME,
        this_is_not=FORBIDDEN_NAMES,
        why=("locked-47 has been read seven times and carried model selection in "
             "three of those runs. A set that has been selected on cannot be a clean "
             "test set afterwards, whatever is frozen now."),
        field=FIELD,
        n=int(len(y)),
        n_locked_total=47,
        excluded_no_label=int(47 - len(y)),
        class_counts={str(int(c)): int(n) for c, n in zip(vals, cnt)},
        baselines=dict(
            locked_own_mode=dict(
                class_=own_mode, accuracy=round(float((y == own_mode).mean()), 4),
                why="hindsight: the deployed model does not know this prevalence"),
            development_mode=dict(
                class_=0.0, accuracy=round(float((y == 0.0).mean()), 4),
                why="what the frozen model actually carries from development"),
        ),
        a2x=mX, a0=m0,
        a2x_minus_a0=paired(y, pX, p0, rng),
        coverage_accuracy_curve=coverage_curve(y, pX, gate),
        frozen_threshold=gate,
        development_comparison=dict(
            a2x=dict(locked_accuracy=mX["accuracy"],
                     development_accuracy=dev["accuracy"],
                     accuracy_change=round(mX["accuracy"] - dev["accuracy"], 4),
                     locked_ba=mX["balanced_accuracy"],
                     development_ba=dev["balanced_accuracy"],
                     ba_change=round(mX["balanced_accuracy"]
                                     - dev["balanced_accuracy"], 4),
                     locked_auroc=mX["auroc"],
                     development_auroc=dev["auroc_macro_ovr"]),
            a0=dict(locked_accuracy=m0["accuracy"],
                    development_accuracy=dev["anchor_a0"]["accuracy"],
                    locked_ba=m0["balanced_accuracy"],
                    development_ba=dev["anchor_a0"]["balanced_accuracy"]),
            not_like_for_like=("development numbers are 5-fold OOF with the head "
                               "refit per fold; locked is one frozen head fit on all "
                               "162 labelled development cases. A change here mixes "
                               "site shift with that protocol difference."),
        ),
        prevalence_flip_warning=dict(
            development_mode_class=0.0, locked_mode_class=own_mode,
            note=("the majority class FLIPS between development (0, 56.8%) and "
                  "locked (1, 53.8%). So the frozen model's carried constant scores "
                  "only 0.4615 here, and accuracy minus that number reads +0.154 -- "
                  "an artefact of the flip, not a gain. The honest single-constant "
                  "comparison on this set is the locked own-mode 0.5385, against "
                  "which A2x is +0.0769, and even that has 39 cases behind it."),
            delta_vs_development_mode=round(mX["accuracy"] - float((y == 0.0).mean()), 4),
            delta_vs_locked_own_mode=round(mX["accuracy"] - float((y == own_mode).mean()), 4),
            which_to_quote="delta_vs_locked_own_mode",
        ),
        result_is_final=("whatever these numbers are, the model is not changed in "
                         "response to them; that is the condition under which the "
                         "read was authorised"),
        locked_cases_seen=int(len(y)),
        limitations=[
            "no independent clinician annotation exists, so nothing here speaks to "
            "clinical correctness",
            "47 cases is small: every interval below is wide and a single case moves "
            "accuracy by about 2 points",
            "the reference standard is the original report wording, single reader",
            "malformation_ratio remains a binary band correspondence; no percentage, "
            "per-mm or micron value may be emitted from it",
            "8 of the 47 locked cases carry no admissible malformation_ratio label, "
            "so this reads 39 cases; they are reported, not imputed",
            "class 1 recall is 0.4286 -- the model misses more than half the "
            "above-band cases on this set, which matters more than the headline "
            "accuracy for anything framed as a rule-out",
        ],
    )

    EXP.mkdir(parents=True, exist_ok=True)
    with open(EXP / "locked_consumed_evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    pd.DataFrame(dict(exam_case_id=cases, y_true=y, a0_prob=p0, a2x_prob=pX)).to_csv(
        EXP / "locked_predictions.csv", index=False)

    print("%s: n=%d (%d of 47 had no label)"
          % (EVALUATION_NAME, len(y), 47 - len(y)))
    print("  dev-mode baseline %.4f | locked-own-mode %.4f"
          % (report["baselines"]["development_mode"]["accuracy"],
             report["baselines"]["locked_own_mode"]["accuracy"]))
    for arm, m in (("A2x", mX), ("A0", m0)):
        print("  %-4s acc %.4f  BA %.4f  AUROC %s  recall %s  collapsed=%s"
              % (arm, m["accuracy"], m["balanced_accuracy"], m["auroc"],
                 m["per_class_recall"], m["collapsed_to_one_class"]))
    g = report["a2x_minus_a0"]
    print("  A2x-A0 BA %+.4f %s excl0=%s"
          % (g["ba_gain"], g["ba_ci"], g["ba_ci_excludes_zero"]))
    d = report["development_comparison"]["a2x"]
    print("  vs development: acc %+.4f  BA %+.4f"
          % (d["accuracy_change"], d["ba_change"]))
    f = report["prevalence_flip_warning"]
    print("  PREVALENCE FLIP: delta vs dev-mode %+.4f is inflated; quote %+.4f"
          % (f["delta_vs_development_mode"], f["delta_vs_locked_own_mode"]))
    print("  ECE %.4f (development %s)" % (mX["ece"], frozen["calibration"].get("ece")))
    print("wrote %s" % (EXP / "locked_consumed_evaluation.json"))


if __name__ == "__main__":
    main()
