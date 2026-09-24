"""A0-A3 / M0-M2 / R0-R2: does external data or medical pretraining raise any field?

Every metric here comes from run_field_matrix_three_arms -- the shipped ruler is
imported, not re-implemented, so a number in this table means exactly what the
same-named number meant in the field matrix:

    accuracy - single-constant-answer (development mode) baseline = delta
    5 poolings [mean, topk_mean, max, cls, std], C=0.03, PCA 64, seed 20260917
    case-level OOF on development_fold, unique key exam_case_id + field

What this script adds on top of that ruler:

  * the adapted arms (A1/M1/R1), each of which reads a checkpoint that saw only
    external unlabelled images -- 0 local images, 0 labels, 0 folds -- so one
    checkpoint is legitimately shared by all outer folds;
  * the paired candidate-minus-A0 difference with its bootstrap CI, judged
    against the effect size pre-registered in preregistration.md BEFORE any of
    these numbers existed: BA gain >= 0.05 with a CI excluding 0, because the
    measured paired-CI half-width at n~180 is 0.055 and 19 of 21 prior
    comparisons could not separate from zero;
  * leave-one-archive-out in all three directions;
  * calibration and abstention coverage/accuracy;
  * an explicit class-collapse check, so an accuracy rise that is really a
    collapse cannot be read as an improvement.

It does NOT touch locked-47, does not select C, does not choose poolings, and
does not decide anything per field after the fact. Detector-based local
morphology features are a separate script (extract_detector_local_features.py)
and are joined in by run_rescue_ladder_local.py; this file is the whole-image
ladder only.

Reproduce:
  python scripts/run_rescue_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402  the shared, shipped ruler
    ANCHOR, DIRTY, IN_SCOPE, LABELS, N_BOOT, SEED, UNIT, ADMISSIBLE,
    fold_directions, hard, load_arm, oof, paired, score, target)

OUT = ROOT / "artifacts" / "experiments" / "rescue_external_20260922" / "ladder"

# The ladder. Each entry is (arm directory, rung label, the arm it must be
# compared against). A1/M1/R1 differ from their rung-0 arm ONLY by the external
# adaptation, which is what makes the paired difference attributable.
LADDER = {
    "A0": ("anchor_dinov2b_deployed",
           "A0  shipped DINOv2-B, no new external data", None),
    "A1": ("dinov2b_da_external",
           "A1  DINOv2-B + external unlabelled domain adaptation",
           "anchor_dinov2b_deployed"),
    "A0L": ("dinov2l_deployed_geometry",
            "A0L capacity control, DINOv2-L, deployed geometry", None),
    "A1L": ("dinov2l_da_external",
            "A1L DINOv2-L + external unlabelled domain adaptation",
            "dinov2l_deployed_geometry"),
    "R0": ("retfound_dinov2_meh",
           "R0  RETFound (continued DINOv2-L SSL on retinal images)", None),
    "R1": ("retfound_da_external",
           "R1  RETFound + external unlabelled domain adaptation",
           "retfound_dinov2_meh"),
    "M0": ("medsiglip_medical", "M0  MedSigLIP-448 frozen vision tower", None),
    "M1": ("medsiglip_da_external",
           "M1  MedSigLIP + external unlabelled domain adaptation",
           "medsiglip_medical"),
}

# Pre-registered in preregistration.md before any arm below was evaluated.
MIN_EFFECT = 0.05
BELOW_RESOLUTION = 0.02


def verdict(pair: dict | None, sc: dict) -> str:
    """The four categories fixed in the pre-registration. No fifth category is
    invented after seeing the numbers."""
    if sc["collapsed_to_one_class"] or sc["minority_class_insufficient"]:
        return "not_measurable"
    if pair is None:
        return "reference_arm"
    g = pair["ba_gain"]
    if abs(g) < BELOW_RESOLUTION:
        return "no_effect"
    if g >= MIN_EFFECT and pair["ba_ci_excludes_zero"]:
        return "meets_effect_threshold"
    if g <= -MIN_EFFECT and pair["ba_ci_excludes_zero"]:
        return "worse_beyond_threshold"
    return "below_resolution"


def calibration(y, prob, multiclass, classes, bins=5):
    """Reliability of the probability actually used for abstention. For binary
    fields this is the positive-class probability; for multiclass it is the
    probability assigned to the predicted class."""
    # oof() always returns prob as (n_cases, n_classes), including binary fields.
    if multiclass:
        conf = prob.max(axis=1)
        correct = (np.asarray(classes)[prob.argmax(axis=1)] == y).astype(float)
    else:
        ppos = np.asarray(prob, float)[:, -1]
        correct = ((ppos >= 0.5).astype(float) == y).astype(float)
        conf = np.maximum(ppos, 1.0 - ppos)
    edges = np.linspace(0.5 if not multiclass else 1.0 / len(classes), 1.0, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not m.any():
            rows.append(dict(bin=[round(lo, 3), round(hi, 3)], n=0))
            continue
        rows.append(dict(bin=[round(lo, 3), round(hi, 3)], n=int(m.sum()),
                         mean_confidence=round(float(conf[m].mean()), 4),
                         observed_accuracy=round(float(correct[m].mean()), 4)))
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return dict(bins=rows, expected_calibration_error=round(float(ece), 4))


def abstention(y, p, prob, multiclass, classes):
    """Coverage / accuracy after refusing the least confident cases. A field that
    is unusable at full coverage may still be deliverable if it can say 'I don't
    know' often enough -- and this is where that has to be demonstrated, not
    asserted."""
    # The DECISION must be the ruler's own -- hard(p) -- so that accuracy at full
    # coverage equals the accuracy reported for this arm. The probability is used
    # only to RANK cases by confidence for refusal. Using prob to decide would
    # silently score a different classifier than the one being compared.
    pred = hard(p, multiclass)
    if multiclass:
        conf = prob.max(axis=1)
    else:
        ppos = np.asarray(prob, float)[:, -1]
        conf = np.maximum(ppos, 1.0 - ppos)
    correct = (pred == y).astype(float)
    out = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(y))))
        keep = np.argsort(-conf)[:k]
        out.append(dict(target_coverage=cov, n_kept=int(k),
                        coverage=round(k / len(y), 4),
                        accuracy=round(float(correct[keep].mean()), 4),
                        confidence_threshold=round(float(conf[keep].min()), 4)))
    return out


def loao(y, p, folds_arr, arch_arr, multiclass, anchor_p=None):
    """Leave-one-archive-out in all three directions. Reported for the arm and,
    when an anchor is given, as the arm-minus-anchor difference within each
    held-out archive -- a gain that only exists in one archive is visible here."""
    yp = hard(p, multiclass)
    ap = None if anchor_p is None else hard(anchor_p, multiclass)
    from sklearn.metrics import balanced_accuracy_score as bas
    rows = {}
    for a in sorted(set(arch_arr)):
        m = arch_arr == a
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            rows[str(a)] = dict(n=int(m.sum()), evaluable=False)
            continue
        vals, cnt = np.unique(y[m], return_counts=True)
        const = float(vals[cnt.argmax()])
        r = dict(n=int(m.sum()), evaluable=True,
                 accuracy=round(float((yp[m] == y[m]).mean()), 4),
                 baseline_constant=round(float((y[m] == const).mean()), 4),
                 balanced_accuracy=round(float(bas(y[m], yp[m])), 4))
        r["delta"] = round(r["accuracy"] - r["baseline_constant"], 4)
        if ap is not None:
            r["ba_gain_vs_anchor"] = round(float(bas(y[m], yp[m]) - bas(y[m], ap[m])), 4)
        rows[str(a)] = r
    g = [v.get("ba_gain_vs_anchor") for v in rows.values() if v.get("evaluable")]
    g = [x for x in g if x is not None]
    return dict(by_held_out_archive=rows,
                directions_evaluated=len(g),
                directions_positive=int(sum(1 for x in g if x > 0)),
                directions_clearly_negative=int(sum(1 for x in g if x <= -MIN_EFFECT)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    locked = set(man.index[man.development_fold.isna()])
    if len(locked) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(locked))
    folds = dev.development_fold.to_dict()

    arms, metas = {}, {}
    for rung, (arm, label, _) in LADDER.items():
        ix, feats, meta = load_arm(arm)
        if set(ix.exam_case_id) & locked:
            raise RuntimeError("locked case present in arm %s" % arm)
        if ix.image_path.astype(str).str.contains(r"rep[_0-9]").any():
            raise RuntimeError("report scan reached arm %s" % arm)
        arms[rung] = (ix, feats)
        metas[rung] = dict(arm=arm, label=label, meta=meta)
        print("loaded %-4s %-24s cases=%d images=%d" % (
            rung, arm, ix.exam_case_id.nunique(), len(ix)), flush=True)

    rng = np.random.default_rng(SEED)
    rows, detail = [], []
    for field in IN_SCOPE:
        y_case, k, extra = target(dev, field)
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy()
        classes = sorted(set(float(v) for v in y))
        mc = k > 2
        arch_arr = dev.archive.reindex(order).to_numpy()
        folds_arr = dev.development_fold.reindex(order).to_numpy()

        preds, ok = {}, np.ones(len(order), bool)
        for rung, (ix, f) in arms.items():
            p, pr = oof(f, ix, y_case, folds, order, mc, classes)
            preds[rung] = (p, pr)
            ok &= ~np.isnan(p)
        yv = y[ok]
        unit, static_ok = UNIT[field]

        for rung, (arm, label, base_arm) in LADDER.items():
            p, pr = preds[rung][0][ok], preds[rung][1][ok]
            sc = score(yv, p, pr, mc, classes, rng)
            ref = None
            if base_arm is not None:
                ref = next(r for r, (a, _, _) in LADDER.items() if a == base_arm)
            pair = None if ref is None else paired(
                yv, p, preds[ref][0][ok], mc, rng)
            fd = None if ref is None else fold_directions(
                yv, p, preds[ref][0][ok], folds_arr[ok], mc)
            lo = loao(yv, p, folds_arr[ok], arch_arr[ok], mc,
                      None if ref is None else preds[ref][0][ok])
            v = verdict(pair, sc)
            # A gain against a WEAK rung-0 arm is not a gain for the product. The
            # thing we would actually replace is the shipped A0, so every candidate
            # is additionally re-anchored on A0. malformation_ratio A1L is exactly
            # this case: +0.0571 against its own ViT-L reference, but only +0.0303
            # against A0, because A0L is the weaker starting point.
            vs_a0 = paired(yv, p, preds["A0"][0][ok], mc, rng) if rung != "A0" else None
            if v == "meets_effect_threshold" and vs_a0 is not None \
                    and vs_a0["ba_gain"] < MIN_EFFECT:
                v = "below_resolution_vs_shipped_a0"
            row = dict(field=field, rung=rung, arm=arm, arm_label=label,
                       compared_against=base_arm, n_classes=k, multiclass=mc,
                       unit_of_observation=unit,
                       observable_from_static_image=static_ok,
                       n=sc["n"], mode_class=sc["mode_class"],
                       baseline_constant=sc["baseline_constant"],
                       accuracy=sc["accuracy"], accuracy_ci=sc["accuracy_ci"],
                       delta=sc["delta"], delta_ci=sc["delta_ci"],
                       balanced_accuracy=sc["balanced_accuracy"],
                       balanced_accuracy_ci=sc["balanced_accuracy_ci"],
                       auroc_macro_ovr=sc.get("auroc_macro_ovr"),
                       collapsed_to_one_class=sc["collapsed_to_one_class"],
                       smallest_class=sc["smallest_class"],
                       minority_class_insufficient=sc["minority_class_insufficient"],
                       ba_gain=None if pair is None else pair["ba_gain"],
                       ba_gain_ci=None if pair is None else pair["ba_ci"],
                       ba_ci_excludes_zero=None if pair is None
                       else pair["ba_ci_excludes_zero"],
                       accuracy_gain=None if pair is None else pair["accuracy_gain"],
                       folds_positive=None if fd is None else fd["folds_positive"],
                       folds_negative=None if fd is None else fd["folds_negative"],
                       loao_directions_positive=lo["directions_positive"],
                       loao_directions_clearly_negative=lo["directions_clearly_negative"],
                       ba_gain_vs_shipped_a0=None if vs_a0 is None else vs_a0["ba_gain"],
                       ba_gain_vs_shipped_a0_ci=None if vs_a0 is None else vs_a0["ba_ci"],
                       verdict=v)
            rows.append(row)
            detail.append(dict(row, per_class_recall=sc["per_class_recall"],
                               class_counts=sc["class_counts"],
                               paired=pair, paired_vs_shipped_a0=vs_a0,
                               fold_directions=fd, loao=lo,
                               calibration=calibration(yv, pr, mc, classes),
                               abstention=abstention(yv, p, pr, mc, classes)))
        print("field %-26s n=%-4d done" % (field, int(ok.sum())), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ladder.csv", index=False, encoding="utf-8-sig")
    (OUT / "ladder.json").write_text(json.dumps(dict(
        run="rescue_external_ladder_whole_image",
        question="does external unlabelled adaptation or medical pretraining raise "
                 "any field under the shipped fixed configuration?",
        ruler="imported from run_field_matrix_three_arms; accuracy minus "
              "single-constant-answer baseline, case-level OOF",
        preregistered_effect=dict(min_ba_gain=MIN_EFFECT,
                                  below_resolution_under=BELOW_RESOLUTION,
                                  source="preregistration.md, written before any arm "
                                         "in this file was evaluated"),
        arms=metas, locked_cases_seen=0,
        locked_note="locked-47 was not read, decoded, featurised or selected on in "
                    "this run. This is a statement about THIS run only; the cohort "
                    "has been consumed 7 times historically and model selection has "
                    "happened on it, so it is a consumed internal hold-out, not a "
                    "clean final test set.",
        n_boot=N_BOOT, seed=SEED, fields=detail), ensure_ascii=False, indent=2),
        encoding="utf-8")

    print()
    piv = df[df.compared_against.notna()].pivot_table(
        index="field", columns="rung", values="ba_gain")
    print(piv.round(4).to_string())
    print()
    print(df.verdict.value_counts().to_string())
    print()
    hit = df[df.verdict == "meets_effect_threshold"]
    print("meets the pre-registered threshold AND survives re-anchoring on the "
          "shipped A0: %d" % len(hit))
    if len(hit):
        print(hit[["field", "rung", "ba_gain", "ba_gain_ci",
                   "ba_gain_vs_shipped_a0", "ba_gain_vs_shipped_a0_ci",
                   "loao_directions_clearly_negative"]].to_string(index=False))
    dropped = df[df.verdict == "below_resolution_vs_shipped_a0"]
    if len(dropped):
        print()
        print("passed against its own rung-0 arm but NOT against the shipped A0 "
              "(the gain measures a weak reference, not the adaptation):")
        print(dropped[["field", "rung", "ba_gain",
                       "ba_gain_vs_shipped_a0"]].to_string(index=False))


if __name__ == "__main__":
    main()
