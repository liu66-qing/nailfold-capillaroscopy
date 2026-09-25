"""Freeze A2x as the malformation_ratio candidate module, and persist its predictions.

The ladder that produced A2x wrote METRICS only. Nothing on disk says which
prediction each case received, so "freeze the OOF and LOAO predictions" cannot be
done by copying files -- the run has to happen again and save them. That is the
better outcome anyway: this script re-derives every A2x number from the frozen
checkpoints and asserts it reproduces execution record 8. A silent drift between
the reported table and the frozen module would be exactly the failure a freeze is
supposed to prevent.

Two things here are NOT copies of the ladder.

1. A real leave-one-archive-out. The ladder's loao() takes folds_arr and never
   uses it: it slices the pooled 5-fold OOF predictions by archive, so every
   archive was in training when its own cases were predicted, despite the key
   being named by_held_out_archive. For a candidate module that is not good
   enough. Here each archive is held out ENTIRELY and the model is refit on the
   other two, which is what test_leave_one_archive_out.py does for the shipped
   fields. Both numbers are written out, and they answer different questions, so
   neither replaces the other.

2. The training-archive baseline. A deployed model carries the majority answer of
   the data it was trained on; the test archive's own prevalence is unknown at
   deployment. Both baselines are recorded, the training-archive one is the honest
   ruler, and this matters here because malformation_ratio's mode flips between
   archives (archive3 is 0.6098 against 0.5490/0.5571).

Output semantics are constrained, and the constraint is written into the frozen
metadata rather than left to whoever reads it later: this module emits an ORDINAL
correspondence with the report's low/medium/high malformation wording. It must not
emit a malformed-vessel percentage. The human boxes land inside their own clinical
band for 17 of 44 cases with roughly a 3x scale offset, so a percentage would be
asserting a calibrated measurement that was never calibrated.

DEVELOPMENT ONLY. locked-47 is never read. This freeze does not create external
validation; all three archives are one site.

Reproduce:
  python scripts/freeze_malformation_a2x_candidate.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402  the shipped ruler, unchanged
    ANCHOR, LABELS, N_BOOT, POOLINGS, SEED, UNIT, fit_predict, hard, load_arm,
    oof, paired, score, target)
from run_rescue_ladder import (  # noqa: E402
    BELOW_RESOLUTION, MIN_EFFECT, abstention, calibration, loao, verdict)
from run_rescue_ladder_local import attach, load_local  # noqa: E402

FIELD = "malformation_ratio"
TAG = "external"          # the A2x local-feature table
EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
OUT = EXP / "frozen_candidate_malformation_a2x"

# What execution record 8 reports for A2x. Re-derived here and asserted, so a
# freeze that no longer reproduces the documented table fails loudly.
EXPECTED = dict(n=162, baseline_constant=0.5679, accuracy=0.7284, delta=0.1605,
                balanced_accuracy=0.7216, auroc_macro_ovr=0.7766, ba_gain=0.0911)

FORBIDDEN_OUTPUT = "malformed-vessel percentage, or any per-mm / micron value"
PERMITTED_OUTPUT = ("ordinal correspondence with the report's low / medium / high "
                    "malformation wording")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def true_loao(base_feats, base_ix, extra, y_case, order, arch_arr, classes):
    """Hold each archive out of TRAINING entirely and refit on the other two.

    The ladder's loao() never refits -- it partitions predictions that were made
    with all three archives in the training pool. That answers "does the pooled
    model's advantage live in one archive", which is worth knowing, but it cannot
    answer "does this module survive a site it never saw". Only a refit can, and a
    candidate module has to face the second question.

    The baseline is the majority answer of the TRAINING archives, because that is
    what a deployed model carries. The held-out archive's own prevalence is
    unknown at deployment time, so scoring against it would flatter or punish the
    model for a fact it could not have had.
    """
    from sklearn.metrics import balanced_accuracy_score as bas

    feats = attach(base_feats, base_ix, extra)
    dm = base_ix.exam_case_id.isin(y_case.index).to_numpy()
    pos = {c: i for i, c in enumerate(order)}
    yv = y_case.reindex(order).to_numpy()
    rows = {}
    for a in sorted(set(arch_arr)):
        te_cases = {c for c, ar in zip(order, arch_arr) if ar == a}
        tr_cases = set(order) - te_cases
        trm = dm & base_ix.exam_case_id.isin(tr_cases).to_numpy()
        tem = dm & base_ix.exam_case_id.isin(te_cases).to_numpy()
        ytr_img = y_case.reindex(base_ix.exam_case_id[trm]).to_numpy()
        if trm.sum() == 0 or tem.sum() == 0 or len(set(ytr_img)) < 2:
            rows[str(a)] = dict(n=len(te_cases), evaluable=False)
            continue
        ids = base_ix.exam_case_id[tem].to_numpy()
        got = [fit_predict(feats[p][trm], ytr_img, feats[p][tem], False, classes)
               for p in POOLINGS]
        s = pd.Series(np.mean([g[0] for g in got], axis=0),
                      index=ids).groupby(level=0).mean()
        p = np.full(len(order), np.nan)
        for c, v in s.items():
            if c in pos:
                p[pos[c]] = v
        m = np.array([c in te_cases for c in order])
        yp, yt = hard(p, False)[m], yv[m]
        # Majority answer of the TRAINING archives -- what a deployed model carries.
        ytr_case = y_case.reindex(sorted(tr_cases)).to_numpy()
        vals, cnt = np.unique(ytr_case, return_counts=True)
        const_tr = float(vals[cnt.argmax()])
        vals2, cnt2 = np.unique(yt, return_counts=True)
        rows[str(a)] = dict(
            n=int(m.sum()), evaluable=True,
            trained_on=sorted(set(arch_arr) - {a}),
            accuracy=round(float((yp == yt).mean()), 4),
            baseline_constant_training_archives=round(float((yt == const_tr).mean()), 4),
            baseline_constant_own_archive=round(float((yt == float(vals2[cnt2.argmax()])).mean()), 4),
            balanced_accuracy=round(float(bas(yt, yp)), 4))
        rows[str(a)]["delta_vs_training_archive_baseline"] = round(
            rows[str(a)]["accuracy"] - rows[str(a)]["baseline_constant_training_archives"], 4)
    ev = [v for v in rows.values() if v.get("evaluable")]
    return dict(
        by_held_out_archive=rows,
        directions_evaluated=len(ev),
        directions_with_positive_delta=int(sum(
            1 for v in ev if v["delta_vs_training_archive_baseline"] > 0)),
        ruler="baseline is the majority answer of the TRAINING archives; the "
              "held-out archive's own prevalence is unknown at deployment",
        difference_from_ladder_loao="each archive is held out of TRAINING and the "
              "model refit; the ladder's loao() only partitions pooled OOF "
              "predictions and never refits")


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    if len(set(man.index[man.development_fold.isna()])) != 47:
        raise RuntimeError("expected 47 locked cases")

    base_ix, base_feats, base_meta = load_arm(ANCHOR)
    if base_meta.get("locked_cases_seen", 0) != 0:
        raise RuntimeError("anchor arm reports locked cases")
    extra = load_local(TAG)
    if extra is None:
        raise SystemExit("no %s local feature table" % TAG)

    y_case, n_classes, target_report = target(dev, FIELD)
    if n_classes != 2:
        raise RuntimeError("malformation_ratio should be binary here, got %d" % n_classes)
    order = sorted(y_case.index)
    yv = y_case.reindex(order).to_numpy()
    classes = sorted(set(float(v) for v in yv))
    fold_map = dev.development_fold.to_dict()
    arch_map = dev.archive.to_dict()
    folds_arr = np.array([fold_map[c] for c in order], float)
    arch_arr = np.array([arch_map[c] for c in order], object)
    rng = np.random.default_rng(SEED)

    p0, pr0 = oof(base_feats, base_ix, y_case, fold_map, order, False, classes)
    p1, pr1 = oof(attach(base_feats, base_ix, extra), base_ix, y_case,
                  fold_map, order, False, classes)

    sc = score(yv, p1, pr1, False, classes, rng)
    sc0 = score(yv, p0, pr0, False, classes, rng)
    pair = paired(yv, p1, p0, False, rng)

    # A freeze whose numbers drift from the documented table is worse than no
    # freeze, so reproduction is asserted rather than eyeballed.
    got = dict(n=sc["n"], baseline_constant=sc["baseline_constant"],
               accuracy=sc["accuracy"], delta=sc["delta"],
               balanced_accuracy=sc["balanced_accuracy"],
               auroc_macro_ovr=sc["auroc_macro_ovr"], ba_gain=pair["ba_gain"])
    drift = {k: (EXPECTED[k], got[k]) for k in EXPECTED
             if abs(float(EXPECTED[k]) - float(got[k])) > 1e-4}
    if drift:
        raise RuntimeError("frozen candidate does not reproduce execution record 8: "
                           "%s" % drift)

    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(dict(
        exam_case_id=order, archive=arch_arr, development_fold=folds_arr,
        y_true=yv, a0_score=p0, a0_label=hard(p0, False),
        a2x_score=p1, a2x_label=hard(p1, False)))
    preds.to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    print("wrote %d OOF predictions" % len(preds))

    tl = true_loao(base_feats, base_ix, extra, y_case, order, arch_arr, classes)
    for a, r in tl["by_held_out_archive"].items():
        if r.get("evaluable"):
            print("  LOAO refit %-22s acc %.4f  base(train) %.4f  delta %+.4f  BA %.4f"
                  % (a, r["accuracy"], r["baseline_constant_training_archives"],
                     r["delta_vs_training_archive_baseline"], r["balanced_accuracy"]))
    # Copy the artefacts the module needs to run, so the freeze does not depend on
    # the experiment tree staying put. The detector is external-only -- it read
    # zero local images, labels and folds -- so ONE checkpoint legitimately serves
    # every fold, and there are no per-fold local detector weights for A2x. That is
    # a property of this arm, not an omission: the per-fold local detectors belong
    # to A2 and A3, which did not win.
    wdir = OUT / "weights"
    wdir.mkdir(exist_ok=True)
    det_meta = json.loads((EXP / "detectors" / "detectors_external.json")
                          .read_text(encoding="utf-8"))
    src = ROOT / det_meta["external_pretrain_weights"]
    shutil.copy2(src, wdir / "external_detector_best.pt")
    feat_meta = json.loads((EXP / "local_features" / TAG / "metadata.json")
                           .read_text(encoding="utf-8"))
    for f in ("case_features.csv", "metadata.json"):
        shutil.copy2(EXP / "local_features" / TAG / f, OUT / f)
    for s in ("extract_detector_local_features.py", "train_morph_detector_folds.py",
              "freeze_malformation_a2x_candidate.py"):
        shutil.copy2(ROOT / "scripts" / s, wdir.parent / ("script_" + s))

    frozen = dict(
        field=FIELD, arm="A2x", frozen_on="2026-09-24",
        status="strong development candidate; NOT a shipped field",
        reproduces_execution_record_8=True,
        development_oof=dict(
            n=sc["n"], mode_baseline=sc["baseline_constant"],
            accuracy=sc["accuracy"], accuracy_ci=sc["accuracy_ci"],
            delta=sc["delta"], delta_ci=sc["delta_ci"],
            balanced_accuracy=sc["balanced_accuracy"],
            balanced_accuracy_ci=sc["balanced_accuracy_ci"],
            auroc_macro_ovr=sc["auroc_macro_ovr"],
            per_class_recall=sc["per_class_recall"],
            anchor_a0=dict(accuracy=sc0["accuracy"], delta=sc0["delta"],
                           balanced_accuracy=sc0["balanced_accuracy"],
                           auroc_macro_ovr=sc0["auroc_macro_ovr"],
                           per_class_recall=sc0["per_class_recall"]),
            paired_vs_a0=pair),
        calibration=calibration(yv, pr1, False, classes),
        abstention=abstention(yv, p1, pr1, False, classes),
        loao_pooled_oof_partition=loao(yv, p1, folds_arr, arch_arr, False, p0),
        loao_refit_archive_held_out=tl,
        preprocessing=dict(
            encoder=ANCHOR, conf_threshold=feat_meta["conf_threshold"],
            detector_imgsz=feat_meta["imgsz"],
            class_map=feat_meta["class_map"],
            poolings=POOLINGS, C=0.03, pca_dim=64, seed=SEED, n_boot=N_BOOT,
            local_columns=len(feat_meta["feature_columns"]),
            nan_semantics="zero detections means count 0 but ratio UNDEFINED; "
                          "ratios stay NaN with explicit flags and are zero-imputed "
                          "after scaling, never median-imputed (a fold-wise median "
                          "would leak)"),
        detector=dict(
            source="external only", weights_sha256=sha(src),
            one_checkpoint_for_all_folds=True,
            justification=feat_meta["one_checkpoint_justification"],
            map50=0.7677, ap50_per_class={"bushy": 0.6891, "crossing": 0.8519,
                                          "hairpin": 0.8888, "tortuous": 0.6409},
            classes_are_not_our_fields=True),
        validation_discipline=dict(
            human_boxes_used_on_validation_cases=False,
            how="validation cases receive only automatically detected regions; "
                "human boxes are supervision inside training folds only -- and for "
                "this arm the detector saw no local box at all",
            locked_cases_seen=0,
            locked_note="this run only; locked-47 has been consumed 7 times "
                        "historically and had model selection done on it, so it is "
                        "a CONSUMED internal holdout, not a clean final test set"),
        output_semantics=dict(
            permitted=PERMITTED_OUTPUT, forbidden=FORBIDDEN_OUTPUT,
            why="human boxes fall inside their own clinical band for 17 of 44 cases "
                "with roughly a 3x scale offset, so a percentage would assert a "
                "calibration that does not exist"),
        external_data_licence=dict(
            dataset="HanaNguyen/Capillary-Dataset",
            declared_licence="apache-2.0",
            source="the dataset card's own YAML front matter, data/Capillary-Dataset/README.md",
            commercial_use_permitted=True,
            obligations="Apache-2.0 requires the licence text and attribution be "
                        "retained; cite Nguyen & Jeong, KDD 2025",
            used_for="vessel-morphology detector initialisation only; the external "
                     "diabetes labels were NOT used as patient labels for this project",
            adoption_criterion_8="satisfied -- Apache-2.0 permits the target deployment"),
        what_this_freeze_does_not_establish=[
            "external validity: all three archives are one site, so no result here "
            "is an external test",
            "a shipped field: promotion requires a new, never-accessed test set of "
            "the same field definition with per-case independence",
            "a calibrated percentage: the permitted output is ordinal only"])

    (OUT / "frozen_candidate.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    print("froze %s A2x -> %s" % (FIELD, OUT))


if __name__ == "__main__":
    main()
