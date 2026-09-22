"""Field verdict and factor attribution over the 15x3 fixed-configuration matrix.

Nothing is refitted here. Every number is a re-reading of the stored OOF
predictions, so the attribution cannot smuggle in a new selection step.

Factors are separated by construction, not by argument:
  capacity_only            ViT-L vs ViT-B, both general pretraining, both at the
                           deployed geometry, same loader -> scale is the only
                           thing that differs.
  medical_pretraining_only BiomedCLIP vs ViT-B. NOT clean: resolution (224 vs
                           518x686), patch (16 vs 14), encoder family and
                           normalisation all differ too, so a BiomedCLIP result
                           is not a fair test of medical pretraining in general.
  loader_geometry_only     recorded as CONFOUNDED INTO the medical contrast for
                           the same reason; the round-one loader control was
                           byte-identical to the anchor and so carried no
                           information.
  target_encoding          how much of a field's score is the recoding rather
                           than the image: measured as the mode baseline itself,
                           since a target compressed onto a dominant class buys
                           accuracy without any visual evidence.
  static_input_mismatch    fields whose unit of observation a still frame cannot
                           carry (time base, or micrometres without calibration).

The verdict is on the ONE ruler: accuracy minus the single-constant-answer
baseline, with its bootstrap CI. BA is reported next to it, never instead of it.

DEVELOPMENT OOF ONLY. Nothing here is a launch claim.

  PYTHONIOENCODING=utf-8 python scripts/build_field_verdict_and_attribution.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
SRC = EXP / "field_matrix_three_arms.json"
ANCHOR = "anchor_dinov2b_deployed"
CAP = "dinov2l_deployed_geometry"
MED = "biomedclip_medical"
TIME_BASE = {"flow_state", "microthrombus"}
NEEDS_CALIBRATION = {"afferent_diameter", "efferent_diameter", "apex_diameter",
                     "loop_length", "capillary_count"}
MIN_MINORITY_RECALL = 0.30    # below this the field is answering with one class


def verdict(r: dict, arch: pd.DataFrame) -> dict:
    """Four-way status on the anchor arm. `signal_only` exists so that a field
    with a real but unshippable result is never written as closed."""
    f = r["field"]
    lo, hi = r["delta_ci"]
    rec = {k: v for k, v in r["per_class_recall"].items()}
    worst = min(rec.values())
    a = arch[(arch.field == f) & (arch.arm == ANCHOR)]
    ba_by_arch = {row.archive: row.ba for row in a.itertuples() if pd.notna(row.ba)}
    # chance BA is 1/k, not 0.5, so a 3-class field is not held to a binary bar
    chance = 1.0 / r["n_classes"]
    all_arch_above_chance = all(v > chance for v in ba_by_arch.values())

    reasons, status = [], None
    if lo > 0 and worst >= MIN_MINORITY_RECALL and all_arch_above_chance:
        status = "usable_on_development"
        reasons.append("delta CI excludes zero, no class is abandoned (worst "
                       "recall %.3f), and BA is above chance (%.3f) in all three "
                       "archives" % (worst, chance))
    elif lo > 0:
        status = "signal_only"
        if worst < MIN_MINORITY_RECALL:
            reasons.append("delta CI excludes zero but the worst per-class recall "
                           "is %.3f, so the gain rests on the dominant class"
                           % worst)
        if not all_arch_above_chance:
            reasons.append("BA is at or below chance (%.3f) in at least one "
                           "archive: %s" % (
                               chance, {k: round(v, 3)
                                        for k, v in ba_by_arch.items()}))
    elif hi > 0:
        status = "signal_only"
        reasons.append("delta %+.3f with CI [%+.3f, %+.3f] spanning zero: the "
                       "result is not distinguishable from the single-constant "
                       "answer at this sample size" % (r["delta"], lo, hi))
    else:
        status = "no_signal_at_this_n"
        reasons.append("delta %+.3f with CI [%+.3f, %+.3f] entirely at or below "
                       "zero" % (r["delta"], lo, hi))

    if r["baseline_constant"] >= 0.75:
        reasons.append("the mode baseline is already %.3f, so accuracy is a weak "
                       "ruler here and delta is the only readable number"
                       % r["baseline_constant"])
    if f in TIME_BASE:
        reasons.append("STATIC INPUT MISMATCH: the unit is a time base, so any "
                       "signal is a correlated static appearance, not an "
                       "observation of flow or of events per minute")
    if f in NEEDS_CALIBRATION:
        reasons.append("the unit needs device calibration (status is "
                       "UNCALIBRATED), so no micrometre or per-mm figure may be "
                       "emitted; only the report's own printed band is admissible")
    if r["smallest_class"] < 25:
        reasons.append("smallest class is %d cases, so per-class recall is "
                       "estimated on very few positives" % r["smallest_class"])
    return dict(field=f, status=status, delta=r["delta"], delta_ci=[lo, hi],
                balanced_accuracy=r["balanced_accuracy"],
                baseline_constant=r["baseline_constant"],
                worst_class_recall=round(float(worst), 4),
                ba_by_archive={k: round(v, 4) for k, v in ba_by_arch.items()},
                reasons=reasons,
                not_a_launch_claim="development OOF only; locked-47 unread")


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    arch = pd.read_csv(EXP / "field_matrix_three_arms_by_archive.csv")
    rows = src["fields"]
    by = {(r["field"], r["arm"]): r for r in rows}
    fields = [r["field"] for r in rows if r["arm"] == ANCHOR]

    verdicts = [verdict(by[(f, ANCHOR)], arch) for f in fields]
    counts = pd.Series([v["status"] for v in verdicts]).value_counts().to_dict()

    # ---- factor attribution, by re-pairing the stored predictions only ----
    attribution = {}
    for f in fields:
        c, m = by[(f, CAP)]["paired_vs_anchor"], by[(f, MED)]["paired_vs_anchor"]
        cf, mf = (by[(f, CAP)]["fold_directions_vs_anchor"],
                  by[(f, MED)]["fold_directions_vs_anchor"])
        attribution[f] = dict(
            capacity_only=dict(
                contrast="ViT-L vs ViT-B, both general, both deployed geometry",
                ba_gain=c["ba_gain"], ba_ci=c["ba_ci"],
                ci_excludes_zero=c["ba_ci_excludes_zero"],
                folds_positive=cf["folds_positive"],
                folds_negative=cf["folds_negative"]),
            medical_pretraining_only=dict(
                contrast="BiomedCLIP vs ViT-B",
                ba_gain=m["ba_gain"], ba_ci=m["ba_ci"],
                ci_excludes_zero=m["ba_ci_excludes_zero"],
                folds_positive=mf["folds_positive"],
                folds_negative=mf["folds_negative"],
                confounded_by=["input 224x224 vs 518x686", "patch 16 vs 14",
                               "encoder family and pretraining objective",
                               "CLIP normalisation vs ImageNet",
                               "aspect-preserving pad vs direct resize"],
                fair_test_of_medical_pretraining=False),
            target_encoding=dict(
                mode_baseline=by[(f, ANCHOR)]["baseline_constant"],
                n_classes=by[(f, ANCHOR)]["n_classes"],
                note="the baseline is what the recoding alone buys; a high "
                     "baseline means most of the accuracy is the encoding, not "
                     "the image"),
            static_input_mismatch=(
                "time base" if f in TIME_BASE else
                "micrometre/per-mm unit without device calibration"
                if f in NEEDS_CALIBRATION else "none"))

    cap_sig = [f for f in fields
               if attribution[f]["capacity_only"]["ci_excludes_zero"]]
    med_sig = [f for f in fields
               if attribution[f]["medical_pretraining_only"]["ci_excludes_zero"]]
    med_pos = [f for f in med_sig
               if attribution[f]["medical_pretraining_only"]["ba_gain"] > 0]
    med_neg = [f for f in med_sig
               if attribution[f]["medical_pretraining_only"]["ba_gain"] < 0]
    cap_pos = [f for f in cap_sig
               if attribution[f]["capacity_only"]["ba_gain"] > 0]
    cap_neg = [f for f in cap_sig
               if attribution[f]["capacity_only"]["ba_gain"] < 0]

    access = json.loads((EXP / "medical_candidate_access.json").read_text(
        encoding="utf-8"))
    out = dict(
        run="field_verdict_and_attribution",
        source=str(SRC.name), refitted=False,
        configuration=src["configuration"],
        status_counts=counts, field_verdicts=verdicts,
        attribution=attribution,
        attribution_summary=dict(
            capacity_only=dict(
                fields_with_ci_excluding_zero=cap_sig, positive=cap_pos,
                negative=cap_neg,
                reading="%d of %d fields; scaling the general encoder from B to L "
                        "changes almost nothing on this ruler" % (len(cap_sig),
                                                                 len(fields))),
            medical_pretraining_only=dict(
                fields_with_ci_excluding_zero=med_sig, positive=med_pos,
                negative=med_neg,
                reading="BiomedCLIP is the only medical candidate that ran, and "
                        "it is confounded on resolution, patch size and encoder "
                        "family, so neither its wins nor its losses settle "
                        "whether medical pretraining helps"),
            loader_geometry_only=dict(
                available=False,
                reason="round one's loader control extracted byte-identical "
                       "features to the anchor, so it carried no information; "
                       "the geometry difference now sits inside the BiomedCLIP "
                       "contrast and cannot be separated from it"),
            target_encoding=dict(
                reading="fields whose mode baseline is already high (%s) owe most "
                        "of their accuracy to the recoding, not to the image"
                        % ", ".join("%s %.2f" % (f, attribution[f]
                                                 ["target_encoding"]["mode_baseline"])
                                    for f in fields
                                    if attribution[f]["target_encoding"]
                                    ["mode_baseline"] >= 0.75)),
            static_input_mismatch=dict(
                time_base=sorted(TIME_BASE),
                needs_calibration=sorted(NEEDS_CALIBRATION))),
        medical_candidate_access={k: v["verdict"]
                                  for k, v in access["gated_candidates"].items()},
        medical_pretraining_conclusion=access["conclusion"],
        forbidden_conclusions=[
            "医学视觉预训练无效 -- RETFound and MedSigLIP never ran; they are "
            "blocked_access, not rejected",
            "医学视觉预训练已验证 -- BiomedCLIP alone cannot establish it",
            "any 15x3 number as launch capability -- these are development OOF",
            "any banded measurement result as a micrometre measurement capability",
            "any flow_state or microthrombus result as flow or event observation",
        ],
        next_step_gate="stop here: no LoRA, adapter, video model or per-field "
                       "model selection before this matrix is read")
    (EXP / "field_verdict_and_attribution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(verdicts).drop(columns=["reasons", "ba_by_archive"]).to_csv(
        EXP / "field_verdict.csv", index=False, encoding="utf-8-sig")

    print("status counts:", counts)
    for v in verdicts:
        print("  %-27s %-22s d%+.3f[%+.3f,%+.3f] ba %.3f worst_rec %.3f" % (
            v["field"], v["status"], v["delta"], v["delta_ci"][0],
            v["delta_ci"][1], v["balanced_accuracy"], v["worst_class_recall"]))
    print("\ncapacity CI-excluding-zero:", cap_sig, "pos", cap_pos, "neg", cap_neg)
    print("medical  CI-excluding-zero:", med_sig, "pos", med_pos, "neg", med_neg)
    print("medical candidates:", out["medical_candidate_access"])


if __name__ == "__main__":
    main()
