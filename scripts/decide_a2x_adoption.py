"""Item 8: the adoption decision for A2x, evaluated mechanically.

The five conditions were set before the locked read. This script reads them off the
frozen artefacts and reports which hold, so the verdict is not an argument written
after seeing the numbers. Failing a condition is a legal outcome and is recorded as
one; the script does not try to find a reading under which A2x passes.

  PYTHONIOENCODING=utf-8 PYTHONPATH=scripts python scripts/decide_a2x_adoption.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
          / "frozen_candidate_malformation_a2x")
CONTRACT = ROOT / "artifacts" / "experiments" / "product_contract_20260924"
LOCKED = ROOT / "artifacts" / "experiments" / "locked_consumed_20260924"


def read(p: Path) -> dict:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    frozen = read(FROZEN / "frozen_candidate.json")
    refit = read(FROZEN / "refit_loao_controlled.json")
    safety = read(CONTRACT / "rag_safety_tests.json")
    contract = read(CONTRACT / "field_output_contract.json")
    locked = read(LOCKED / "locked_consumed_evaluation.json")
    heads = read(CONTRACT / "final_heads" / "manifest.json")

    # 1. Refit LOAO: no material regression against the A0 anchor.
    arch = refit["per_archive"]
    gains = {a: d["a2x_minus_a0"]["ba_gain"] for a, d in arch.items()}
    c1 = dict(
        condition="refit LOAO shows no clear regression vs A0",
        per_archive_ba_gain=gains,
        n_archives_positive=sum(g > 0 for g in gains.values()),
        n_ci_excluding_zero=sum(d["a2x_minus_a0"]["ba_ci_excludes_zero"]
                                for d in arch.values()),
        passes=all(g > -0.05 for g in gains.values()),
    )

    # 2. No class collapse on the locked read, in either arm.
    c2 = dict(
        condition="no class collapse on the consumed holdout",
        a2x_collapsed=locked["a2x"]["collapsed_to_one_class"],
        a0_collapsed=locked["a0"]["collapsed_to_one_class"],
        a2x_per_class_recall=locked["a2x"]["per_class_recall"],
        passes=not locked["a2x"]["collapsed_to_one_class"],
    )

    # 3. Output semantics stayed a binary ordinal hint.
    sem = frozen["output_semantics"]
    permitted = sorted(sem["permitted_values"])
    c3 = dict(
        condition="output semantics remain a binary band correspondence",
        model_form=sem["model_form"], permitted_values=permitted,
        percentage_forbidden=any("percentage" in f for f in sem["forbidden_outputs"]),
        passes=(sem["model_form"] == "binary classifier"
                and permitted == ["high_band_correspondence",
                                  "low_band_correspondence", "rejected", "unknown"]),
    )

    # 4. The 70% coverage + 85% accuracy bar. If unmet, an explicit lower-coverage
    #    refusal policy must exist -- and must not be presented as meeting the bar.
    crit6 = frozen["adoption_criteria"]["criterion_6_product_bar"]
    rule = heads["decision_rule"]
    c4 = dict(
        condition="either the 70/85 bar is met, or an explicit lower-coverage "
                  "refusal policy is in force and the bar is not claimed",
        bar_met=bool(crit6["passes"]),
        development_coverage=rule["development_coverage_at_gate"],
        development_accuracy=rule["development_accuracy_at_gate"],
        locked_coverage_at_frozen_gate=next(
            r["coverage"] for r in locked["coverage_accuracy_curve"]
            if "at_frozen_threshold" in r),
        locked_accuracy_at_frozen_gate=next(
            r["accuracy"] for r in locked["coverage_accuracy_curve"]
            if "at_frozen_threshold" in r),
        refusal_policy_in_force=rule["abstain_below"],
        bar_not_claimed=("does not meet" in rule["bar_note"]),
        passes=bool(crit6["passes"]) or (rule["abstain_below"] is not None
                                         and "does not meet" in rule["bar_note"]),
        passes_via=("bar_met" if crit6["passes"]
                    else "explicit_lower_coverage_refusal_policy"),
        qualification=(
            "this is the weak branch of the condition. The bar is NOT met: 0.7876 at "
            "69.75% coverage on development, and on the consumed holdout the best "
            "point on the whole curve is 0.70 at 51% coverage -- no amount of "
            "refusing reaches 85% on either set. The condition holds only in the "
            "sense the user specified: a lower-coverage refusal policy exists and "
            "the bar is not claimed. It must never be reported as the bar being met."),
        best_locked_accuracy_any_coverage=max(
            r["accuracy"] for r in locked["coverage_accuracy_curve"]
            if r.get("accuracy") is not None),
    )

    # 5. All RAG safety rules pass, with the mutation control showing they have teeth.
    c5 = dict(
        condition="all RAG safety rules pass and the mutation control catches every "
                  "injected defect",
        n_passed=safety["summary"]["n_passed"], n_cases=safety["summary"]["n_cases"],
        n_mutants_caught=safety["summary"]["n_mutants_caught"],
        n_mutants=safety["summary"]["n_mutants"],
        passes=(safety["summary"]["all_passed"]
                and safety["summary"]["n_mutants_caught"]
                == safety["summary"]["n_mutants"]),
    )

    # 6. The image-correlation label must be attached, not merely intended.
    spec = contract["fields"]["malformation_ratio"]
    c6 = dict(
        condition="labelled image correlation, not diagnosis",
        status=spec["status"], not_a_diagnosis=spec["not_a_diagnosis"],
        passes=(spec["status"] == "image_correlation"
                and spec["not_a_diagnosis"] is True),
    )

    conds = {"refit_loao_no_regression": c1, "no_class_collapse": c2,
             "binary_ordinal_semantics": c3, "coverage_accuracy_policy": c4,
             "rag_safety": c5, "image_correlation_label": c6}
    failed = sorted(k for k, c in conds.items() if not c["passes"])

    # The conditions gate entry as a first-version RAG candidate. They do NOT gate
    # "shipped": the bar was missed, the holdout is consumed, and there is no external
    # validation or clinician annotation, so the ceiling on any verdict here is
    # research / internal-education candidate.
    if failed:
        role = "not_a_candidate"
    else:
        role = "first_version_rag_candidate_not_shipped"
    weak = sorted(k for k, c in conds.items()
                  if c["passes"] and c.get("passes_via", "").startswith("explicit"))

    verdict = dict(
        run="a2x_adoption_decision_20260924",
        field="malformation_ratio", arm="A2x",
        conditions=conds,
        conditions_failed=failed,
        all_conditions_hold=not failed,
        conditions_holding_only_via_the_weak_branch=weak,
        honest_reading=("6 of 6 hold, but coverage_accuracy_policy holds only "
                        "because the user allowed an explicit lower-coverage refusal "
                        "in place of the bar. The original 70% coverage + 85% "
                        "accuracy target is not met on development or on the "
                        "holdout, at any coverage point."),
        role=role,
        status_line=("strong development candidate / not clinically validated / "
                     "not yet shipped"),
        may_be_called=["image correlation", "research candidate",
                       "internal health-education candidate"],
        may_not_be_called=["shipped model", "validated model", "diagnostic model",
                           "clinically confirmed", "meets the 70/85 bar",
                           "externally validated"],
        headline_numbers=dict(
            development_oof=dict(
                n=frozen["development_oof"]["n"],
                accuracy=frozen["development_oof"]["accuracy"],
                balanced_accuracy=frozen["development_oof"]["balanced_accuracy"],
                delta_vs_mode=frozen["development_oof"]["delta"]),
            consumed_holdout=dict(
                n=locked["n"], accuracy=locked["a2x"]["accuracy"],
                balanced_accuracy=locked["a2x"]["balanced_accuracy"],
                auroc=locked["a2x"]["auroc"],
                delta_to_quote=locked["prevalence_flip_warning"][
                    "delta_vs_locked_own_mode"],
                class1_recall=locked["a2x"]["per_class_recall"]["1"],
                ece=locked["a2x"]["ece"]),
            cross_archive_refit=gains,
        ),
        what_the_holdout_showed=[
            "accuracy 0.6154 and BA 0.6310 on 39 labelled locked cases, down 0.1130 "
            "and 0.0906 from development -- the drop is kept, not explained away",
            "A2x still ranks above A0 (BA +0.0476) but the interval includes zero on "
            "39 cases, so the cross-archive advantage is not reconfirmed here",
            "class 1 recall 0.4286: more than half the above-band cases are missed",
            "ECE 0.1570: the probabilities are overconfident on this set",
            "the honest single-constant delta is +0.0769, not the +0.1539 that the "
            "development mode would suggest, because the majority class flips",
        ],
        no_model_change=("the read was authorised on the condition that the result "
                        "stands. No threshold, feature or head was changed after "
                        "these numbers were seen."),
        remaining_blockers_not_solvable_by_modelling=[
            "no external test set collected under the same definitions",
            "no independent clinician annotation, so clinical correctness is untested",
            "locked-47 is consumed and cannot serve as a clean test again",
            "low-prevalence PPV is unaddressed; at 5% prevalence this would be a "
            "rule-out tool at best, and class 1 recall 0.4286 undermines even that",
        ],
    )
    with open(LOCKED / "a2x_adoption_decision.json", "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    print("A2x adoption: %d/%d conditions hold -> %s"
          % (len(conds) - len(failed), len(conds), role))
    for k, c in conds.items():
        print("  %s %s" % ("ok  " if c["passes"] else "FAIL", k))
    if failed:
        print("  failed: %s" % ", ".join(failed))
    if weak:
        print("  weak branch only: %s (the 70/85 bar is NOT met)" % ", ".join(weak))
    print("wrote %s" % (LOCKED / "a2x_adoption_decision.json"))


if __name__ == "__main__":
    main()
