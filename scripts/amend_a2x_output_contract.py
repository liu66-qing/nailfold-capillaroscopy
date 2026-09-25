"""Amendment 1 to the frozen A2x candidate: correct the output semantics.

Two things were wrong in the freeze, both found by the user's review:

1. The freeze's output_semantics permitted "ordinal correspondence with the
   report's low / medium / high malformation wording". The model is BINARY.
   ADMISSIBLE["malformation_ratio"] is "binary <=10% vs >10%", with class 1
   collapsing 10--30%, 30--60% and >60% into one bucket. A three-band output
   from a two-class model would assert a distinction the model never learned --
   it cannot tell 15% from 70%, because those were the same training label.

2. Adoption criterion 6 (the project's 70% coverage + 85% accuracy bar) was
   treated as satisfied. It is not: at 69.75% coverage the frozen candidate's
   accuracy is 0.7876, which is 0.0624 short. The abstention table was in the
   freeze all along; the criterion was simply scored wrong.

So "8/8" becomes "7/8 research-candidate conditions", and the headline status
becomes strong development candidate / not clinically validated / not yet shipped.

The superseded text is copied into this amendment verbatim rather than deleted,
so the record shows what was claimed and what replaced it. The frozen numbers,
weights, predictions and LOAO are untouched -- this changes only what the module
is permitted to SAY.

  PYTHONIOENCODING=utf-8 python scripts/amend_a2x_output_contract.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
       / "frozen_candidate_malformation_a2x")

# The four permitted output values. Two of them are not predictions at all --
# unknown and rejected exist so that "the model did not answer" can never be
# rendered as "the model found nothing wrong".
PERMITTED_VALUES = {
    "low_band_correspondence": (
        "the image corresponds to the report's <=10% malformation band"),
    "high_band_correspondence": (
        "the image corresponds to the report's >10% malformation band"),
    "unknown": (
        "no usable reading -- NOT a normal finding, and must never be rendered "
        "as one"),
    "rejected": (
        "input or confidence failed a precondition; the field is withheld"),
}

FORBIDDEN_OUTPUTS = [
    "any malformed-vessel percentage (e.g. 'malformed vessels 23%')",
    "any per-mm value (条/mm)",
    "any micron value",
    "'the patient has <disease>'",
    "'<disease> is ruled out'",
    "any three-band low/medium/high wording -- the model is binary and cannot "
    "separate 10--30% from >60%; those were one training label",
]


def main() -> None:
    p = OUT / "frozen_candidate.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    superseded_semantics = j.get("output_semantics")
    superseded_licence = j.get("external_data_licence", {}).get(
        "adoption_criterion_8")

    abst = {round(float(r["coverage"]), 4): r for r in j["abstention"]}
    at70 = [r for r in j["abstention"] if r["target_coverage"] == 0.7]
    if len(at70) != 1:
        raise RuntimeError("expected exactly one 70%% coverage row")
    at70 = at70[0]
    # Re-derived from the frozen table rather than retyped, so the shortfall
    # cannot drift away from the numbers the candidate actually produced.
    bar_cov, bar_acc = 0.70, 0.85
    shortfall = round(bar_acc - float(at70["accuracy"]), 4)
    if shortfall <= 0:
        raise RuntimeError("this amendment exists because the bar was NOT met; "
                           "the table now says it was, so re-check by hand")

    # The lowest coverage in the frozen table still does not reach 0.85, which is
    # the honest way to answer "then just abstain harder".
    best = max(j["abstention"], key=lambda r: r["accuracy"])
    reachable = bool(best["accuracy"] >= bar_acc)

    j["output_semantics"] = dict(
        model_form="binary classifier",
        decision_boundary="report band <=10% (class 0) vs >10% (class 1)",
        class1_collapses=["10--30%", "30--60%", ">60%"],
        permitted_values=PERMITTED_VALUES,
        forbidden_outputs=FORBIDDEN_OUTPUTS,
        why_not_three_bands=(
            "ADMISSIBLE['malformation_ratio'] is 'binary <=10% vs >10%'. The "
            "three abnormal bands are one training label, so the model has never "
            "been shown the difference between them and cannot report it."),
        why_not_a_percentage=(
            "human boxes fall inside their own clinical band for 17 of 44 cases "
            "with roughly a 3x scale offset, so a percentage would assert a "
            "calibration that does not exist"),
        status_line="image correlation, not a diagnosis",
        superseded_by_this_amendment=superseded_semantics,
    )

    j["status"] = dict(
        headline="strong development candidate / not clinically validated / "
                 "not yet shipped",
        development_candidate=True,
        clinically_validated=False,
        shipped=False,
        external_same_definition_test_set="none exists yet",
        previous_status=j.get("status"),
    )

    j["adoption_criteria"] = dict(
        scope=("these are RESEARCH-CANDIDATE conditions on development data. "
               "Passing them all would still not be a launch decision."),
        criterion_6_product_bar=dict(
            bar="70% coverage with 85% accuracy (project engineering standard, "
                "not a regulatory or field-wide threshold)",
            observed_coverage=at70["coverage"],
            observed_accuracy=at70["accuracy"],
            n_kept=at70["n_kept"],
            confidence_threshold=at70["confidence_threshold"],
            shortfall=shortfall,
            passes=False,
            note=("the earlier record scored this as satisfied; it was not. The "
                  "abstention table was already in the freeze."),
            bar_reachable_by_abstaining_further=reachable,
            best_accuracy_in_table=best["accuracy"],
            best_accuracy_coverage=best["coverage"],
        ),
        criterion_8_licence=dict(
            declared_licence="apache-2.0", passes=True,
            previous_wording=superseded_licence),
        tally="7 of 8 research-candidate conditions; criterion 6 fails",
        must_not_be_written_as="8/8, or as a launch approval",
    )

    j["amendments"] = j.get("amendments", []) + [dict(
        n=1, date=str(date.today()),
        by="user review of execution record 9",
        changes=[
            "output semantics corrected from three ordinal bands to the binary "
            "band correspondence the model actually learned",
            "unknown and rejected added as first-class output values",
            "adoption criterion 6 re-scored as FAILING (0.7876 at 69.75% "
            "coverage against an 85% bar)",
            "'8/8' retracted; it is 7/8 research-candidate conditions",
            "status set to strong development candidate / not clinically "
            "validated / not yet shipped",
        ],
        unchanged=["weights", "per-case predictions", "development OOF metrics",
                   "both LOAO tables", "calibration", "abstention table"],
    )]

    p.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(
        criterion_6=j["adoption_criteria"]["criterion_6_product_bar"],
        tally=j["adoption_criteria"]["tally"],
        status=j["status"]["headline"],
        permitted=sorted(PERMITTED_VALUES),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
