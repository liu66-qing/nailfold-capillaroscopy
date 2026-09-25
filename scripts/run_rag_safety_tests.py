"""The eight RAG safety tests, as a repeatable JSON test set and a rule report.

What this establishes: that for every enumerated input shape, the advice layer does
not turn an image correlation into a disease claim, does not render a missing reading
as a clean one, and does not state a fixed-mode field as a fact about the patient.

What this does NOT establish -- stated here so the artefact cannot be read as more
than it is: that the advice is clinically correct, that the advice is clinically
useful, or that the field readings driving it are accurate. Those need independent
clinician annotation, which this project does not have.

The test set is data (rag_safety_tests.json), so it can be re-run against a changed
engine and diffed. Tests are written to fail loudly: a test that cannot construct its
own adversarial input is an error, not a pass.

  PYTHONIOENCODING=utf-8 python scripts/run_rag_safety_tests.py
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_safety_rules import (
    DISEASE_WORDS, MEASUREMENT_WORDS, NORMALISING_WORDS, RULE_OUT_WORDS,
    confidence_gate, load_contract, render_advice,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "experiments" / "product_contract_20260924"

LOW, HIGH = "low_band_correspondence", "high_band_correspondence"


def rd(value, conf=0.99, source="model", abstain=False):
    return dict(value=value, confidence=conf, source=source, abstain=abstain)


def build_cases(gates: dict) -> list[dict]:
    """The eight properties, each with the inputs that would break it.

    Confidence values straddle the frozen gate deliberately: `just_below` is derived
    from the artefact, not written by hand, so a change to the frozen threshold moves
    the test with it instead of silently passing.
    """
    mg = gates["malformation_ratio"]["threshold"]
    just_below = round(mg - 0.01, 4)
    just_above = round(mg + 0.01, 4)
    return [
        dict(
            id="T1_unknown_is_not_normal",
            property="a field whose reading is unknown must not be rendered as normal",
            readings={"clarity": rd(None, None, "missing"),
                      "exudation": rd(None, None, "missing"),
                      "malformation_ratio": rd("unknown", None, "model")},
            expect=dict(no_normalising_words=True, uncertain=True,
                        recommend_human=True, used_fields=[]),
        ),
        dict(
            id="T2_low_confidence_abstains",
            property="a reading below the frozen confidence gate must be withheld",
            readings={"malformation_ratio": rd(HIGH, just_below)},
            expect=dict(used_fields=[], uncertain=True,
                        withheld_contains="below_confidence_gate"),
            note="gate %.4f from the frozen abstention table; input %.4f" % (mg, just_below),
        ),
        dict(
            id="T3_single_positive_no_disease_claim",
            property="one positive field must not produce a disease assertion",
            readings={"malformation_ratio": rd(HIGH, just_above)},
            expect=dict(no_disease_words=True, no_measurement_words=True,
                        recommend_human=True, used_fields=["malformation_ratio"]),
        ),
        dict(
            id="T4_conflicting_fields_report_uncertainty",
            property="disagreeing advice-eligible fields must yield uncertainty",
            readings={"malformation_ratio": rd(HIGH, just_above),
                      "clarity": rd(LOW, 0.95), "exudation": rd(LOW, 0.93)},
            expect=dict(uncertain=True, no_disease_words=True),
        ),
        dict(
            id="T5_missing_fields_ask_dont_guess",
            property="missing fields must prompt history/referral, not a guess",
            readings={"clarity": rd(LOW, 0.95)},
            expect=dict(uncertain=False, advice_nonempty=True,
                        no_normalising_words=True),
            note="only one advice-eligible field present and it is negative; the "
                 "answer must still be a non-claim",
        ),
        dict(
            id="T6_no_positive_to_diagnosis_path",
            property="no input makes the engine name a disease",
            readings={"malformation_ratio": rd(HIGH, 0.999),
                      "clarity": rd(HIGH, 0.999), "exudation": rd(HIGH, 0.999)},
            expect=dict(no_disease_words=True, no_measurement_words=True,
                        recommend_human=True),
            note="all three tier-1 fields maximally positive -- the strongest input "
                 "the product can receive",
        ),
        dict(
            id="T7_no_negative_to_rule_out_path",
            property="no input makes the engine rule a disease out",
            readings={"malformation_ratio": rd(LOW, 0.999),
                      "clarity": rd(LOW, 0.999), "exudation": rd(LOW, 0.999)},
            expect=dict(no_rule_out_words=True, no_normalising_words=True,
                        no_disease_words=True),
            note="all three tier-1 fields maximally negative -- the input a "
                 "reassurance bug would need",
        ),
        dict(
            id="T8_fixed_mode_is_not_a_patient_fact",
            property="a field answered by the training mode must not become a finding",
            readings={"microthrombus": rd(LOW, 1.0, "fixed_mode"),
                      "rbc_aggregation": rd(LOW, 1.0, "fixed_mode"),
                      "flow_state": rd(LOW, 1.0, "fixed_mode"),
                      # The path that actually matters: an advice-eligible field
                      # falling back to the training mode. The three tier-3 fields
                      # above are already stopped by their tier, so on its own that
                      # would let a source bug hide behind a second barrier.
                      "exudation": rd(LOW, 1.0, "fixed_mode")},
            expect=dict(used_fields=[], no_normalising_words=True, uncertain=True,
                        withheld_contains="fixed_mode_is_not_a_patient_fact"),
            note="these are fields the product answers with the development mode; a "
                 "confidence of 1.0 must not buy them entry, and the tier-1 member "
                 "must be stopped by source alone",
        ),
    ]


def check(case: dict, adv, contract: dict) -> dict:
    """Evaluate one case. Every expectation is a separate named check."""
    text, exp, checks = adv.text(), case["expect"], {}

    def hit(words):
        return sorted(w for w in words if w in text)

    if exp.get("no_disease_words"):
        bad = hit(DISEASE_WORDS)
        checks["no_disease_words"] = dict(passed=not bad, found=bad)
    if exp.get("no_rule_out_words"):
        bad = hit(RULE_OUT_WORDS)
        checks["no_rule_out_words"] = dict(passed=not bad, found=bad)
    if exp.get("no_normalising_words"):
        bad = hit(NORMALISING_WORDS)
        checks["no_normalising_words"] = dict(passed=not bad, found=bad)
    if exp.get("no_measurement_words"):
        bad = hit(MEASUREMENT_WORDS)
        checks["no_measurement_words"] = dict(passed=not bad, found=bad)
    if "uncertain" in exp:
        checks["uncertain"] = dict(passed=adv.uncertain == exp["uncertain"],
                                   got=adv.uncertain, want=exp["uncertain"])
    if "recommend_human" in exp:
        checks["recommend_human"] = dict(
            passed=adv.recommend_human == exp["recommend_human"],
            got=adv.recommend_human, want=exp["recommend_human"])
    if "used_fields" in exp:
        got = sorted(adv.used_fields)
        checks["used_fields"] = dict(passed=got == sorted(exp["used_fields"]),
                                     got=got, want=sorted(exp["used_fields"]))
    if exp.get("advice_nonempty"):
        checks["advice_nonempty"] = dict(passed=bool(adv.sentences),
                                         n_sentences=len(adv.sentences))
    if "withheld_contains" in exp:
        want = exp["withheld_contains"]
        found = sorted(f for f, why in adv.withheld.items() if want in why)
        checks["withheld_contains"] = dict(passed=bool(found), reason=want,
                                           fields=found)
    # Two invariants hold for every case, whatever it asked for.
    allow = set(contract["advice_allowed_whitelist"])
    off = [s for s in adv.sentences if s not in allow]
    checks["only_whitelisted_sentences"] = dict(passed=not off, off_whitelist=off)
    disclaimer = [s for s in allow if "不是诊断" in s]
    checks["disclaimer_always_present"] = dict(
        passed=any(s in adv.sentences for s in disclaimer))

    return dict(id=case["id"], property=case["property"],
                note=case.get("note"), readings=case["readings"],
                advice=adv.as_dict(), checks=checks,
                passed=all(c["passed"] for c in checks.values()))


def mutations(contract: dict, gates: dict) -> list[dict]:
    """Negative controls: eight broken engines that each test must catch.

    A suite that passes on the real engine proves nothing on its own -- it could be
    checking nothing at all. Each mutant below is the defect the matching test claims
    to prevent, so the test must FAIL against it. A mutant nothing catches means the
    test is decorative.
    """
    from rag_safety_rules import Advice

    def normalising(readings, *_):
        a = Advice()
        a.sentences = ["本次图像未见异常"]  # renders a non-reading as normal
        return a

    def ignores_gate(readings, contract, gates=None):
        # accepts every reading regardless of confidence
        loose = {k: dict(v, threshold=0.0) for k, v in (gates or {}).items()}
        return render_advice(readings, contract, loose)

    def names_disease(readings, *_):
        a = Advice()
        a.sentences = ["图像提示你患有系统性硬化症"]
        a.recommend_human = True
        return a

    def resolves_conflict(readings, contract, gates=None):
        a = render_advice(readings, contract, gates)
        a.uncertain = False  # picks a winner instead of reporting disagreement
        return a

    def silent(readings, *_):
        return Advice()  # empty answer when fields are missing

    def reports_percentage(readings, contract, gates=None):
        a = render_advice(readings, contract, gates)
        a.sentences.insert(0, "畸形血管约 23%")
        return a

    def rules_out(readings, *_):
        a = Advice()
        a.sentences = ["图像正常，可以排除微循环疾病"]
        return a

    def trusts_fixed_mode(readings, contract, gates=None):
        stripped = {k: dict(v, source="model") for k, v in readings.items()}
        return render_advice(stripped, contract, gates)

    return [
        dict(id="T1_unknown_is_not_normal", fn=normalising,
             defect="renders a missing reading as 未见异常"),
        dict(id="T2_low_confidence_abstains", fn=ignores_gate,
             defect="drops the confidence gate to zero"),
        dict(id="T3_single_positive_no_disease_claim", fn=names_disease,
             defect="names a disease from one positive field"),
        dict(id="T4_conflicting_fields_report_uncertainty", fn=resolves_conflict,
             defect="suppresses the uncertainty flag on disagreement"),
        dict(id="T5_missing_fields_ask_dont_guess", fn=silent,
             defect="returns nothing instead of asking for more information"),
        dict(id="T6_no_positive_to_diagnosis_path", fn=reports_percentage,
             defect="emits a malformation percentage"),
        dict(id="T7_no_negative_to_rule_out_path", fn=rules_out,
             defect="rules a disease out from a negative image"),
        dict(id="T8_fixed_mode_is_not_a_patient_fact", fn=trusts_fixed_mode,
             defect="treats a fixed-mode answer as a model reading"),
    ]


def run_mutations(cases: list[dict], contract: dict, gates: dict) -> list[dict]:
    by_id = {c["id"]: c for c in cases}
    rows = []
    for m in mutations(contract, gates):
        case = by_id[m["id"]]
        adv = m["fn"](case["readings"], contract, gates)
        res = check(case, adv, contract)
        rows.append(dict(id=m["id"], defect=m["defect"],
                         caught=not res["passed"],
                         caught_by=sorted(k for k, c in res["checks"].items()
                                          if not c["passed"])))
    return rows


def main() -> None:
    contract = load_contract()
    gates = confidence_gate()
    cases = build_cases(gates)
    results = [check(c, render_advice(c["readings"], contract, gates), contract)
               for c in cases]
    muts = run_mutations(cases, contract, gates)

    n_pass = sum(r["passed"] for r in results)
    if len(results) != 8:
        raise RuntimeError("expected the eight enumerated properties, got %d"
                           % len(results))

    report = dict(
        run="rag_safety_tests_20260924",
        what_this_establishes=[
            "for each enumerated input shape, no disease claim is emitted",
            "an unknown or abstained reading is never rendered as normal",
            "a field fixed to the development mode never becomes a patient fact",
            "every rendered answer carries the not-a-diagnosis line, and no "
            "sentence outside the frozen whitelist can be emitted",
        ],
        what_this_does_not_establish=[
            "that the advice is clinically correct -- no independent clinician "
            "annotation exists, so clinical correctness is untested",
            "that the field readings are accurate; that is the accuracy work, "
            "separate from this",
            "coverage of unenumerated inputs: these are eight constructed paths, "
            "not a proof over all inputs",
        ],
        contract=str(CONTRACT_REL),
        confidence_gates=gates,
        operating_point_note=(
            "malformation_ratio's gate is the frozen abstention row at 70% target "
            "coverage: coverage 0.6975, accuracy 0.7876. That does NOT meet the "
            "project's 70% coverage + 85% accuracy bar, and abstaining further does "
            "not reach it (0.8148 at 50% coverage). The gate is the explicit "
            "lower-coverage refusal policy, not a claim that the bar was met."),
        summary=dict(n_cases=len(results), n_passed=n_pass,
                     n_failed=len(results) - n_pass,
                     all_passed=n_pass == len(results),
                     n_mutants=len(muts),
                     n_mutants_caught=sum(m["caught"] for m in muts)),
        mutation_control=dict(
            why=("a suite that passes on the real engine could be checking nothing. "
                 "Each mutant is the defect its test claims to prevent, so the test "
                 "must fail against it."),
            asymmetry=("catching all mutants shows the tests have teeth on these "
                       "eight paths; it does not show the engine is safe on inputs "
                       "nobody enumerated"),
            results=muts),
        cases=results,
        locked_cases_seen=0,
        locked_note=("this run reads no image and no case data at all: it exercises "
                     "the rule engine on constructed field readings"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rag_safety_tests.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("rag safety: %d/%d passed" % (n_pass, len(results)))
    for r in results:
        flag = "ok  " if r["passed"] else "FAIL"
        bad = [k for k, c in r["checks"].items() if not c["passed"]]
        print("  %s %-40s %s" % (flag, r["id"], ",".join(bad)))
    n_caught = sum(m["caught"] for m in muts)
    print("mutation control: %d/%d mutants caught" % (n_caught, len(muts)))
    for m in muts:
        print("  %s %-40s %s" % ("ok  " if m["caught"] else "MISS", m["id"],
                                 ",".join(m["caught_by"]) or m["defect"]))
    print("wrote %s" % (OUT / "rag_safety_tests.json"))
    if n_caught != len(muts):
        # An uncaught mutant means the matching test does not test anything.
        raise SystemExit("mutation control failed: %s"
                         % [m["id"] for m in muts if not m["caught"]])
    if n_pass != len(results):
        raise SystemExit(1)


CONTRACT_REL = ("artifacts/experiments/product_contract_20260924/"
                "field_output_contract.json")

if __name__ == "__main__":
    main()
