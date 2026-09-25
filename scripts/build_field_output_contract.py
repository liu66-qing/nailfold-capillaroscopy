"""The frozen field output contract and RAG routing, written BEFORE locked-47 is read.

This exists so that the locked-47 evaluation cannot change anything: the schema,
the field tiers, the status vocabulary, the abstention rule and the advice
whitelist are all fixed here, and the locked run is only allowed to fill in
numbers afterwards.

Three things this encodes that the modelling work does not:

1. The product emits HEALTH OBSERVATIONS, not diagnoses. Every field carries
   not_a_diagnosis, an advice whitelist and an advice blacklist, so a downstream
   RAG prompt cannot be the only thing standing between an image correlation and
   a disease claim.

2. status is a closed vocabulary of four values. diagnosis, disease_fact and
   clinically_confirmed are not in it and no field may be assigned them -- the
   builder asserts this rather than trusting the caller.

3. unknown is NOT normal. Every field whose value is unknown must be rendered as
   "not assessed", because the failure mode that matters here is a missing
   reading being read as a clean result.

Field tiers come from measured results, not from ambition:
  tier 1 (may enter RAG as input): clarity, exudation, malformation_ratio
  tier 2 (research hint only): SVP, blood_color, papilla, capillary_count,
                               crossing_ratio
  tier 3 (unknown / not modelled): the dynamic, count and absolute-measurement
                                   fields, plus flow_speed_um_s which is deleted

Nothing here is a claim that any tier-1 field is clinically validated. None is.

  PYTHONIOENCODING=utf-8 python scripts/build_field_output_contract.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "experiments" / "product_contract_20260924"

# The only legal values of `status`. A field's status says what KIND of statement
# the field supports, never how confident one reading was.
STATUS_VOCAB = {
    "reliable_observation": (
        "a property of the image itself that the model reads dependably; still "
        "not a diagnosis"),
    "image_correlation": (
        "the image looks like cases the report placed in this band; no calibrated "
        "measurement and no disease claim"),
    "unknown": (
        "not assessed. MUST NOT be rendered as normal, negative or reassuring"),
    "rejected": (
        "a precondition failed (quality, confidence, missing input); the field is "
        "withheld from the advice layer entirely"),
}

FORBIDDEN_STATUS = ["diagnosis", "disease_fact", "clinically_confirmed",
                    "confirmed", "positive_diagnosis"]

# Advice that any tier-1 field is allowed to trigger. Deliberately conservative
# and deliberately identical across fields: the model does not know which disease
# to name, so it may only recommend that a human look.
ADVICE_ALLOWED = [
    "建议结合症状、既往史和既往检查综合判断",
    "如有雷诺现象、手指发白发紫、疼痛或反复皮肤改变，建议咨询医生",
    "建议在相同条件下复查，或由专业人员进一步评估",
    "本结果为图像相关性提示，不是诊断",
]

ADVICE_FORBIDDEN = [
    "你患有系统性硬化症",
    "你患有糖尿病",
    "你有血栓",
    "你没有疾病 / 可以排除疾病",
    "立即开始治疗或停用药物",
    "用图像排除任何疾病",
    "任何具体疾病的确诊或排除",
    "任何百分比、条/mm 或微米数值",
]

# tier, status, reference standard, and why. Every measured number cited here is
# development OOF under the single shipped configuration.
FIELDS = {
    # ---- tier 1: may enter the RAG as an input ----
    "clarity": dict(
        tier=1, status="reliable_observation",
        reference_standard="original report wording, single reviewer, no "
                           "independent second read",
        evidence="passes the three-way archive check; delta +0.087 on locked "
                 "after the prevalence-flip correction",
        role="image quality / interpretability gate, not a clinical finding",
        may_gate_other_fields=True),
    "exudation": dict(
        tier=1, status="image_correlation",
        reference_standard="original report wording, single reviewer",
        evidence="passes the three-way archive check",
        role="conservative image correlation",
        may_gate_other_fields=False),
    "malformation_ratio": dict(
        tier=1, status="image_correlation",
        reference_standard="original report band (<=10% vs >10%); the human "
                           "boxes are NOT the same measurement -- 17/44 inside "
                           "band, ~3x scale offset",
        evidence="A2x frozen candidate: development OOF delta +0.1605, BA 0.7216, "
                 "refit LOAO beats a refit A0 in all three archives "
                 "(+0.1444/+0.0840/+0.1850, 2 of 3 CIs exclude zero)",
        role="binary band correspondence only; percentages forbidden",
        may_gate_other_fields=False,
        binary_values=["low_band_correspondence", "high_band_correspondence"],
        product_bar_met=False,
        product_bar_note="0.7876 accuracy at 69.75% coverage against an 85% bar; "
                         "not reachable by abstaining further (0.8148 at 50%)"),

    # ---- tier 2: research hint only, never an advice input ----
    "subpapillary_venous_plexus": dict(
        tier=2, status="image_correlation",
        reference_standard="original report wording",
        evidence="archive-dependent; fails the leave-one-archive-out check",
        role="research hint", may_gate_other_fields=False),
    "blood_color": dict(
        tier=2, status="image_correlation",
        reference_standard="original report wording",
        evidence="archive-dependent; fails the leave-one-archive-out check",
        role="research hint", may_gate_other_fields=False),
    "papilla": dict(
        tier=2, status="image_correlation",
        reference_standard="original report wording, 3 bands kept",
        evidence="best arm BA 0.5115 on a 3-class field; pre-registered fusion "
                 "was worse than the best single arm. Model side closed.",
        role="research hint", may_gate_other_fields=False),
    "capillary_count": dict(
        tier=2, status="unknown",
        reference_standard="printed band; no device calibration, no denominator, "
                           "no field-of-view metadata",
        evidence="blocked by the calibration prohibition, not by model quality",
        role="blocked pending scale and denominator definition",
        may_gate_other_fields=False),
    "crossing_ratio": dict(
        tier=2, status="unknown",
        reference_standard="NOT ESTABLISHED -- the local crossing definition is "
                           "the bottleneck; cross_vessel detector AP caps at "
                           "0.3156 across all five folds",
        evidence="no arm clears threshold; the reference standard must be redone "
                 "by an experienced reader before modelling continues",
        role="blocked pending a redone reference standard",
        may_gate_other_fields=False),
}

# tier 3: present in the schema so that a consumer sees them as explicitly not
# assessed rather than simply missing. Reason is the unit that is unavailable.
TIER3 = {
    "microthrombus": "needs a time base; also clinically contested in this data",
    "flow_state": "needs a time base",
    "rbc_aggregation": "needs a time base",
    "vasomotion": "needs a time base",
    "wbc_count": "needs a time base and a counting protocol",
    "sweat_duct": "sample size makes this arithmetically unreachable at n=186",
    "hemorrhage": "no usable reference standard (external kappa -0.208)",
    "afferent_diameter": "absolute micron measurement; device is UNCALIBRATED",
    "efferent_diameter": "absolute micron measurement; device is UNCALIBRATED",
    "apex_diameter": "absolute micron measurement; device is UNCALIBRATED",
    "loop_length": "absolute micron measurement; device is UNCALIBRATED",
}

DELETED = {"flow_speed_um_s": "deleted: an absolute velocity the device cannot "
                              "calibrate and the report back-calculates"}

# The per-field JSON schema every consumer sees. abstain and status are separate
# on purpose: a field can be answered with low confidence (abstain=true) while its
# status stays image_correlation, and conflating them would let a withheld reading
# inherit the status line of an answered one.
FIELD_SCHEMA = {
    "type": "object",
    "required": ["field", "value", "confidence", "status", "abstain",
                 "not_a_diagnosis", "advice_allowed", "advice_forbidden",
                 "reference_standard"],
    "additionalProperties": False,
    "properties": {
        "field": {"type": "string"},
        "value": {"type": ["string", "null"],
                  "description": "null only when abstain is true"},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "status": {"type": "string", "enum": sorted(STATUS_VOCAB)},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": ["string", "null"]},
        "not_a_diagnosis": {"const": True,
                            "description": "always true; there is no code path "
                                           "that may set this false"},
        "advice_allowed": {"type": "array", "items": {"type": "string"}},
        "advice_forbidden": {"type": "array", "items": {"type": "string"}},
        "reference_standard": {"type": "string"},
        "tier": {"type": "integer", "enum": [1, 2, 3]},
        "may_enter_advice": {"type": "boolean"},
    },
}


def build_field(name: str, spec: dict) -> dict:
    tier = spec["tier"]
    st = spec["status"]
    if st not in STATUS_VOCAB:
        raise RuntimeError("field %s has status %r outside the vocabulary"
                           % (name, st))
    if st in FORBIDDEN_STATUS:
        raise RuntimeError("field %s claims a forbidden status %r" % (name, st))
    may_advise = bool(tier == 1 and st in ("reliable_observation",
                                           "image_correlation"))
    return dict(
        field=name, tier=tier, status=st,
        may_enter_advice=may_advise,
        not_a_diagnosis=True,
        advice_allowed=list(ADVICE_ALLOWED) if may_advise else [],
        advice_forbidden=list(ADVICE_FORBIDDEN),
        reference_standard=spec["reference_standard"],
        evidence=spec.get("evidence"),
        role=spec.get("role"),
        may_gate_other_fields=spec.get("may_gate_other_fields", False),
        **{k: v for k, v in spec.items()
           if k in ("binary_values", "product_bar_met", "product_bar_note")},
    )


def main() -> None:
    fields = {n: build_field(n, s) for n, s in FIELDS.items()}
    for n, why in TIER3.items():
        if n in fields:
            raise RuntimeError("%s is in both the modelled set and tier 3" % n)
        fields[n] = dict(
            field=n, tier=3, status="unknown", may_enter_advice=False,
            not_a_diagnosis=True, advice_allowed=[],
            advice_forbidden=list(ADVICE_FORBIDDEN),
            reference_standard="not established for product use",
            evidence=why, role="not modelled; reported as not assessed",
            may_gate_other_fields=False)

    # Assertions, not documentation. Each one is a rule that a later edit could
    # silently break, so it fails the build instead.
    advisable = sorted(n for n, f in fields.items() if f["may_enter_advice"])
    if advisable != ["clarity", "exudation", "malformation_ratio"]:
        raise RuntimeError("the advice-eligible set drifted: %s" % advisable)
    for n, f in fields.items():
        if f["not_a_diagnosis"] is not True:
            raise RuntimeError("%s may not disable not_a_diagnosis" % n)
        if f["status"] in FORBIDDEN_STATUS:
            raise RuntimeError("%s claims a forbidden status" % n)
        if f["advice_allowed"] and f["tier"] != 1:
            raise RuntimeError("%s is tier %d but carries advice" % (n, f["tier"]))
        # Every field, in every tier, must carry the blacklist -- a tier-3 field
        # rendered by a careless consumer is exactly where a disease claim would
        # slip in.
        if not f["advice_forbidden"]:
            raise RuntimeError("%s has no advice blacklist" % n)
    for bad in FORBIDDEN_STATUS:
        if bad in STATUS_VOCAB:
            raise RuntimeError("%r must not be a legal status" % bad)

    contract = dict(
        run="frozen field output contract and RAG routing",
        frozen_on=str(date.today()),
        frozen_before="any locked-47 read in this round",
        product_framing=(
            "this product emits health observations from an image. It does not "
            "diagnose, does not exclude disease, and has no independent disease "
            "endpoint in its training data"),
        status_vocabulary=STATUS_VOCAB,
        forbidden_status_values=FORBIDDEN_STATUS,
        unknown_is_not_normal=(
            "a field with status unknown or abstain=true must be rendered as "
            "'not assessed'. Rendering it as normal, negative, clear or "
            "reassuring is a defect, not a wording choice"),
        advice_allowed_whitelist=ADVICE_ALLOWED,
        advice_forbidden_blacklist=ADVICE_FORBIDDEN,
        field_schema=FIELD_SCHEMA,
        fields=fields,
        tiers=dict(
            tier1=dict(fields=advisable,
                       meaning="may enter the RAG as an input"),
            tier2=dict(fields=sorted(n for n, f in fields.items()
                                     if f["tier"] == 2),
                       meaning="research hint only; never reaches the advice layer"),
            tier3=dict(fields=sorted(n for n, f in fields.items()
                                     if f["tier"] == 3),
                       meaning="not assessed; present so a consumer cannot mistake "
                               "absence for normality"),
        ),
        deleted_fields=DELETED,
        what_this_does_not_establish=[
            "that any tier-1 field is clinically validated -- none is",
            "that the advice whitelist is clinically correct; without independent "
            "clinician annotation that cannot be tested, only engineering safety "
            "can",
            "that malformation_ratio meets the project's 70%/85% bar -- it does "
            "not, and abstaining further does not reach it",
        ],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "field_output_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    import pandas as pd
    pd.DataFrame([
        dict(field=f["field"], tier=f["tier"], status=f["status"],
             may_enter_advice=f["may_enter_advice"],
             may_gate_other_fields=f["may_gate_other_fields"],
             role=f.get("role"), reference_standard=f["reference_standard"])
        for f in sorted(fields.values(), key=lambda x: (x["tier"], x["field"]))
    ]).to_csv(OUT / "field_routing.csv", index=False, encoding="utf-8-sig")

    print("fields: %d (tier1=%d tier2=%d tier3=%d), deleted=%d"
          % (len(fields), len(contract["tiers"]["tier1"]["fields"]),
             len(contract["tiers"]["tier2"]["fields"]),
             len(contract["tiers"]["tier3"]["fields"]), len(DELETED)))
    print("advice-eligible:", advisable)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
