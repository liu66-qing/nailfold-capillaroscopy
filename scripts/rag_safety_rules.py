"""The advice-layer rule engine: deny-by-default routing from field readings to text.

This is an ENGINEERING safety component, not a clinical one. Without independent
clinician annotation there is no way to show that a piece of health advice is
clinically correct. What can be shown is narrower and still worth shipping: that no
input path turns an image correlation into a disease claim, that a missing reading is
never rendered as a clean one, and that a field fixed to its training mode never
becomes a statement about this patient.

Deny-by-default in practice: the engine cannot emit a sentence that is not in the
frozen contract's whitelist. There is no template string in this file containing a
disease name, so no prompt, no retrieved passage and no field value can produce one.
The eight tests in run_rag_safety_tests.py attack that claim from eight directions.

Frozen inputs, read and never written:
  artifacts/experiments/product_contract_20260924/field_output_contract.json
  artifacts/experiments/rescue_external_20260922/frozen_candidate_malformation_a2x/
    frozen_candidate.json   (the confidence gate, taken from the abstention table)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (ROOT / "artifacts" / "experiments" / "product_contract_20260924"
            / "field_output_contract.json")
FROZEN_A2X = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
              / "frozen_candidate_malformation_a2x" / "frozen_candidate.json")

# The coverage point the product operates at. 70% coverage reaches accuracy 0.7876,
# which does NOT meet the project's 70%/85% bar, and abstaining further does not
# reach it either (0.8148 at 50%). So the gate is not a way to claim the bar; it is
# the explicit lower-coverage refusal policy the bar's failure forces.
OPERATING_COVERAGE = 0.7

# Substrings that must never appear in rendered advice, whatever the input.
DISEASE_WORDS = ["硬化症", "硬皮病", "系统性硬化", "糖尿病", "风湿", "血栓",
                 "雷诺病", "确诊", "患有"]
RULE_OUT_WORDS = ["排除", "不必担心", "没有问题", "一切正常", "未见异常",
                  "可以放心", "无需就医", "阴性"]
NORMALISING_WORDS = ["正常", "未见异常", "无异常", "阴性", "健康"]
MEASUREMENT_WORDS = ["%", "条/mm", "μm", "um/s", "微米", "个/min"]

# Reading sources. fixed_mode is the dangerous one: those fields are answered with
# the training majority and carry no information about the person in front of us.
SOURCES = ("model", "fixed_mode", "missing")


def load_contract() -> dict:
    with open(CONTRACT, encoding="utf-8") as fh:
        return json.load(fh)


def confidence_gate() -> dict:
    """Per-field confidence thresholds, read from frozen artefacts only.

    malformation_ratio's gate is the frozen abstention table's row at the operating
    coverage -- not a number chosen here, because choosing one now would be
    threshold selection after the fact.
    """
    with open(FROZEN_A2X, encoding="utf-8") as fh:
        frozen = json.load(fh)
    rows = frozen["abstention"]
    row = min(rows, key=lambda r: abs(r["target_coverage"] - OPERATING_COVERAGE))
    return {
        "malformation_ratio": {
            "threshold": row["confidence_threshold"],
            "provenance": "frozen_candidate.json abstention target_coverage=%s"
                          % row["target_coverage"],
            "coverage_at_gate": row["coverage"],
            "accuracy_at_gate": row["accuracy"],
        },
        # clarity and exudation have no frozen per-field abstention table, so they
        # get the neutral majority gate rather than a tuned one.
        "clarity": {"threshold": 0.5, "provenance": "neutral majority gate"},
        "exudation": {"threshold": 0.5, "provenance": "neutral majority gate"},
    }


class Advice:
    """What the advice layer produced, and the machine-checkable reasons why."""

    def __init__(self):
        self.sentences: list[str] = []
        self.uncertain = False
        self.recommend_human = False
        self.not_assessed: list[str] = []
        self.used_fields: list[str] = []
        self.withheld: dict[str, str] = {}
        self.notes: list[str] = []

    def text(self) -> str:
        return "".join(self.sentences)

    def as_dict(self) -> dict:
        return dict(sentences=list(self.sentences), uncertain=self.uncertain,
                    recommend_human=self.recommend_human,
                    not_assessed=sorted(self.not_assessed),
                    used_fields=sorted(self.used_fields),
                    withheld=dict(self.withheld), notes=list(self.notes))


def _pick(allow: list[str], *keys: str) -> list[str]:
    return [s for s in allow if any(k in s for k in keys)]


def render_advice(readings: dict, contract: dict, gates: dict | None = None) -> Advice:
    """Route field readings to advice text.

    readings: {field: {"value": str|None, "confidence": float|None,
                       "abstain": bool, "source": "model"|"fixed_mode"|"missing"}}

    Every exit path appends the disclaimer, and the only sentences that can be
    appended are the contract's whitelist entries.
    """
    gates = gates if gates is not None else confidence_gate()
    fields = contract["fields"]
    allow = contract["advice_allowed_whitelist"]
    out = Advice()
    positives: list[str] = []
    unknowns: list[str] = []

    for name in sorted(readings):
        r = readings[name] or {}
        spec = fields.get(name)
        if spec is None:
            # An unrecognised field carries no contract, so it cannot be trusted.
            out.withheld[name] = "not_in_contract"
            out.notes.append("field %r is not in the frozen contract; dropped" % name)
            out.uncertain = True
            continue
        src = r.get("source", "model")
        if src == "fixed_mode":
            # A field answered with the training majority says nothing about this
            # person. It may not enter advice and may not be reported as a finding.
            out.withheld[name] = "fixed_mode_is_not_a_patient_fact"
            out.not_assessed.append(name)
            unknowns.append(name)
            continue
        if src == "missing" or r.get("value") in (None, "unknown") or r.get("abstain"):
            out.withheld[name] = "abstained_or_missing"
            out.not_assessed.append(name)
            unknowns.append(name)
            continue
        if spec["status"] in ("unknown", "rejected"):
            out.withheld[name] = "status_%s" % spec["status"]
            out.not_assessed.append(name)
            unknowns.append(name)
            continue
        conf = r.get("confidence")
        gate = gates.get(name, {}).get("threshold")
        if gate is not None and (conf is None or conf < gate):
            out.withheld[name] = "below_confidence_gate_%.4f" % gate
            out.not_assessed.append(name)
            unknowns.append(name)
            continue
        if not spec["may_enter_advice"]:
            # tier 2 and 3 readings may exist in the payload but never drive advice.
            out.withheld[name] = "tier_%d_not_advice_eligible" % spec["tier"]
            out.not_assessed.append(name)
            continue
        out.used_fields.append(name)
        if str(r.get("value", "")).startswith("high_band"):
            positives.append(name)

    negatives = sorted(set(out.used_fields) - set(positives))

    if positives:
        # A positive reading gets the full conservative set: context, the red-flag
        # referral, the recheck, and the disclaimer. No disease is named because no
        # sentence in the whitelist names one.
        out.recommend_human = True
        out.sentences = list(allow)
        if negatives:
            # Disagreement among advice-eligible fields is reported as such rather
            # than resolved by picking a side.
            out.uncertain = True
            out.notes.append("advice-eligible fields disagree: high=%s other=%s"
                             % (sorted(positives), negatives))
    if unknowns:
        out.uncertain = True
        out.recommend_human = True
        for s in _pick(allow, "结合症状", "复查"):
            if s not in out.sentences:
                out.sentences.append(s)
    if not out.used_fields and not out.sentences:
        # Nothing usable came back. The answer is an explicit non-answer, never a
        # reassurance built out of silence.
        out.uncertain = True
        out.recommend_human = True
        out.sentences = _pick(allow, "结合症状", "复查")
    for s in _pick(allow, "不是诊断"):
        if s not in out.sentences:
            out.sentences.append(s)
    return out
