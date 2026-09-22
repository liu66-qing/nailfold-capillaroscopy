"""Build the field matrix BEFORE any further training, as the user required.

For each of the 15 non-fixed fields plus the 6 excluded ones, this reports from
the label file itself:

  - does the column exist, and how many development cases have a usable value
  - what the value vocabulary actually is, and how dirty it is
  - the single-constant-answer baseline (the mode share) and the class counts
  - the smallest class size, which decides whether a CI can be estimated at all
  - whether the task is observable from a single static image, from the field's
    own unit (a time base, a full field of view, or a device calibration)
  - whether it can be evaluated in the encoder comparison, and if not, why

Nothing is trained here and no metric about any encoder is produced. Anything
that cannot be determined from the file is written as unknown, not guessed.

DEVELOPMENT ONLY. locked-47 is never loaded.

  PYTHONIOENCODING=utf-8 python scripts/build_field_matrix.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"

# The user's list, verbatim and in their order.
IN_SCOPE = ["clarity", "exudation", "blood_color", "subpapillary_venous_plexus",
            "papilla", "rbc_aggregation", "malformation_ratio",
            "capillary_count", "afferent_diameter", "efferent_diameter",
            "apex_diameter", "loop_length", "crossing_ratio", "flow_state",
            "microthrombus"]
EXCLUDED = {
    "vasomotion": "fixed not_modelled; needs a time base",
    "wbc_count": "fixed not_modelled",
    "sweat_duct": "fixed not_modelled; needs the full field of view",
    "hemorrhage": "fixed not_modelled; needs the full field of view",
    "flow_speed_um_s": "empty column",
    "output_input_ratio": "derived quantity, not modelled on its own",
}

# Unit of observation, which decides what a static single crop can support at
# all. This is the classification already established in the project, restated
# here so the matrix is self-contained; it is an assertion about the FIELD's
# definition, not about any model's accuracy.
UNIT = {
    "clarity": ("static image appearance", True),
    "exudation": ("static image appearance", True),
    "blood_color": ("static image appearance", True),
    "subpapillary_venous_plexus": ("static image appearance", True),
    "papilla": ("static image shape", True),
    "rbc_aggregation": ("static image appearance, but graded from flow context",
                        True),
    "malformation_ratio": ("static image shape, ratio over vessels in view", True),
    "capillary_count": ("count per unit length; needs device calibration and a "
                        "defined denominator", False),
    "afferent_diameter": ("micrometres; needs device calibration", False),
    "efferent_diameter": ("micrometres; needs device calibration", False),
    "apex_diameter": ("micrometres; needs device calibration", False),
    "loop_length": ("micrometres; needs device calibration", False),
    "crossing_ratio": ("static image shape, ratio over vessels in view", True),
    "flow_state": ("time base; graded from motion", False),
    "microthrombus": ("time base; an event observed over time", False),
}

DIRTY = re.compile(r"^\s*$|^\[|\]$|^nan$|^NaN$|^None$|^未见$|^-+$|^/+$"
                   r"|^未测$|^无法|^模板|^单位|^条/|^个/", re.I)
# Only these four are printed as numbers. capillary_count is printed as a BAND
# (">=7", "5--6", "3--4", "<1"), so it is categorical in the label file and no
# absolute count has to be predicted; the calibration ban still blocks turning
# that band into a 条/mm figure.
NUMERIC_FIELDS = {"afferent_diameter", "efferent_diameter", "apex_diameter",
                  "loop_length"}
BANDED_STRING_FIELDS = {"capillary_count", "malformation_ratio", "crossing_ratio",
                        "microthrombus"}

# The target each field is admissible for, declared here rather than inferred,
# so no field can be given a target after seeing a result. "none" means no
# target is admissible from static images at all.
ADMISSIBLE = {
    "clarity": ("binary 清晰 vs 不清/模糊", {0: ["清晰"], 1: ["不清", "模糊"]}),
    "exudation": ("binary 无 vs +/++/+++", {0: ["无"], 1: ["+", "++", "+++"]}),
    "blood_color": ("binary 浅红/淡红 vs 暗红/暗紫",
                    {0: ["浅红", "淡红"], 1: ["暗红", "暗紫"]}),
    "subpapillary_venous_plexus": ("binary 不见 vs 可见",
                                   {0: ["不见"],
                                    1: ["可见1排", "可见2排", ">2排,扩张"]}),
    "papilla": ("3-class 波纹状/浅波纹状/平坦",
                {0: ["波纹状"], 1: ["浅波纹状"], 2: ["平坦"]}),
    "rbc_aggregation": ("binary 无 vs 轻/中/重",
                        {0: ["无"], 1: ["轻度", "中度", "重度"]}),
    "malformation_ratio": ("binary <=10% vs >10%",
                           {0: ["<=10%"], 1: ["10--30%", "30--60%", ">60%"]}),
    "crossing_ratio": ("binary <=30% vs >30%",
                       {0: ["<=30%"], 1: ["30--60%", "60--80%", ">80%"]}),
    "capillary_count": ("banded 3-class as printed >=7 / 5--6 / <=4",
                        {0: [">=7"], 1: ["5--6"], 2: ["3--4", "1--2", "<1"]}),
    "microthrombus": ("binary 无 vs >=1",
                      {0: ["无"], 1: ["1--2", ">2"]}),
    # flow_state and microthrombus are both defined on a time base (粒/线流 is a
    # motion grade; microthrombus is printed as 个/min). A static frame cannot
    # observe either unit. They are still given a target and still evaluated,
    # because refusing to evaluate them would be the same as declaring them
    # closed, which is exactly what must not happen. Any signal they show must be
    # read as a correlated static appearance, not as an observation of the unit.
    "flow_state": ("binary 线流/线粒流 vs 粒线流/粒流/粒缓流/粒摆流/全停",
                   {0: ["线流", "线粒流"],
                    1: ["粒线流", "粒流", "粒缓流", "粒摆流", "全停"]}),
    "afferent_diameter": ("banded 3-class from the printed 正常值 range", None),
    "efferent_diameter": ("banded 3-class from the printed 正常值 range", None),
    "apex_diameter": ("banded 3-class from the printed 正常值 range", None),
    "loop_length": ("banded 3-class from the printed 正常值 range", None),
}
# printed 正常值 column in the report, both ends inclusive
NORMAL_RANGE = {"afferent_diameter": (9.0, 13.0), "efferent_diameter": (11.0, 17.0),
                "apex_diameter": (12.0, 18.0), "loop_length": (150.0, 250.0)}
MIN_CLASS_FOR_CI = 10        # below this a per-class recall CI is not meaningful


def describe(dev: pd.DataFrame, field: str) -> dict:
    if field not in dev.columns:
        return dict(field=field, column_present=False,
                    evaluable=False, reason="column absent from the label file")
    # the file is arrow-backed, so a genuinely empty column stays float dtype
    # and astype(str) leaves NaN floats behind; coerce explicitly
    raw = dev[field].map(lambda s: "" if pd.isna(s) else str(s).strip())
    dirty_mask = raw.map(lambda s: bool(DIRTY.match(s)))
    clean = raw[~dirty_mask]
    vc = clean.value_counts()
    out = dict(field=field, column_present=True,
               n_development=int(len(dev)),
               n_usable=int(len(clean)), n_dirty_or_blank=int(dirty_mask.sum()),
               distinct_values=int(vc.size),
               value_counts={str(k): int(v) for k, v in vc.head(12).items()})
    if len(clean) == 0:
        out.update(evaluable=False, reason="every development value is blank, "
                                           "bracketed or otherwise unusable")
        return out
    if field in NUMERIC_FIELDS:
        num = pd.to_numeric(clean, errors="coerce").dropna()
        out["n_numeric_parsed"] = int(len(num))
        out["distinct_numeric_values"] = int(num.nunique())
        if len(num):
            out["numeric_summary"] = dict(
                min=round(float(num.min()), 3), median=round(float(num.median()), 3),
                max=round(float(num.max()), 3),
                sorted_distinct_head=[round(float(v), 3)
                                      for v in sorted(num.unique())[:12]])
        if field in NORMAL_RANGE and len(num):
            lo, hi = NORMAL_RANGE[field]
            band = pd.Series(np.where(num < lo, "low",
                                      np.where(num > hi, "high", "normal")),
                             index=num.index)
            bc = band.value_counts()
            out["banded_by_printed_normal_range"] = {
                str(k): int(v) for k, v in bc.items()}
            out["band_mode_share"] = round(float(bc.max() / bc.sum()), 4)
            out["band_smallest_class"] = int(bc.min())
            out["suggested_target"] = ("3-class 低于正常/正常/高于正常, from the "
                                       "report's own printed range")
        else:
            out["suggested_target"] = ("relative ordinal only; no printed normal "
                                       "range, and absolute micron output is "
                                       "forbidden without device calibration")
    unit, static_ok = UNIT.get(field, ("unknown", None))
    out["unit_of_observation"] = unit
    out["observable_from_static_image"] = static_ok
    target_desc, mapping = ADMISSIBLE.get(field, ("unknown", None))
    out["admissible_target"] = target_desc

    # Class counts on the MAPPED target, not on the raw vocabulary. The raw
    # vocabulary contains one-off dirty strings that vanish under the mapping,
    # so counting them would fabricate a tiny minority class.
    counts = None
    if mapping is not None:
        m = {v: k for k, vs in mapping.items() for v in vs}
        y = clean.map(m).dropna()
        counts = y.value_counts()
        out["n_after_mapping"] = int(len(y))
        out["dropped_by_mapping"] = int(len(clean) - len(y))
    elif field in NORMAL_RANGE and out.get("n_numeric_parsed"):
        num = pd.to_numeric(clean, errors="coerce").dropna()
        lo, hi = NORMAL_RANGE[field]
        y = pd.Series(np.where(num < lo, 0, np.where(num > hi, 2, 1)),
                      index=num.index)
        counts = y.value_counts()
        out["n_after_mapping"] = int(len(y))
    if counts is not None and len(counts):
        out["target_class_counts"] = {str(int(k)): int(v)
                                      for k, v in counts.sort_index().items()}
        out["target_mode_share"] = round(float(counts.max() / counts.sum()), 4)
        out["target_smallest_class"] = int(counts.min())
        out["target_n_classes"] = int(len(counts))

    reasons = []
    if static_ok is False and field not in NUMERIC_FIELDS \
            and field != "capillary_count":
        reasons.append("the field's unit is a time base, which a single static "
                       "frame does not carry; it is still evaluated, but any "
                       "signal must be read as correlated static appearance, "
                       "not as an observation of the unit")
    if field in NUMERIC_FIELDS:
        reasons.append("absolute micron output stays forbidden (device "
                       "calibration is UNCALIBRATED); only the banded target "
                       "from the report's own printed range is admissible")
        d = out.get("distinct_numeric_values", 999)
        if d <= 30:
            reasons.append("the label has only %d distinct numeric values, so it "
                           "is ordinal in practice, not continuous" % d)
    if field == "capillary_count":
        reasons.append("the label is already a printed band, so no 条/mm figure "
                       "is predicted; a per-mm output would still need a defined "
                       "denominator and device calibration")
    if counts is not None and len(counts) and counts.min() < MIN_CLASS_FOR_CI:
        reasons.append("smallest target class is %d cases, under %d, so a "
                       "per-class recall CI is not estimable"
                       % (int(counts.min()), MIN_CLASS_FOR_CI))
    if out.get("n_usable", 0) < 30:
        reasons.append("fewer than 30 usable development cases")
    out["blocking_reasons"] = reasons
    out["evaluable_in_encoder_comparison"] = bool(
        target_desc not in (None, "none", "unknown")
        and out.get("n_after_mapping", 0) >= 30
        and out.get("target_n_classes", 0) >= 2)
    out["deliverable_status_not_decided_here"] = (
        "evaluable means the comparison can produce a number; whether the field "
        "can ship is a separate question answered by the metrics, not by this file")
    return out


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))

    already = ["malformation_ratio", "clarity", "exudation",
               "subpapillary_venous_plexus", "blood_color", "papilla",
               "rbc_aggregation"]
    rows = [describe(dev, f) for f in IN_SCOPE]
    for r in rows:
        r["tested_in_round_one"] = r["field"] in already

    excl = []
    for f, why in EXCLUDED.items():
        d = dict(field=f, excluded_by_rule=why,
                 column_present=f in dev.columns)
        if f in dev.columns:
            raw = dev[f].map(lambda s: "" if pd.isna(s) else str(s).strip())
            clean = raw[~raw.map(lambda s: bool(DIRTY.match(s)))]
            d["n_usable"] = int(len(clean))
            d["distinct_values"] = int(clean.nunique())
        excl.append(d)

    out = dict(
        purpose="field matrix required before any further encoder experiment; "
                "no model is trained and no encoder metric appears here",
        label_file=str(LABELS.relative_to(ROOT)).replace("\\", "/"),
        n_development=int(len(dev)), locked_read=False,
        min_class_for_ci=MIN_CLASS_FOR_CI,
        in_scope=rows, excluded_by_rule=excl,
        notes=[
            "tested_in_round_one=false means UNTESTED in this round, not "
            "'no signal' and not 'closed'",
            "observable_from_static_image is a statement about the field's "
            "definition and unit, not about any model's accuracy",
            "absolute micron output stays forbidden: device_calibration_status "
            "is UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION",
            "the 3-class banding uses the range printed in the report's own "
            "正常值 column, inclusive at both ends; it is not our invention",
        ])
    (EXP / "field_matrix.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = pd.DataFrame([{k: v for k, v in r.items()
                          if not isinstance(v, (dict, list))} for r in rows])
    flat.to_csv(EXP / "field_matrix.csv", index=False, encoding="utf-8-sig")

    for r in rows:
        print("%-28s n=%-4s classes=%-4s smallest=%-4s tested=%-5s eval=%-5s  %s"
              % (r["field"], r.get("n_after_mapping"), r.get("target_n_classes"),
                 r.get("target_smallest_class"), r["tested_in_round_one"],
                 r.get("evaluable_in_encoder_comparison"),
                 r.get("admissible_target")))
        for b in r.get("blocking_reasons", []):
            print("      - %s" % b)
    print("\nwrote %s" % (EXP / "field_matrix.json"))


if __name__ == "__main__":
    main()
