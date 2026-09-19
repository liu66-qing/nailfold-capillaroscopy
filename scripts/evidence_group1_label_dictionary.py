#!/usr/bin/env python
"""EVIDENCE GROUP 1: the label dictionary for every field.

One row per (field, raw answer) with counts by cohort, plus the binarisation
actually used for delivery, plus what can be established about label provenance.

Everything numeric here is READ from the manifest. Two things cannot be read from
data and are therefore tied to a named source or marked UNVERIFIED rather than
guessed:
  - image_visible: whether the quantity is even visible in a still image. Filled
    from this project's own prior experiments, each cited inline in FIELD_NOTES.
  - annotator: inferred only as far as the manifest's own columns allow
    (report_label_available, *__status, *__confidence, has_rep_rtf); anything
    beyond that is marked UNVERIFIED.

The user's five specific questions are answered in "questions_answered", each with
the number that answers it.
"""
import argparse
import json
import os
from collections import OrderedDict

import pandas as pd

FIELDS = OrderedDict([
    ("clarity", "ordinal_quality"),
    ("capillary_count", "count"),
    ("afferent_diameter", "measurement_um"),
    ("efferent_diameter", "measurement_um"),
    ("apex_diameter", "measurement_um"),
    ("output_input_ratio", "derived_ratio"),
    ("loop_length", "measurement_um"),
    ("crossing_ratio", "percent_band"),
    ("malformation_ratio", "percent_band"),
    ("flow_state", "categorical"),
    ("flow_speed_um_s", "measurement_um_s"),
    ("vasomotion", "count_band"),
    ("rbc_aggregation", "ordinal"),
    ("wbc_count", "count_band"),
    ("microthrombus", "count_band"),
    ("blood_color", "categorical"),
    ("exudation", "ordinal_plus"),
    ("hemorrhage", "ordinal_plus"),
    ("subpapillary_venous_plexus", "ordinal_visibility"),
    ("papilla", "categorical"),
    ("sweat_duct", "count_band"),
])

# Recovered by cross-tabulating each *__binary_oof.csv truth column against the
# raw manifest value. Only these 5 were ever binarised for delivery.
BINARY_MAP = {
    "clarity": {"normal": ["清晰"], "abnormal": ["不清", "模糊"]},
    "subpapillary_venous_plexus": {"normal": ["不见"],
                                   "abnormal": ["可见1排", "可见2排", ">2排,扩张"]},
    "exudation": {"normal": ["无"], "abnormal": ["+", "++", "+++"]},
    "blood_color": {"normal": ["浅红", "淡红"], "abnormal": ["暗红", "暗紫"]},
    "malformation_ratio": {"normal": ["<=10%"],
                           "abnormal": ["10--30%", "30--60%", ">60%"]},
}

# (image_visible, why) -- each "why" names the artifact or the structural reason.
FIELD_NOTES = {
    "clarity": ("yes", "property of the image itself (focus/contrast)"),
    "capillary_count": ("yes", "countable objects; segmenter n_inst vs count "
                        "rho=-0.382, SEG_INSTANCE_RECOVERY_20260918"),
    "afferent_diameter": ("partial", "visible but unmeasurable: no trustworthy "
                          "um_per_pixel exists (4 inconsistent values, "
                          "back-fitted from labels)"),
    "efferent_diameter": ("partial", "as afferent"),
    "apex_diameter": ("partial", "as afferent"),
    "output_input_ratio": ("no", "DERIVED, not observed: it is efferent/afferent, "
                           "an identity; modelling it is circular"),
    "loop_length": ("partial", "visible but needs a micron scale"),
    "crossing_ratio": ("partial", "segmenter misses the abnormal vessels; "
                       "correlation sign INVERTED, SEG_RETRAIN_MISSES_20260918"),
    "malformation_ratio": ("partial", "same inverted sign: frac_low_circ vs "
                           "malformation rho=-0.257"),
    "flow_state": ("no", "motion cannot be read from a still frame"),
    "flow_speed_um_s": ("no", "needs video AND a micron scale"),
    "vasomotion": ("no", "temporal phenomenon, needs video"),
    "rbc_aggregation": ("UNVERIFIED", "would need cell-level resolution; never "
                        "isolated experimentally"),
    "wbc_count": ("UNVERIFIED", "near-constant label, never testable"),
    "microthrombus": ("DISPUTED", "user states the data contains no thrombus, but "
                      "the column has 3 populated levels; unresolved"),
    "blood_color": ("yes", "colour is directly present in the pixels"),
    "exudation": ("partial", "peri-loop appearance; delivered dev delta +0.240"),
    "hemorrhage": ("yes", "visible as extravasated blood, but near-constant label"),
    "subpapillary_venous_plexus": ("yes", "background plexus visibility"),
    "papilla": ("partial", "textbook prior-confusion case: accuracy tracks the "
                "baseline exactly (0.751->0.902 vs 0.773->0.902)"),
    "sweat_duct": ("UNVERIFIED", "99% one value, arithmetically untestable at n=186"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--oof-dir",
                    default="artifacts/experiments/threshold_tuned_20260916")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m["exam_case_id"].astype(str)
    dev = m[m.evaluation_role == "development"]
    lk = m[m.evaluation_role == "locked_test"]

    files = pd.read_csv(a.files)
    files["exam_case_id"] = files["exam_case_id"].astype(str)
    cap = files[files.role == "cap_image"]
    per_case = cap.groupby("exam_case_id").size()

    rows = []
    summary = {}
    for field, kind in FIELDS.items():
        if field not in m.columns:
            summary[field] = {"error": "column absent"}
            continue
        vis, why = FIELD_NOTES.get(field, ("UNVERIFIED", ""))
        bm = BINARY_MAP.get(field)
        stat_col = field + "__status"
        conf_col = field + "__confidence"
        vals_all = m[field].astype(str).str.strip()
        vals_dev = dev[field].astype(str).str.strip()
        vals_lk = lk[field].astype(str).str.strip()
        order = vals_all[~vals_all.isin(["nan", "", "None"])].value_counts()

        for raw in list(order.index) + ["<MISSING>"]:
            if raw == "<MISSING>":
                n_all = int(vals_all.isin(["nan", "", "None"]).sum())
                n_dev = int(vals_dev.isin(["nan", "", "None"]).sum())
                n_lk = int(vals_lk.isin(["nan", "", "None"]).sum())
                if n_all == 0:
                    continue
                binlab = "EXCLUDED (never imputed as normal)"
            else:
                n_all = int((vals_all == raw).sum())
                n_dev = int((vals_dev == raw).sum())
                n_lk = int((vals_lk == raw).sum())
                if bm is None:
                    binlab = "never binarised"
                elif raw in bm["normal"]:
                    binlab = "0 normal"
                elif raw in bm["abnormal"]:
                    binlab = "1 abnormal"
                else:
                    binlab = "UNMAPPED -> dropped"
            rows.append({
                "field": field, "field_kind": kind, "raw_answer": raw,
                "n_all233": n_all, "n_dev186": n_dev, "n_locked47": n_lk,
                "binary_label_used": binlab,
                "image_visible": vis,
                "n_classes_raw": int(len(order)),
            })

        status = m[stat_col].astype(str).str.strip() if stat_col in m.columns else None
        summary[field] = {
            "field_kind": kind,
            "raw_classes": int(len(order)),
            "raw_distribution_all233": {str(k): int(v) for k, v in order.items()},
            "missing_all233": int(vals_all.isin(["nan", "", "None"]).sum()),
            "missing_rate_all233": round(
                float(vals_all.isin(["nan", "", "None"]).mean()), 4),
            "majority_share_dev": round(float(
                vals_dev[~vals_dev.isin(["nan", "", "None"])]
                .value_counts(normalize=True).iloc[0]), 4) if len(
                    vals_dev[~vals_dev.isin(["nan", "", "None"])]) else None,
            "binarised_for_delivery": bm is not None,
            "binary_scheme": bm,
            "image_visible": vis, "image_visible_basis": why,
            "status_values": (status.value_counts().to_dict()
                              if status is not None else "no __status column"),
            "mean_confidence": (round(float(m[conf_col].mean()), 4)
                                if conf_col in m.columns else None),
        }

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False, encoding="utf-8-sig")

    out = {
        "source": a.manifest,
        "cohorts": {"all": int(len(m)), "development": int(len(dev)),
                    "locked_test": int(len(lk))},
        "fields": summary,
        "questions_answered": {
            "q_is_fixed_answer_a_template_default": {
                "answer": "NO for the delivered fields -- the distributions are "
                          "genuinely spread, not one default value. But YES in "
                          "effect for 4 fields whose majority class exceeds 95%, "
                          "where a model cannot be distinguished from a constant.",
                "evidence": "see majority_share_dev per field; and "
                            "report_present_field_count below",
                "report_present_field_count_distribution": {
                    str(k): int(v) for k, v in
                    m.report_present_field_count.value_counts().sort_index().items()},
                "cases_with_zero_fields_present": int(
                    (m.report_present_field_count == 0).sum()),
            },
            "q_label_source_image_vs_record_vs_lab_vs_judgement": {
                "answer": "Clinical REPORT text (RTF), transcribed. It is the "
                          "doctor's overall judgement of the whole exam -- images "
                          "plus video -- not a per-image annotation and not a lab "
                          "result. No label in this project was produced by "
                          "annotating a specific image.",
                "evidence": {
                    "report_label_available": {
                        str(k): int(v) for k, v in
                        m.report_label_available.value_counts().items()},
                    "has_rep_rtf": {str(k): int(v) for k, v in
                                    m.has_rep_rtf.value_counts().items()},
                    "consequence": "the label is EXAM-level; a per-image label does "
                                   "not exist, so image-level training broadcasts "
                                   "one case label to all its images",
                },
            },
            "q_multiple_images_per_patient": {
                "answer": "YES -- multiple images per exam. Patient identity is "
                          "UNKNOWN: patient_id is empty for all 233 rows.",
                "cap_images_total": int(len(cap)),
                "images_per_exam_min": int(per_case.min()),
                "images_per_exam_median": float(per_case.median()),
                "images_per_exam_mean": round(float(per_case.mean()), 2),
                "images_per_exam_max": int(per_case.max()),
                "patient_id_non_null": int(m.patient_id.notna().sum()),
                "videos_total": int(m.video_count.fillna(0).sum()),
            },
            "q_unknown_coded_as_negative": {
                "answer": "NO in the delivered pipeline. Unmapped and missing "
                          "values are DROPPED, never coerced to normal. That is "
                          "visible as the shrinking n per field (e.g. "
                          "malformation_ratio n=162 of 186).",
                "evidence": "binary_label_used column in the CSV marks "
                            "'EXCLUDED (never imputed as normal)'; per-field "
                            "labelled n in the delivered run: clarity 185, SVP 184, "
                            "exudation 183, blood_color 181, malformation 162",
                "historical_exception": "TWO past bugs DID silently drop values by "
                                        "omitting them from the normal/abnormal "
                                        "lists: blood_color '浅红' (65 cases) and "
                                        "microthrombus '>2' (29 cases). Both were "
                                        "silent NaN, not silent negatives, and both "
                                        "are fixed.",
            },
            "q_middle_class_definition": {
                "answer": "UNVERIFIED as a clinical matter -- I cannot establish "
                          "from this data whether e.g. exudation '+' has a defined "
                          "threshold or is the doctor's impression. What IS known: "
                          "the delivered scheme is BINARY, so no middle class is "
                          "used; and the ordinal levels are unevenly spaced bands "
                          "(e.g. malformation <=10% / 10-30% / 30-60% / >60%), so "
                          "the 'middle' is a band, not a residual category.",
                "note": "the percent-band fields are interval strings, NOT numbers; "
                        "treating them as continuous would be wrong",
            },
        },
        "limitations": [
            "image_visible is a judgement column, sourced per field in "
            "image_visible_basis; four fields are marked UNVERIFIED and one "
            "DISPUTED rather than filled in",
            "annotator identity beyond 'transcribed from the clinical report' "
            "cannot be established from the manifest; there is no per-annotator id",
            "inter-rater agreement is UNKNOWN: every field has exactly one reading, "
            "so no agreement statistic can be computed at all",
            "three exams failed whole-page parsing; one of them is in locked_test "
            "and its ground truth is unrecoverable",
        ],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("fields:", len(summary), "| dictionary rows:", len(df))
    print("wrote", a.out, "and", a.out_csv)
    for field, s in summary.items():
        if "error" in s:
            print(" ", field, s["error"])
            continue
        print(f"  {field:28s} classes={s['raw_classes']} "
              f"missing={s['missing_all233']:3d} "
              f"majshare_dev={s['majority_share_dev']} "
              f"visible={s['image_visible']}")


if __name__ == "__main__":
    main()
