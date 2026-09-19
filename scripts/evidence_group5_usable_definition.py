#!/usr/bin/env python
"""EVIDENCE GROUP 5: what the "5 usable fields" actually are, and what "usable" means.

This script does not fit any model. It reads the delivered artifacts and states, for
each candidate field, which ruler it passes and which it fails -- because the answer
to "which 5 fields" depends on the ruler, and previous delivery documents quietly
mixed rulers.

For the measurement fields it answers the user's specific sub-questions from the
repository: is the target localisable, are there pixel/keypoint annotations, is
there a trustworthy micron scale, is the target continuous or pre-discretised, and
what is the inter-doctor agreement.
"""
import argparse
import collections
import json
import os

import pandas as pd

MEASUREMENT_FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter",
                      "loop_length", "flow_speed_um_s", "output_input_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--locked",
                    default="artifacts/experiments/locked_delivered_20260918/"
                            "locked_delivered.json")
    ap.add_argument("--group3",
                    default="artifacts/evidence/group3_20260919/full_metrics.json")
    ap.add_argument("--group4",
                    default="artifacts/evidence/group4_20260919/controls.json")
    ap.add_argument("--ann-dir",
                    default="artifacts/doctor_annotation_round1_v3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    lk = json.load(open(a.locked, encoding="utf-8"))["fields"]
    g3 = json.load(open(a.group3, encoding="utf-8"))["fields"]
    g4 = json.load(open(a.group4, encoding="utf-8"))["part2_controls"][
        "run_by_this_script"]

    # ---- which 5 fields, under which ruler ---------------------------------
    per_field = {}
    for field, e in lk.items():
        dev = e["dev_frozen"]
        L = e["locked"]
        LM = e["locked_vs_locked_majority"]
        cnn = g4[field].get("control2_imagenet_cnn_backbone", {})
        hum = max(g4[field]["control3_human_features_logreg"]["delta"],
                  g4[field]["control3_human_features_randomforest"]["delta"])
        per_field[field] = {
            "development_oof": {
                "accuracy": dev["dev_acc"], "delta": dev["dev_delta"],
                "baseline": dev["dev_baseline"], "n": dev["dev_n"],
                "passes_on_development": dev["dev_passes"],
            },
            "locked_ruler_A_frozen_dev_majority": {
                "delta": L["delta"], "ci95": L["ci95"], "passes": L["passes"]},
            "locked_ruler_B_locked_own_majority": {
                "delta": LM["delta"], "ci95": LM["ci95"], "passes": LM["passes"],
                "is_the_stricter_ruler": bool(
                    e["prevalence_shift"]["dev_majority_answer_is_wrong_on_locked"])},
            "locked_auroc": g3[field]["locked"]["auroc"],
            "locked_balanced_accuracy": g3[field]["locked"]["balanced_accuracy"],
            "locked_ppv": g3[field]["locked"]["ppv_precision_abnormal"],
            "cnn_backbone_delta_on_development": cnn.get("delta"),
            "best_human_feature_delta_on_development": hum,
            "beaten_by_a_simple_feature_model_on_locked": (
                g3[field]["simple_feature_baseline_on_locked"]["accuracy"]
                >= g3[field]["locked"]["accuracy"]),
            "verdict_under_the_strict_ruler": (
                "PASS" if LM["passes"] else "FAIL -- CI covers zero"),
        }

    which_five = {
        "the_five_fields_named_in_the_delivery_form": [
            "clarity", "subpapillary_venous_plexus", "exudation", "blood_color",
            "malformation_ratio"],
        "but_that_list_was_never_one_configuration": (
            "The delivery form's A1 table mixed two pipelines: clarity, exudation, "
            "blood_color and malformation_ratio came from a geometry+DINOv2 concat "
            "run, while subpapillary_venous_plexus came from the 5-pooling "
            "ensemble. SVP is not even a field in the concat run. Under ONE "
            "configuration the development count is FOUR, not five, because "
            "malformation_ratio's development CI already covered zero."),
        "count_by_ruler": {
            "development_OOF_single_configuration": 4,
            "locked_ruler_A_frozen_dev_majority": sum(
                1 for v in per_field.values()
                if v["locked_ruler_A_frozen_dev_majority"]["passes"]),
            "locked_ruler_B_stricter": sum(
                1 for v in per_field.values()
                if v["locked_ruler_B_locked_own_majority"]["passes"]),
        },
        "honest_answer": (
            "On the held-out cohort, under the stricter of the two baselines, "
            "ZERO of the five fields pass. The '5 usable fields' number came from "
            "development OOF on a mixed configuration and must not be reported as "
            "product capability."),
    }

    what_usable_meant = {
        "the_standard_actually_used_in_this_project": {
            "definition": "case-level OOF accuracy minus the accuracy of a single "
                          "constant answer (the frozen development majority "
                          "class), with a 2000-sample bootstrap 95% CI whose "
                          "lower bound must exceed 0",
            "it_is_NOT": [
                "not an F1 threshold",
                "not a doctor's acceptability judgement -- no clinician has "
                "reviewed these outputs",
                "not stability across hospitals -- all data is the same three "
                "archives from the same source, and no device metadata exists",
                "not 90% accuracy or any absolute number",
            ],
            "it_IS_mostly": "validation-set (development OOF) accuracy above a "
                            "constant-answer baseline. Which is exactly the weak "
                            "reading: it is a development-set number.",
        },
        "why_the_baseline_matters_more_than_the_accuracy": (
            "A field with 5% prevalence gets 95% accuracy from always answering "
            "'normal'. Accuracy alone is meaningless here, which is why every "
            "number in this project is reported as a delta against a constant "
            "answer. Group 3 adds balanced accuracy, macro-F1, AUROC/AUPRC and "
            "PPV/NPV so the user can apply their own ruler."),
        "two_baselines_and_why": (
            "The protocol baseline is the FROZEN development majority answer. "
            "When prevalence flips between cohorts that frozen answer can be the "
            "WRONG answer on the new cohort, which drops the baseline and inflates "
            "the delta with no model improvement. clarity is exactly this: ruler A "
            "gives +0.3913, ruler B gives +0.0870 with a CI covering zero. Rule: "
            "record both, report the stricter one, never use either to select a "
            "configuration."),
        "no_clinical_acceptability_standard_exists_here": (
            "UNVERIFIED and honestly UNDEFINED. No clinician has set a numeric "
            "threshold for this product, no regulator sets one, and this project "
            "has never asked a doctor whether any of these outputs is acceptable. "
            "Any 'usable' claim is currently an engineering statement only."),
    }

    # ---- measurement fields -------------------------------------------------
    ann_status = collections.Counter()
    ann_reviewer = collections.Counter()
    n_instances = 0
    n_with_scale = 0
    n_files = 0
    cdir = os.path.join(a.ann_dir, "corrections")
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            d = json.load(open(os.path.join(cdir, fn), encoding="utf-8"))
            n_files += 1
            ann_status[d.get("status")] += 1
            ann_reviewer[d.get("reviewer") or "<empty>"] += 1
            n_instances += len(d.get("instances") or [])
            if (d.get("scale") or {}).get("um_per_pixel") is not None:
                n_with_scale += 1

    meas = {}
    for field in MEASUREMENT_FIELDS:
        v = m[field].astype(str).str.strip()
        present = v[~v.isin(["nan", "", "None"])]
        vc = present.value_counts()
        meas[field] = {
            "n_labelled": int(len(present)),
            "n_missing": int(len(v) - len(present)),
            "n_distinct_values": int(len(vc)),
            "most_common_values": {str(k): int(x) for k, x in vc.head(8).items()},
            "continuous_or_discretised": (
                "PRE-DISCRETISED by the reporting doctor -- the report records a "
                "small number of round values, not a measured continuum. "
                f"{len(vc)} distinct values over {len(present)} exams, and the "
                "top values carry most of the mass."),
        }
    meas_answers = {
        "q_can_the_target_be_localised_in_the_image": {
            "answer": "The loops are visible and a segmenter does find instances, "
                      "but the segmenter systematically MISSES the abnormal "
                      "vessels. The correlation between segmenter-derived "
                      "geometry and the clinical label has the WRONG SIGN "
                      "(frac_low_circ vs malformation rho=-0.257; n_inst vs "
                      "capillary_count rho=-0.382). So localisation exists but is "
                      "biased against exactly the cases that matter.",
            "source": "nailfold-seg-misses-abnormal; "
                      "SEG_RETRAIN_MISSES_20260918",
        },
        "q_are_there_pixel_or_keypoint_annotations": {
            "answer": "NO. ZERO human annotations exist.",
            "annotation_template_files": n_files,
            "status_counts": dict(ann_status),
            "reviewer_counts": dict(ann_reviewer),
            "total_annotated_instances": n_instances,
            "files_with_a_um_per_pixel": n_with_scale,
            "what_this_means": "The annotation round was prepared -- 43 case "
                               "templates with polygon/centerline/apex/afferent-"
                               "base/efferent-base slots and a MedSAM draft "
                               "overlay -- and never filled in. Every file is "
                               "status=needs_annotation with an empty reviewer "
                               "and zero instances. There is no pixel-level, "
                               "keypoint-level or mask-level human ground truth "
                               "anywhere in this project.",
        },
        "q_is_there_a_reliable_micron_scale": {
            "answer": "NO. There is no calibration.",
            "detail": "calibration_factors.json holds per-fold-per-archive coef/"
                      "intercept pairs BACK-FITTED from the labels themselves, "
                      "with four mutually inconsistent values. That is label "
                      "regression, not calibration. The annotation schema itself "
                      "records scale.source='missing', um_per_pixel=null, "
                      "magnification=null on all 43 files, and the round manifest "
                      "states absolute_measurement_status='blocked until "
                      "scale.um_per_pixel is externally calibrated'.",
            "no_acquisition_metadata": "device, magnification and protocol are "
                                       "recorded nowhere (group 2); EXIF is empty; "
                                       "so the scale cannot be recovered after "
                                       "the fact either",
            "hard_rule": "no micron value may be output for any field, and "
                         "um_per_pixel must never be cited as calibration "
                         "evidence",
        },
        "q_continuous_regression_or_pre_discretised": {
            "answer": "PRE-DISCRETISED, and coarsely. The labels are a handful of "
                      "round numbers per field, so the label granularity is the "
                      "ceiling, not the model. Measured consequence: 4 of 5 "
                      "measurement targets do not beat a median-constant "
                      "baseline, and the spacing between adjacent label levels is "
                      "SMALLER than the model's MAE -- the target is unresolvable "
                      "at this granularity.",
            "per_field": meas,
            "source": "nailfold-measurement-reality; "
                      "nailfold-label-granularity-ceiling",
        },
        "q_inter_doctor_agreement": {
            "answer": "UNKNOWN AND UNCOMPUTABLE. Every field in this dataset has "
                      "exactly ONE reading, transcribed from one clinical report. "
                      "There is no second reader for any exam, so no kappa, no "
                      "ICC, no agreement statistic of any kind can be computed. "
                      "A decision was previously taken not to run a two-reader "
                      "re-annotation.",
            "consequence": "the irreducible label noise is unmeasured, so there "
                           "is no known ceiling to compare model performance "
                           "against -- a model at 0.74 accuracy cannot be said to "
                           "be near or far from the human ceiling",
        },
        "which_measurement_field_is_deliverable": (
            "NONE as a micron value. The only measurement-derived output that was "
            "ever defensible is a BINARY 'is it dilated' recast, and even that has "
            "not been validated on the held-out cohort under the strict ruler. "
            "output_input_ratio must not be modelled at all: it is efferent/"
            "afferent, an arithmetic identity of two other fields."),
    }

    out = {
        "which_five_fields": which_five,
        "what_usable_meant": what_usable_meant,
        "per_field_all_rulers": per_field,
        "measurement_fields": meas_answers,
        "limitations": [
            "every 'passes' flag here is a delta-vs-constant-answer test at "
            "n<=47 on locked, where the CI half-width is about +-0.14 to +-0.21; "
            "a FAIL means UNRESOLVED, not proven-zero",
            "locked-47 is the same sites and the same capture pipeline as "
            "development: internal held-out, NOT external validation. This "
            "project has zero external validation.",
            "the cohort's one-shot budget was already spent at least five times "
            "with model selection performed on it in three runs",
        ],
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out)
    print("count by ruler:", which_five["count_by_ruler"])
    print("annotation files", n_files, "instances", n_instances,
          "with scale", n_with_scale)
    for f, v in per_field.items():
        print(f"  {f:28s} A={v['locked_ruler_A_frozen_dev_majority']['delta']:+.4f}"
              f" B={v['locked_ruler_B_locked_own_majority']['delta']:+.4f}"
              f" {v['verdict_under_the_strict_ruler']}")


if __name__ == "__main__":
    main()
