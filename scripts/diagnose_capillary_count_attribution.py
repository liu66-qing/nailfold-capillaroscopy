"""Automatic, development-only attribution audit for capillary-count errors.

Inputs are existing remote development OOF/count artifacts. The audit keeps
pixel-domain proxies separate from medical ground truth: real connected
components and skeleton branch counts are unavailable in the supplied case
summary and are reported as such rather than reconstructed from aggregates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


LABELS = ["3--4", "5--6", ">=7"]
ORDINAL = {label: index for index, label in enumerate(LABELS)}


def quantile_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-oof", type=Path, required=True)
    parser.add_argument("--seg-features", type=Path, required=True)
    parser.add_argument("--video-audit", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roles = pd.read_csv(args.roles)
    development_ids = set(roles.loc[roles.evaluation_role.eq("development"), "exam_case_id"].astype(str))
    locked_ids = set(roles.loc[roles.evaluation_role.eq("locked_test"), "exam_case_id"].astype(str))
    count = pd.read_csv(args.count_oof)
    count["case_id"] = count["case_id"].astype(str)
    # The remote OOF artifact is expected to be development-only; enforce it.
    count = count[count.case_id.isin(development_ids)].copy()
    if set(count.case_id) & locked_ids:
        raise ValueError("locked case leaked into count OOF input")
    seg = pd.read_parquet(args.seg_features)
    seg["case_id"] = seg["case_id"].astype(str)
    seg = seg[seg.case_id.isin(development_ids)].copy()
    video = pd.read_csv(args.video_audit)
    video["exam_case_id"] = video["exam_case_id"].astype(str)
    video = video[video.exam_case_id.isin(development_ids)].copy()
    geometry = pd.read_csv(args.geometry)
    geometry["case_id"] = geometry["case_id"].astype(str)
    geometry = geometry[geometry.case_id.isin(development_ids)].copy()

    # frame_features() emits 35 values and case summaries concatenate mean,
    # median, std. These are explicitly pixel/instance proxies, not masks truth.
    frame_mean = seg[["case_id", "feature_18", "feature_19", "feature_22", "feature_23", "feature_34"]].rename(columns={
        "feature_18": "mask_area_fraction",
        "feature_19": "image_focus_laplacian",
        "feature_22": "mask_component_area_mean_proxy",
        "feature_23": "mask_component_area_std_proxy",
        "feature_34": "abnormal_instance_ratio",
    })
    frame_mean = frame_mean.drop_duplicates("case_id")
    quality = video.groupby("exam_case_id", as_index=False).agg(
        video_count=("video_path", "count"),
        video_stability_median=("stability", "median"),
        video_stability_std=("stability", "std"),
        anchor_instances_median=("anchor_instances", "median"),
        decoded_median=("decoded", "median"),
        used_median=("used", "median"),
    )
    quality["device"] = "unavailable_in_manifest"
    frame = count.merge(frame_mean, left_on="case_id", right_on="case_id", how="left")
    frame = frame.merge(quality, left_on="case_id", right_on="exam_case_id", how="left").drop(columns=["exam_case_id"], errors="ignore")
    frame = frame.merge(geometry, on="case_id", how="left", suffixes=("", "_geometry"))
    frame["target"] = frame.target.astype(str)
    frame["oof_prediction"] = frame.oof_prediction.astype(str)
    frame["target_order"] = frame.target.map(ORDINAL)
    frame["prediction_order"] = frame.oof_prediction.map(ORDINAL)
    frame["error_direction"] = np.select(
        [frame.prediction_order < frame.target_order, frame.prediction_order > frame.target_order],
        ["undercount", "overcount"], default="correct",
    )
    frame["ordinal_error"] = (frame.prediction_order - frame.target_order).abs()
    frame["mask_count_proxy"] = frame.normal_count_mean + frame.abnormal_count_mean
    frame["mask_area_ratio"] = frame.mask_component_area_mean_proxy / frame.mask_area_fraction.clip(lower=1e-6)
    frame["connected_component_proxy"] = frame.normal_count_mean + frame.abnormal_count_mean
    frame["skeleton_count"] = frame.count_skeleton_mean
    frame["skeleton_to_mask_ratio"] = frame.skeleton_count / frame.mask_count_proxy.clip(lower=1e-6)
    frame["skeleton_branch_count"] = np.nan
    frame["skeleton_branch_status"] = "unavailable_from_case_aggregate"
    threshold_columns = [c for c in frame.columns if c.startswith("count_t") and c.endswith("_mean")]
    threshold_columns = sorted(threshold_columns, key=lambda c: float(re.search(r"count_t([0-9p]+)_", c).group(1).replace("p", ".")))
    frame["threshold_count_low"] = frame[threshold_columns[0]]
    frame["threshold_count_high"] = frame[threshold_columns[-1]]
    frame["threshold_span"] = frame.threshold_count_low - frame.threshold_count_high
    frame["threshold_relative_span"] = frame.threshold_span / frame.count_t0p225_mean.clip(lower=1e-6)
    frame["frame_count_cv"] = frame.count_t0p225_std / frame.count_t0p225_mean.clip(lower=1e-6)
    frame["mask_count_ratio_to_target_midpoint"] = frame.mask_count_proxy / frame.target.map({"3--4": 3.5, "5--6": 5.5, ">=7": 7.5}).clip(lower=1e-6)

    # Conservative candidate flags. They are mechanism hypotheses, not labels.
    frame["candidate_missed_detection"] = (frame.error_direction == "undercount") & ((frame.mask_count_proxy < 12) | (frame.mask_area_fraction < 0.008) | (frame.video_stability_median < 0.08))
    frame["candidate_adhesion"] = (frame.error_direction == "undercount") & (frame.mask_area_fraction > 0.025) & (frame.skeleton_to_mask_ratio > 1.15)
    # All cases are sensitive to threshold by construction; flag only the
    # upper-tail cases where the span is materially larger than the dev set.
    frame["candidate_threshold_shift"] = frame.threshold_relative_span > 4.0
    frame["candidate_fragmentation_or_fp"] = (frame.error_direction == "overcount") & ((frame.mask_count_proxy > 30) | (frame.abnormal_instance_ratio > 0.25))
    frame["candidate_frame_instability"] = frame.frame_count_cv > 0.65
    flags = ["candidate_missed_detection", "candidate_adhesion", "candidate_threshold_shift", "candidate_fragmentation_or_fp", "candidate_frame_instability"]
    frame["candidate_flags"] = frame[flags].apply(lambda row: "|".join(name.removeprefix("candidate_") for name in flags if bool(row[name])) or "none", axis=1)

    output_columns = [
        "case_id", "archive", "device", "target", "oof_prediction", "error_direction", "ordinal_error",
        "mask_count_proxy", "mask_area_fraction", "mask_area_ratio", "connected_component_proxy",
        "skeleton_count", "skeleton_branch_count", "skeleton_branch_status", "skeleton_to_mask_ratio",
        "threshold_count_low", "count_t0p225_mean", "threshold_count_high", "threshold_span", "threshold_relative_span",
        "frame_count", "count_t0p225_std", "frame_count_cv", "video_stability_median", "video_stability_std",
        "anchor_instances_median", "decoded_median", "used_median", "image_focus_laplacian", "abnormal_instance_ratio",
        *flags, "candidate_flags",
    ]
    output = frame[[c for c in output_columns if c in frame.columns]].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "per_case_attribution.csv", index=False)

    error_labels = ["correct", "undercount", "overcount"]
    cm = confusion_matrix(frame.error_direction, frame.candidate_flags.map(lambda x: "flagged" if x != "none" else "unflagged"), labels=error_labels) if False else None
    flag_confusion = {}
    for flag in flags:
        flag_confusion[flag] = pd.crosstab(frame.error_direction, frame[flag]).reindex(index=error_labels, columns=[False, True], fill_value=0).astype(int).values.tolist()
    report = {
        "schema_version": "capillary-count-attribution-development/1.0",
        "evaluation_role": "development_oof",
        "locked_cases_seen": 0,
        "locked_cases_in_source_manifest": len(locked_ids),
        "development_cases_in_source_manifest": len(development_ids),
        "cases_in_count_oof_after_development_filter": len(frame),
        "oof_errors": int((frame.error_direction != "correct").sum()),
        "error_direction_counts": frame.error_direction.value_counts().to_dict(),
        "flag_counts": {flag: int(frame[flag].sum()) for flag in flags},
        "flag_by_error_direction_false_true": flag_confusion,
        "feature_distributions": {column: quantile_summary(frame[column]) for column in ["mask_count_proxy", "mask_area_fraction", "connected_component_proxy", "skeleton_count", "skeleton_to_mask_ratio", "threshold_span", "threshold_relative_span", "frame_count_cv", "video_stability_median", "anchor_instances_median"]},
        "archive_error_rates": {
            str(archive): {str(k): int(v) for k, v in group.error_direction.value_counts().items()}
            for archive, group in frame.groupby("archive")
        },
        "device": {"status": "unavailable_in_supplied_manifest", "used_value": "unavailable_in_manifest"},
        "quality_proxies": ["video stability median/std", "anchor instance median", "decoded/used frame counts", "Laplacian focus proxy"],
        "limitations": [
            "mask_count_proxy and connected_component_proxy are detector instance counts, not validated capillary masks",
            "mask area is a feature summary proxy; no per-instance area/connected-domain table was supplied",
            "skeleton branch count is unavailable from the aggregate artifacts and is emitted as null",
            "flags are candidate mechanisms requiring visual/manual review and do not prove causality",
            "OOF target/prediction is development-only; locked cases were excluded before merge",
        ],
        "next_step_judgement": {
            "evidence_supported": "Threshold sensitivity and frame instability are measurable; missed-detection/fragmentation candidates can be prioritized by OOF direction.",
            "cannot_determine": "Whether adhesion or missed detection dominates without instance-level masks, connected components, skeleton branches, device IDs, and manual adjudication.",
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(frame), "oof_errors": int((frame.error_direction != 'correct').sum()), "locked_cases_seen": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
