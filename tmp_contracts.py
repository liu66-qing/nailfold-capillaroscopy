from __future__ import annotations

import hashlib
from typing import Any

DYNAMIC_FIELDS = {
    "flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "microthrombus"
}
OPTIONAL_FIELD_ARTIFACTS = {"microthrombus": "microthrombus_binary.joblib"}

UNITS = {
    "afferent_diameter": "um",
    "efferent_diameter": "um",
    "apex_diameter": "um",
    "loop_length": "um",
    "crossing_ratio": "%",
    "malformation_ratio": "%",
}


def evidence_id(value: str, index: int) -> str:
    digest = hashlib.sha256(f"{index}:{value}".encode()).hexdigest()[:16]
    return f"evidence-{digest}"


def upgrade_report_v1(
    report: dict[str, Any],
    *,
    request_id: str,
    label_schema_version: str,
    preprocessing_version: str,
    device_profile_version: str,
    confidence_threshold: float = 0.70,
    timing: dict[str, int] | None = None,
) -> dict[str, Any]:
    evidence = []
    for index, path in enumerate(report.get("evidence_images", [])):
        item_id = evidence_id(str(path), index)
        evidence.append(
            {
                "evidence_id": item_id,
                "media_id": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16],
                "media_type": "image",
                "input_type": "original",
                "frame_index": None,
                "start_ms": None,
                "end_ms": None,
                "quality_status": "acceptable",
            }
        )
    static_evidence_ids = [item["evidence_id"] for item in evidence]
    fields: dict[str, Any] = {}
    review_reasons = list(report.get("quality", {}).get("warnings", []))
    for name, source in report["fields"].items():
        value = source.get("value")
        confidence = source.get("confidence")
        raw_status = str(source.get("status", ""))
        if value is None:
            status = "unavailable"
            missing_reason = raw_status or "no model output"
        elif "derived" in raw_status:
            status = "derived"
            missing_reason = None
        elif any(token in raw_status for token in ("low", "uncalibrated", "support")):
            status = "indeterminate"
            missing_reason = raw_status
        else:
            status = "predicted"
            missing_reason = None
        needs_review = (
            status in {"indeterminate", "unavailable"}
            or (confidence is not None and float(confidence) < confidence_threshold)
            or name in DYNAMIC_FIELDS
        )
        review_status = "review_required" if needs_review else "auto_pass"
        if needs_review:
            review_reasons.append(f"field:{name}:{raw_status or status}")
        fields[name] = {
            "value": value,
            "unit": UNITS.get(name),
            "confidence": confidence,
            "status": status,
            "missing_reason": missing_reason,
            "evidence_ids": [] if name in DYNAMIC_FIELDS else static_evidence_ids,
            "model_version": str(report.get("model_version", "unknown")),
            "preprocessing_version": preprocessing_version,
            "multi_view_consistency": None,
            "review_status": review_status,
        }
    quality_source = report.get("quality", {})
    quality_status = "limited" if review_reasons else "acceptable"
    review_status = "professional_review_required" if review_reasons else "auto_pass"
    timing_value = timing or {
        "queued_ms": 0,
        "preprocessing_ms": 0,
        "inference_ms": 0,
        "total_ms": 0,
    }
    return {
        "schema_version": "nailfold-analysis-result/2.0",
        "request_id": request_id,
        "case_id": str(report.get("case_id", request_id)),
        "task_status": "completed_with_review" if review_reasons else "completed",
        "versions": {
            "analysis_model": str(report.get("model_version", "unknown")),
            "label_schema": label_schema_version,
            "preprocessing": preprocessing_version,
            "device_profile": device_profile_version,
            "report_schema": str(report.get("schema_version", "nailfold-report/1.0")),
        },
        "fields": fields,
        "evidence": evidence,
        "quality": {
            "status": quality_status,
            "usable_views": int(quality_source.get("usable_images", 0)),
            "warnings": list(quality_source.get("warnings", [])),
            "reacquire_recommended": quality_status != "acceptable",
        },
        "review": {"status": review_status, "reasons": sorted(set(review_reasons))},
        "errors": [],
        "timing": timing_value,
    }

