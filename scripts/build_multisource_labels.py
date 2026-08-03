from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.labels.parsing import canonical_field_value  # noqa: E402
from nailfold_report.labels.schema import ALL_REPORT_FIELDS  # noqa: E402


NUMERIC_RANGES = {
    # Historical reports contain legitimate integer entries such as 67/89/1.3.
    # Keep permissive physical bounds and let redundant relationships identify OCR errors.
    "afferent_diameter": (1.0, 120.0),
    "efferent_diameter": (1.0, 120.0),
    "output_input_ratio": (0.2, 5.0),
    "apex_diameter": (2.0, 120.0),
    "loop_length": (30.0, 1000.0),
    "flow_speed_um_s": (0.0, 5000.0),
    "morphology_score": (0.0, 30.0),
    "flow_score": (0.0, 30.0),
    "periloop_score": (0.0, 30.0),
    "total_score": (0.0, 90.0),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def decode_rtf(path: Path) -> str:
    raw = path.read_bytes().decode("latin1", errors="ignore")
    data = bytearray()
    index = 0
    while index < len(raw):
        if raw[index:index + 2] == "\\'" and index + 4 <= len(raw):
            try:
                data.append(int(raw[index + 2:index + 4], 16))
                index += 4
                continue
            except ValueError:
                pass
        if raw[index] not in "{}":
            data.extend(raw[index].encode("latin1", errors="ignore"))
        index += 1
    text = data.decode("gb18030", errors="ignore")
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    text = re.sub(r"\\[^a-zA-Z]", " ", text)
    return re.sub(r"\s+", "", text)


def rtf_evidence(data_root: Path, case_id: str) -> dict[str, set[str]]:
    case_dir = data_root.joinpath(*case_id.split("/"))
    text = "".join(
        decode_rtf(path) for path in (
            case_dir / "rep_rch1.rtf", case_dir / "prn_rch1.rtf"
        ) if path.exists()
    )
    evidence: dict[str, set[str]] = defaultdict(set)
    rules = {
        "clarity": (("清晰度清晰", "清晰"), ("清晰度不清", "不清")),
        "capillary_count": (("数量基本正常", ">=7"),),
        "exudation": (("未见渗出", "无"), ("无渗出", "无")),
        "hemorrhage": (("未见出血", "无"), ("无出血", "无")),
        "microthrombus": (("未见白微栓", "无"), ("无白微栓", "无")),
        "subpapillary_venous_plexus": (("静脉丛不见", "不见"),),
    }
    for field, pairs in rules.items():
        for phrase, value in pairs:
            if phrase in text:
                evidence[field].add(value)
    return dict(evidence)


def engine_summary(
    rows: list[dict[str, Any]], field: str, engine: str
) -> dict[str, Any] | None:
    values = [
        canonical_field_value(field, row.get("fields", {}).get(field))
        for row in rows if row.get("status") == "ok"
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    counts = Counter(values)
    selected, support = counts.most_common(1)[0]
    agreement = support / len(values)
    if engine == "rapidocr":
        confidences = [
            row.get("audit", {}).get("confidence", {}).get(field)
            for row in rows
            if canonical_field_value(field, row.get("fields", {}).get(field)) == selected
        ]
        confidences = [float(value) for value in confidences if value is not None]
        ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.6
    else:
        ocr_confidence = 0.88
    reliability = agreement * ocr_confidence
    return {
        "engine": engine, "selected": selected, "copies": len(values),
        "support": support, "agreement": agreement,
        "ocr_confidence": ocr_confidence, "reliability": reliability,
        "counts": dict(counts),
    }


def engine_explicitly_missing(rows: list[dict[str, Any]], field: str) -> bool:
    usable = [row for row in rows if row.get("status") == "ok"]
    return bool(usable) and all(
        canonical_field_value(field, row.get("fields", {}).get(field)) is None
        for row in usable
    )


def choose_label(summaries: list[dict[str, Any]]) -> tuple[str | None, float, str]:
    if not summaries:
        return None, 0.0, "missing"
    scores: dict[str, float] = defaultdict(float)
    for summary in summaries:
        scores[summary["selected"]] += summary["reliability"]
    selected = max(scores, key=scores.get)
    total = sum(scores.values())
    share = scores[selected] / total if total else 0.0
    engines_agree = len({summary["selected"] for summary in summaries}) == 1
    mean_quality = sum(summary["reliability"] for summary in summaries) / len(summaries)
    if engines_agree and len(summaries) >= 2:
        confidence = 0.72 + 0.18 * mean_quality + 0.10 * share
        status = "high_consensus"
    elif len(summaries) >= 2:
        confidence = 0.30 + 0.40 * share + 0.15 * mean_quality
        status = "engine_conflict"
    else:
        confidence = 0.45 + 0.40 * mean_quality
        status = "single_engine"
    return selected, min(confidence, 1.0), status


def apply_constraints(
    values: dict[str, Any], confidence: dict[str, float],
    status: dict[str, list[str]], rtf: dict[str, set[str]],
) -> None:
    for field, bounds in NUMERIC_RANGES.items():
        if values.get(field) is None:
            continue
        value = number(values[field])
        if value is None:
            values[field] = None
            confidence[field] = 0.0
            status[field].append("invalid_non_numeric_removed")
        elif not bounds[0] <= value <= bounds[1]:
            confidence[field] = min(confidence[field], 0.01)
            status[field].append("invalid_numeric_range")

    afferent = number(values.get("afferent_diameter"))
    efferent = number(values.get("efferent_diameter"))
    ratio = number(values.get("output_input_ratio"))
    if afferent and efferent and ratio is not None:
        derived = efferent / afferent
        difference = abs(derived - ratio)
        if difference <= max(0.12, 0.12 * derived):
            confidence["output_input_ratio"] = min(
                1.0, confidence["output_input_ratio"] + 0.08
            )
            status["output_input_ratio"].append("ratio_consistent")
        else:
            confidence["output_input_ratio"] *= 0.55
            confidence["afferent_diameter"] *= 0.85
            confidence["efferent_diameter"] *= 0.85
            status["output_input_ratio"].append(
                f"ratio_mismatch_derived={derived:.3f}"
            )

    score_fields = ("morphology_score", "flow_score", "periloop_score")
    parts = [number(values.get(field)) for field in score_fields]
    total = number(values.get("total_score"))
    if total is not None and all(value is not None for value in parts):
        difference = abs(sum(value for value in parts if value is not None) - total)
        if difference <= 0.11:
            for field in (*score_fields, "total_score"):
                confidence[field] = min(1.0, confidence[field] + 0.06)
                status[field].append("score_additivity_consistent")
        else:
            weakest = min((*score_fields, "total_score"), key=confidence.get)
            confidence[weakest] *= 0.45
            status[weakest].append(f"score_additivity_mismatch={difference:.2f}")

    for field, candidates in rtf.items():
        value = values.get(field)
        if value in candidates:
            confidence[field] = min(1.0, confidence[field] + 0.04)
            status[field].append("rtf_semantic_support")
        elif value is not None and candidates:
            confidence[field] *= 0.85
            status[field].append("rtf_semantic_conflict")


def reconcile_diameter_triplet(
    values: dict[str, Any], confidence: dict[str, float],
    status: dict[str, list[str]], candidates: dict[str, list[str]],
) -> None:
    fields = ("afferent_diameter", "efferent_diameter", "output_input_ratio")
    options: dict[str, list[float]] = {}
    for field in fields:
        if confidence[field] < 0.4:
            options[field] = []
            continue
        bounds = NUMERIC_RANGES[field]
        parsed = {
            value for value in (number(item) for item in candidates.get(field, []))
            if value is not None and bounds[0] <= value <= bounds[1]
        }
        current = number(values.get(field))
        if current is not None and bounds[0] <= current <= bounds[1]:
            parsed.add(current)
        options[field] = sorted(parsed)

    combinations = []
    for afferent in options["afferent_diameter"]:
        for efferent in options["efferent_diameter"]:
            for ratio in options["output_input_ratio"]:
                relative_error = abs(efferent / afferent - ratio) / max(ratio, 0.2)
                combinations.append((relative_error, afferent, efferent, ratio))
    if combinations:
        error, afferent, efferent, ratio = min(combinations)
        current_afferent = number(values.get("afferent_diameter"))
        current_efferent = number(values.get("efferent_diameter"))
        current_ratio = number(values.get("output_input_ratio"))
        current_error = (
            abs(current_efferent / current_afferent - current_ratio)
            / max(current_ratio, 0.2)
            if current_afferent and current_efferent and current_ratio else math.inf
        )
        if error <= 0.12 and error + 0.08 < current_error:
            replacements = dict(zip(fields, (afferent, efferent, ratio)))
            for field, replacement in replacements.items():
                if number(values.get(field)) != replacement:
                    old = values.get(field)
                    values[field] = f"{replacement:g}"
                    confidence[field] = max(0.58, confidence[field] * 0.9)
                    status[field].append(
                        f"redundancy_corrected_from={old}"
                    )

    afferent = number(values.get("afferent_diameter"))
    efferent = number(values.get("efferent_diameter"))
    ratio = number(values.get("output_input_ratio"))
    ratio_valid = ratio is not None and NUMERIC_RANGES["output_input_ratio"][0] <= ratio <= \
        NUMERIC_RANGES["output_input_ratio"][1]
    if afferent and efferent and not ratio_valid:
        derived = round(efferent / afferent, 1)
        old = values.get("output_input_ratio")
        values["output_input_ratio"] = f"{derived:.1f}"
        confidence["output_input_ratio"] = min(
            confidence["afferent_diameter"], confidence["efferent_diameter"]
        ) * 0.55
        status["output_input_ratio"].append(
            f"derived_from_diameters_replacing={old}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapidocr", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for engine, path in (("rapidocr", args.rapidocr), ("qwen", args.qwen)):
        for row in read_jsonl(path):
            if row.get("status") == "ok":
                grouped[str(row["exam_case_id"])][engine].append(row)

    output_rows = []
    audit = {}
    for case_id, engines in sorted(grouped.items()):
        values: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        statuses: dict[str, list[str]] = {}
        case_audit: dict[str, Any] = {}
        for field in ALL_REPORT_FIELDS:
            summaries = [
                summary for summary in (
                    engine_summary(engines.get(engine, []), field, engine)
                    for engine in ("rapidocr", "qwen")
                ) if summary is not None
            ]
            value, confidence, state = choose_label(summaries)
            if (
                value is not None
                and engine_explicitly_missing(
                    engines.get("rapidocr", []), field
                )
            ):
                confidence *= 0.25
                state = "rapid_layout_missing_qwen_only"
            values[field] = value
            confidences[field] = confidence
            statuses[field] = [state]
            case_audit[field] = {"sources": summaries}
        candidates = {
            field: [
                summary["selected"]
                for summary in case_audit[field]["sources"]
            ]
            for field in ALL_REPORT_FIELDS
        }
        reconcile_diameter_triplet(values, confidences, statuses, candidates)
        semantic = rtf_evidence(args.data_root, case_id)
        apply_constraints(values, confidences, statuses, semantic)
        row: dict[str, Any] = {"exam_case_id": case_id}
        for field in ALL_REPORT_FIELDS:
            row[field] = values[field]
            row[f"{field}__confidence"] = round(confidences[field], 6)
            row[f"{field}__status"] = "|".join(statuses[field])
            case_audit[field].update({
                "selected": values[field],
                "confidence": confidences[field],
                "status": statuses[field],
            })
        row["mean_label_confidence"] = round(
            sum(confidences.values()) / len(confidences), 6
        )
        row["low_confidence_field_count"] = sum(
            value < 0.55 for value in confidences.values()
        )
        output_rows.append(row)
        audit[case_id] = {
            "fields": case_audit,
            "rtf_evidence": {
                field: sorted(values) for field, values in semantic.items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "cases": len(output_rows),
        "mean_case_confidence": sum(
            row["mean_label_confidence"] for row in output_rows
        ) / len(output_rows),
        "fields_below_0.55": sum(
            row["low_confidence_field_count"] for row in output_rows
        ),
        "per_field": {
            field: {
                "non_missing": sum(row[field] not in (None, "") for row in output_rows),
                "mean_confidence_non_missing": (
                    sum(
                        row[f"{field}__confidence"] for row in output_rows
                        if row[field] not in (None, "")
                    )
                    / max(sum(
                        row[field] not in (None, "") for row in output_rows
                    ), 1)
                ),
                "below_0.55": sum(
                    row[field] not in (None, "")
                    and row[f"{field}__confidence"] < 0.55
                    for row in output_rows
                ),
                "engine_conflicts": sum(
                    "engine_conflict" in row[f"{field}__status"]
                    for row in output_rows
                ),
                "redundancy_corrections": sum(
                    "redundancy_corrected" in row[f"{field}__status"]
                    for row in output_rows
                ),
            }
            for field in ALL_REPORT_FIELDS
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
