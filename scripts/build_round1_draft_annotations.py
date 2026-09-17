"""Build machine draft annotations for the existing 10-case review package.

Drafts are candidates only. They are never treated as doctor gold labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = {
    "recovered_archive1/1": "CAPorg3.jpg",
    "recovered_archive1/24": "CAPorg4.jpg",
    "recovered_archive1/28": "CAPorg5.jpg",
    "recovered_archive1/29": "CAPorg4.jpg",
    "recovered_archive1/30": "CAPorg5.jpg",
    "recovered_archive1/31": "CAPorg3.jpg",
    "recovered_archive1/32": "CAPorg5.jpg",
    "recovered_archive1/33": "CAPorg5.jpg",
    "recovered_archive1/34": "CAPorg4.jpg",
    "recovered_archive1/100": "CAPorg5.jpg",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum instance confidence included in the actionable draft (raw candidates are still counted).",
    )
    args = p.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    by_case = {}
    for item in predictions.values():
        case = str(item.get("case_id", ""))
        if case in CASES and case not in by_case:
            by_case[case] = item
    missing = sorted(set(CASES) - set(by_case))
    if missing:
        raise ValueError(f"missing draft cases: {missing}")
    output = {
        "schema_version": "machine-draft-annotation-round/1.1",
        "evaluation_role": "development",
        "locked_cases_included": 0,
        "draft_is_gold": False,
        "source": str(args.predictions),
        "min_confidence": args.min_confidence,
        "cases": [],
    }
    for case, item in sorted(by_case.items()):
        instances = []
        low_confidence_candidate_count = 0
        raw_candidate_count = 0
        for n, ins in enumerate(item.get("instances", []), 1):
            if str(ins.get("class_name", "")).lower() not in {"normal", "abnormal", "capillary"}:
                continue
            polygon = ins.get("polygon", [])
            if len(polygon) < 3:
                continue
            raw_candidate_count += 1
            confidence = float(ins.get("confidence", 0.0))
            if confidence < args.min_confidence:
                low_confidence_candidate_count += 1
                continue
            bbox = ins.get("bbox", [0, 0, 0, 0])
            width = float(bbox[2]) - float(bbox[0]) if len(bbox) >= 4 else 0.0
            height = float(bbox[3]) - float(bbox[1]) if len(bbox) >= 4 else 0.0
            instances.append({"draft_id": f"{case.replace('/', '_')}_{n:03d}", "polygon": polygon, "confidence": confidence, "visible": True, "measurable": bool(height >= 25 and width >= 3), "apex_point": None, "afferent_branch_point": None, "efferent_branch_point": None, "source_class": str(ins.get("class_name", ""))})
        hemo = [ins for ins in item.get("instances", []) if str(ins.get("class_name", "")).lower() == "hemo"]
        output["cases"].append({"exam_case_id": case, "image_name": CASES[case], "draft_instances": instances, "raw_candidate_count": raw_candidate_count, "low_confidence_candidate_count": low_confidence_candidate_count, "draft_hemorrhage_candidates": [{"bbox": x.get("bbox"), "confidence": float(x.get("confidence", 0.0))} for x in hemo], "draft_exudation_candidate": None, "review_status": "needs_correction", "notes": "Machine draft only. Correct/add/delete each loop; add apex and branch points; mark glare/shadow and non-measurable loops. Low-confidence candidates are omitted from the actionable draft but counted above as background for review."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(output["cases"]), "instances": sum(len(x["draft_instances"]) for x in output["cases"]), "raw_candidates": sum(x["raw_candidate_count"] for x in output["cases"]), "low_confidence_candidates": sum(x["low_confidence_candidate_count"] for x in output["cases"]), "hemorrhage_candidates": sum(len(x["draft_hemorrhage_candidates"]) for x in output["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
