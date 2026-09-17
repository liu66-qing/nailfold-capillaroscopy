from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nailfold_report.data.report_layout import REPORT_FIELDS  # noqa: E402
from nailfold_report.labels.parsing import validate_ocr_payload  # noqa: E402


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def report_groups(data_root: Path) -> list[list[Path]]:
    by_hash: dict[str, list[Path]] = {}
    paths = sorted(data_root.glob("recovered_archive*/[0-9]*/rep*.jpg.jpg"))
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path)
    return list(by_hash.values())


def token_geometry(row: list[Any]) -> tuple[float, float, str, float]:
    box, text, confidence = row
    return (
        sum(point[0] for point in box) / 4,
        sum(point[1] for point in box) / 4,
        str(text).strip(),
        float(confidence),
    )


def score_value(
    tokens: list[tuple[float, float, str, float]],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[str | None, float | None]:
    candidates = [
        (text, confidence)
        for x, y, text, confidence in tokens
        if x0 <= x <= x1 and y0 <= y <= y1 and NUMBER.search(text)
    ]
    if not candidates:
        return None, None
    text, confidence = max(candidates, key=lambda item: item[1])
    match = NUMBER.search(text)
    return (match.group() if match else None), confidence


def keyword_number(
    tokens: list[tuple[float, float, str, float]], keyword: str
) -> tuple[str | None, float | None]:
    candidates = [
        (text, confidence)
        for _, _, text, confidence in tokens
        if keyword in text and NUMBER.search(text)
    ]
    if not candidates:
        return None, None
    text, confidence = max(candidates, key=lambda item: item[1])
    matches = NUMBER.findall(text)
    return (matches[-1] if matches else None), confidence


def transcribe(engine: RapidOCR, path: Path) -> tuple[dict[str, str | None], dict[str, Any]]:
    with Image.open(path) as source:
        table = source.convert("RGB").crop((0, 0, source.width, 465))
        table = table.resize((table.width * 2, table.height * 2), Image.Resampling.LANCZOS)
    rows, elapsed = engine(np.asarray(table))
    tokens = [token_geometry(row) for row in (rows or [])]
    fields: dict[str, Any] = {}
    confidences: dict[str, float | None] = {}

    for field in REPORT_FIELDS:
        target_y = field.center_y * 2
        candidates = [
            (abs(y - target_y), -confidence, text, confidence)
            for x, y, text, confidence in tokens
            if 430 <= x <= 650 and abs(y - target_y) <= 15
        ]
        if candidates:
            _, _, text, confidence = min(candidates)
            fields[field.name] = text
            confidences[field.name] = confidence
        else:
            fields[field.name] = None
            confidences[field.name] = None
        if field.name in {
            "capillary_count",
            "vasomotion",
            "wbc_count",
            "sweat_duct",
        } and fields[field.name]:
            fields[field.name] = re.sub(
                r"^[nNoO](?=-{1,2}\d)", "0", fields[field.name]
            )

    score_specs = {
        "morphology_score": (235, 350, 845, 878),
        "periloop_score": (1040, 1170, 845, 878),
        "total_score": (235, 350, 886, 918),
    }
    for field, (x0, x1, y0, y1) in score_specs.items():
        fields[field], confidences[field] = score_value(
            tokens, x0=x0, x1=x1, y0=y0, y1=y1
        )
    for field, keyword in {
        "morphology_score": "\u5f62\u6001\u79ef\u5206",
        "periloop_score": "\u7965\u5468\u79ef\u5206",
        "total_score": "\u603b\u79ef\u5206",
    }.items():
        keyword_value, keyword_confidence = keyword_number(tokens, keyword)
        if keyword_value is not None:
            fields[field] = keyword_value
            confidences[field] = keyword_confidence

    flow_tokens = [
        (text, confidence)
        for x, y, text, confidence in tokens
        if 450 <= x <= 780 and 845 <= y <= 878 and "\u79ef\u5206" in text
    ]
    if flow_tokens:
        text, confidence = max(flow_tokens, key=lambda item: item[1])
        matches = NUMBER.findall(text)
        fields["flow_score"] = matches[-1] if matches else None
        confidences["flow_score"] = confidence
    else:
        fields["flow_score"] = None
        confidences["flow_score"] = None

    assessment_tokens = [
        (text, confidence)
        for x, y, text, confidence in tokens
        if 450 <= x <= 950 and 886 <= y <= 918 and "\u7efc\u5408\u5224\u65ad" in text
    ]
    if assessment_tokens:
        text, confidence = max(assessment_tokens, key=lambda item: item[1])
        fields["overall_assessment"] = re.sub(
            r"^.*?\u7efc\u5408\u5224\u65ad\s*[:\uff1a]?", "", text
        ).strip() or None
        confidences["overall_assessment"] = confidence
    else:
        fields["overall_assessment"] = None
        confidences["overall_assessment"] = None

    return validate_ocr_payload(fields), {
        "confidence": confidences,
        "tokens": [
            {"x": round(x, 1), "y": round(y, 1), "text": text, "confidence": confidence}
            for x, y, text, confidence in tokens
        ],
        "elapsed": elapsed,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    groups = report_groups(args.data_root)
    if args.limit is not None:
        groups = groups[: args.limit]
    engine = RapidOCR()
    with args.output.open("a", encoding="utf-8") as stream:
        for index, group in enumerate(groups, start=1):
            try:
                fields, audit = transcribe(engine, group[0])
                result: dict[str, Any] = {
                    "status": "ok",
                    "engine": "rapidocr-onnxruntime-1.4.4",
                    "fields": fields,
                    "audit": audit,
                }
            except Exception as error:
                result = {
                    "status": "error",
                    "engine": "rapidocr-onnxruntime-1.4.4",
                    "error": f"{type(error).__name__}: {error}",
                }
            for path in group:
                relative = path.relative_to(args.data_root).as_posix()
                row = {
                    **result,
                    "exam_case_id": "/".join(relative.split("/")[:2]),
                    "report_path": relative,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(groups)}] x{len(group)} {result['status']}", flush=True)


if __name__ == "__main__":
    main()
