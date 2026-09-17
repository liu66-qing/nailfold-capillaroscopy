from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

from .schema import ALL_REPORT_FIELDS


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model output")


def normalize_raw_value(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    replacements = {
        "\uff05": "%",
        "\u03bc": "u",
        "\u00b5": "u",
        "\u2013": "-",
        "\u2014": "-",
        "\uff0d": "-",
        "\u81f3": "-",
        "\uff1e": ">",
        "\uff1c": "<",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text)


def canonical_field_value(field: str, value: Any) -> str | None:
    """Normalize OCR-engine spelling differences without inventing missing labels."""
    text = normalize_raw_value(value)
    if text is None:
        return None
    # The historical report places the normal/reference value in square brackets
    # immediately to the right of the doctor's measurement.  A layout-shifted OCR
    # result must never turn that template text into a training target.
    if re.fullmatch(r"[\[【（(].*[\]】）)]", text):
        return None
    text = text.strip("[]【】")
    text = text.replace("一", "-") if field in {
        "vasomotion", "wbc_count", "sweat_duct"
    } else text
    if field in {
        "capillary_count", "vasomotion", "wbc_count", "sweat_duct",
        "crossing_ratio", "malformation_ratio",
    }:
        text = re.sub(r"(?<=\d)-(?=\d)", "--", text)
        text = re.sub(r"-{3,}", "--", text)
    aliases = {
        "capillary_count": {
            ">7": ">=7", "≥7": ">=7", "7以上": ">=7",
            "7--9": ">=7", "7-9": ">=7",
        },
        "crossing_ratio": {
            "<30%": "<=30%", "≤30%": "<=30%", "30%-60%": "30--60%",
            "60%-80%": "60--80%",
        },
        "malformation_ratio": {
            "<10%": "<=10%", "≤10%": "<=10%", "10%-30%": "10--30%",
            "30%-60%": "30--60%",
        },
        "microthrombus": {"未见": "无", "不见": "无"},
        "hemorrhage": {"未见": "无", "不见": "无"},
        "exudation": {"未见": "无", "不见": "无"},
        "subpapillary_venous_plexus": {"未见": "不见", "无": "不见"},
        "blood_color": {"淡红色": "淡红", "浅红色": "浅红", "暗红色": "暗红"},
    }
    text = aliases.get(field, {}).get(text, text)

    # Units and fragments from neighbouring columns are not measurements.
    unit_fragments = (
        "条/mm", "个/15s", "个/min", "次/min", "um/s", "μm/s",
        "管袢/一指甲襞", "个/-指甲襞", "个/一指甲襞",
    )
    if text in unit_fragments:
        return None
    text = re.sub(r"(条/mm|个/15s|个/min|次/min|个/[一-]指甲襞)$", "", text)

    # Closed vocabularies prevent rare OCR garbage from becoming a class of its
    # own.  Values outside the protocol remain in the audit but are not labels.
    allowed = {
        "clarity": {"清晰", "模糊", "不清"},
        "capillary_count": {"<1", "1--2", "3--4", "5--6", ">=7"},
        "crossing_ratio": {"<=30%", "10--30%", "30--60%", "60--80%", ">80%"},
        "malformation_ratio": {"<=10%", "10--30%", "30--60%", "60--80%", ">60%"},
        "blood_color": {"淡红", "浅红", "暗红", "暗紫"},
        "exudation": {"无", "+", "++", "+++"},
        "hemorrhage": {"无", "1--2", "3--4", ">2", ">=5"},
        "subpapillary_venous_plexus": {
            "不见", "可见1排", "可见2排", ">2排,扩张",
        },
        "papilla": {"平坦", "浅波纹状", "波纹状"},
        "sweat_duct": {"0--2", "3--4", ">=5"},
        "flow_state": {
            "全停", "停-流", "粒摆流", "粒缓流", "粒流",
            "粒线流", "线粒流", "线流",
        },
        "vasomotion": {"0--1", "2--4", ">4"},
        "rbc_aggregation": {"无", "轻度", "中度", "重度"},
        "wbc_count": {"0", "1--30", ">30"},
        "microthrombus": {"无", "1--2", ">2", "有"},
    }
    if field in allowed and text not in allowed[field]:
        return None
    return text


def validate_ocr_payload(payload: dict[str, Any]) -> dict[str, str | None]:
    return {
        field: canonical_field_value(field, payload.get(field))
        for field in ALL_REPORT_FIELDS
    }


def consensus_for_case(
    report_payloads: Iterable[dict[str, Any]],
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    payloads = [validate_ocr_payload(payload) for payload in report_payloads]
    consensus: dict[str, str | None] = {}
    audit: dict[str, dict[str, Any]] = {}
    for field in ALL_REPORT_FIELDS:
        values = [payload[field] for payload in payloads]
        non_null = [value for value in values if value is not None]
        counts = Counter(non_null)
        if counts:
            selected, selected_count = counts.most_common(1)[0]
            conflict = len(counts) > 1 and selected_count <= len(non_null) / 2
        else:
            selected, conflict = None, False
        consensus[field] = selected
        audit[field] = {
            "raw_values": values,
            "counts": dict(counts),
            "selected": selected,
            "conflict": conflict,
            "support": counts.get(selected, 0) if selected is not None else 0,
            "copies": len(values),
        }
    return consensus, audit


def group_ocr_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[str(row["exam_case_id"])].append(dict(row["fields"]))
    return dict(grouped)

