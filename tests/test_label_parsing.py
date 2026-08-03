from nailfold_report.labels.parsing import (
    canonical_field_value,
    consensus_for_case,
    extract_json_object,
    normalize_raw_value,
)


def test_extract_json_object_from_wrapped_text() -> None:
    expected = {"clarity": "\u6e05\u6670"}
    assert extract_json_object(f"answer: {expected!r}".replace("'", '"')) == expected


def test_normalize_preserves_missing_semantics() -> None:
    assert normalize_raw_value("\u65e0") == "\u65e0"
    assert normalize_raw_value("\u4e0d\u89c1") == "\u4e0d\u89c1"
    assert normalize_raw_value("") is None
    assert normalize_raw_value("30\uff05") == "30%"


def test_consensus_uses_report_copies() -> None:
    consensus, audit = consensus_for_case(
        [
            {"clarity": "\u6e05\u6670"},
            {"clarity": "\u6e05\u6670"},
            {"clarity": "\u5c1a\u6e05"},
        ]
    )
    assert consensus["clarity"] == "\u6e05\u6670"
    assert not audit["clarity"]["conflict"]


def test_field_canonicalization_preserves_ranges() -> None:
    assert canonical_field_value("crossing_ratio", "30–60％") == "30--60%"
    assert canonical_field_value("capillary_count", ">7") == ">=7"
    assert canonical_field_value("microthrombus", "未见") == "无"


def test_field_canonicalization_rejects_template_and_unit_columns() -> None:
    assert canonical_field_value("exudation", "[无]") is None
    assert canonical_field_value("blood_color", "[淡红色]") is None
    assert canonical_field_value("capillary_count", "条/mm") is None
    assert canonical_field_value("sweat_duct", "0--2个/一指甲襞") == "0--2"
    assert canonical_field_value("hemorrhage", "管袢/一指甲襞") is None
    assert canonical_field_value("vasomotion", "中度") is None
    assert canonical_field_value("wbc_count", ">2") is None
