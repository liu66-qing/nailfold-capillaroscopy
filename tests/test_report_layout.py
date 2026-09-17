from pathlib import Path

from nailfold_report.data.report_layout import REPORT_FIELDS, report_field_presence


def test_report_field_names_are_unique() -> None:
    names = [field.name for field in REPORT_FIELDS]
    assert len(names) == len(set(names))


def test_blank_report_has_no_measurement_values() -> None:
    report = Path(
        r"E:\甲劈微循环\data\recovered_archive1\12\rep.jpg.jpg"
    )
    if report.exists():
        assert not any(report_field_presence(report).values())
