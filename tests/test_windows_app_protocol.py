from pathlib import Path

from scripts.windows_report_app import parse_report_path


def test_report_path_protocol_survives_non_ascii_workspace() -> None:
    stdout = '{"report_path": "E:\\\\u7532\\\\captures_app\\\\1\\\\report.json"}\n'
    assert parse_report_path(stdout) == Path("E:\\u7532\\captures_app\\1\\report.json")


def test_report_path_protocol_accepts_legacy_output() -> None:
    assert parse_report_path("C:\\project\\captures\\1\\report.json\n") == Path(
        "C:\\project\\captures\\1\\report.json"
    )
