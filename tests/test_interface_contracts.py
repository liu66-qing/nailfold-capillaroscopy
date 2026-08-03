import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).parents[1]
CONTRACTS = (
    ("analysis-request.schema.json", "analysis-request.example.json"),
    ("task-status.schema.json", "task-status.example.json"),
    ("error-response.schema.json", "error-response.example.json"),
    ("nailfold-report.schema.json", "report.full.example.json"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_interface_examples_match_schemas() -> None:
    for schema_name, example_name in CONTRACTS:
        schema = load_json(ROOT / "schemas" / schema_name)
        example = load_json(ROOT / "examples" / example_name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(example)


def test_full_report_contains_exactly_current_twenty_output_fields() -> None:
    report = load_json(ROOT / "examples" / "report.full.example.json")
    assert len(report["fields"]) == 20
    assert "flow_speed_um_s" not in report["fields"]
