import json
from pathlib import Path

from nailfold_report.windows_inference import deterministic_scores


def test_report_scores_are_deterministic_and_additive() -> None:
    path = Path("artifacts/windows_model_adapt/score_rules.json")
    if not path.exists():
        return
    rules = json.loads(path.read_text(encoding="utf-8"))
    values = {
        "clarity": "清晰",
        "capillary_count": ">=7",
        "afferent_diameter": 10.0,
        "efferent_diameter": 13.0,
        "output_input_ratio": 1.3,
        "apex_diameter": 16.0,
        "loop_length": 180.0,
        "crossing_ratio": "<=30%",
        "malformation_ratio": "<=10%",
        "flow_state": "线粒流",
        "flow_speed_um_s": None,
        "vasomotion": "0--1",
        "rbc_aggregation": "无",
        "wbc_count": "1--30",
        "microthrombus": "无",
        "blood_color": "淡红",
        "exudation": "无",
        "hemorrhage": "无",
        "subpapillary_venous_plexus": "不见",
        "papilla": "波纹状",
        "sweat_duct": "0--2",
    }
    fields = {field: {"value": value} for field, value in values.items()}
    result = deterministic_scores(fields, rules)
    assert result["total_score"] == round(
        result["morphology_score"]
        + result["flow_score"]
        + result["periloop_score"],
        1,
    )
    assert result["overall_assessment"] in rules["assessment_rule"]["labels"]
