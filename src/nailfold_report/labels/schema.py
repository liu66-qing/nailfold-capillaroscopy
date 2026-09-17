from __future__ import annotations


MEASUREMENT_FIELDS: tuple[str, ...] = (
    "clarity",
    "capillary_count",
    "afferent_diameter",
    "efferent_diameter",
    "output_input_ratio",
    "apex_diameter",
    "loop_length",
    "crossing_ratio",
    "malformation_ratio",
    "flow_state",
    "flow_speed_um_s",
    "vasomotion",
    "rbc_aggregation",
    "wbc_count",
    "microthrombus",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
    "sweat_duct",
)

SCORE_FIELDS: tuple[str, ...] = (
    "morphology_score",
    "flow_score",
    "periloop_score",
    "total_score",
    "overall_assessment",
)

ALL_REPORT_FIELDS = MEASUREMENT_FIELDS + SCORE_FIELDS


REPORT_OCR_PROMPT = """
你是固定版式中文医疗报告的文字转录器，不进行医学推断。
只读取图片上方表格中“测量值”一列、下方四个积分以及“综合判断”。
禁止根据正常值、积分、建议文字或临床图片猜测缺失内容。
图片中空白就输出 null；保留原始范围符号、百分号、加减号和中文原词。
特别注意：
1. “流速”第一行是流态文字；其下一行 um/s 才是 flow_speed_um_s。
2. “无”“不见”“未见”“0”和空白不能互相替换。
3. 不要改正看起来异常的数值。
4. 只返回一个 JSON 对象，不要解释，不要 Markdown。

JSON 键必须完整且仅包含：
{
  "clarity": null,
  "capillary_count": null,
  "afferent_diameter": null,
  "efferent_diameter": null,
  "output_input_ratio": null,
  "apex_diameter": null,
  "loop_length": null,
  "crossing_ratio": null,
  "malformation_ratio": null,
  "flow_state": null,
  "flow_speed_um_s": null,
  "vasomotion": null,
  "rbc_aggregation": null,
  "wbc_count": null,
  "microthrombus": null,
  "blood_color": null,
  "exudation": null,
  "hemorrhage": null,
  "subpapillary_venous_plexus": null,
  "papilla": null,
  "sweat_duct": null,
  "morphology_score": null,
  "flow_score": null,
  "periloop_score": null,
  "total_score": null,
  "overall_assessment": null
}
""".strip()
