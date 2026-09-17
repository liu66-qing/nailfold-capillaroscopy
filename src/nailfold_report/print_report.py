from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any


REPORT_ROWS = (
    ("clarity", "清晰度", "", "清晰", "morphology"),
    ("capillary_count", "管袢数", "条/mm", "≥7条/mm", "morphology"),
    ("afferent_diameter", "输入枝管径", "μm", "9–13μm", "morphology"),
    ("efferent_diameter", "输出枝管径", "μm", "11–17μm", "morphology"),
    ("output_input_ratio", "输出/输入枝管径", "", "1.3", "morphology"),
    ("apex_diameter", "袢顶直径", "μm", "12–18μm", "morphology"),
    ("loop_length", "管袢长", "μm", "150–250μm", "morphology"),
    ("crossing_ratio", "交叉管袢数", "", "<30%", "morphology"),
    ("malformation_ratio", "畸形管袢数", "", "<10%", "morphology"),
    ("flow_state", "流态", "", "线流、线粒流", "flow"),
    ("vasomotion", "血管运动性", "次/min", "0–1次/min", "flow"),
    ("rbc_aggregation", "红细胞聚集", "", "无", "flow"),
    ("wbc_count", "白细胞数", "个/15s", "1–30个/15s", "flow"),
    ("microthrombus", "白微栓", "个/min", "未见", "flow"),
    ("blood_color", "血色", "", "淡红色", "periloop"),
    ("exudation", "渗出", "", "无", "periloop"),
    ("hemorrhage", "出血", "管袢/一指甲襞", "未见", "periloop"),
    ("subpapillary_venous_plexus", "乳头下静脉丛", "", "不见", "periloop"),
    ("papilla", "乳头", "", "波纹状", "periloop"),
    ("sweat_duct", "汗腺导管", "个/一指甲襞", "0–2个/一指甲襞", "periloop"),
)


def _display_value(field: str, details: dict[str, Any]) -> str:
    value = details.get("value")
    if value is None:
        return "—"
    if isinstance(value, float):
        digits = 1 if field != "output_input_ratio" else 2
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value).replace("--", "–").replace("<=", "≤").replace(">=", "≥")


def _image_uri(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode()}"


def render_report_html(report: dict[str, Any], output: Path) -> Path:
    """Create a self-contained A4 report suitable for browser print/PDF."""
    fields = report.get("fields", {})
    scores = report.get("scores", {})
    contributions = scores.get("field_contributions", {})
    patient = report.get("patient", {})
    rows = []
    for field, label, unit, normal, group in REPORT_ROWS:
        details = fields.get(field, {})
        status = str(details.get("status", ""))
        low = "low" if "low_" in status or "uncalibrated" in status else ""
        score = contributions.get(field)
        score_text = "—" if score is None else f"{float(score):.1f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td class='value {low}'>{html.escape(_display_value(field, details))}</td>"
            f"<td>{html.escape(unit)}</td>"
            f"<td>{html.escape(normal)}</td>"
            f"<td>{score_text}</td>"
            "</tr>"
        )
    pictures = []
    for path in report.get("evidence_images", [])[:4]:
        uri = _image_uri(path)
        if uri:
            pictures.append(f"<img src='{uri}' alt='甲襞证据图'>")
    warnings = report.get("quality", {}).get("warnings", [])
    warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
    recommendations = report.get("recommendations", [])
    recommendation_html = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in recommendations
    )
    case_id = report.get("case_id", "")
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>甲襞微循环检测报告 {html.escape(str(case_id))}</title>
<style>
@page {{ size: A4; margin: 10mm 12mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Microsoft YaHei","SimSun",sans-serif; color:#111; margin:0; font-size:11px; }}
.toolbar {{ position:fixed; right:18px; top:12px; z-index:3; }}
.toolbar button {{ padding:8px 18px; border:0; border-radius:4px; background:#1769aa; color:white; }}
h1 {{ text-align:center; font-size:21px; letter-spacing:5px; margin:0 0 7px; }}
.meta {{ display:grid; grid-template-columns:repeat(4,1fr); gap:3px 14px; border-top:2px solid #111;
 border-bottom:1px solid #555; padding:6px 3px; margin-bottom:5px; }}
.meta span {{ min-height:16px; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
th,td {{ border:1px solid #777; padding:3px 5px; text-align:center; height:24px; }}
th {{ background:#eee; font-weight:600; }}
td:first-child {{ text-align:left; padding-left:10px; }}
.value {{ font-weight:700; font-size:12px; }}
.value.low {{ color:#a33; text-decoration:underline dotted; }}
.summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-top:6px; font-size:12px; }}
.summary div {{ border:1px solid #777; padding:5px; }}
.images {{ display:grid; grid-template-columns:repeat(4,1fr); gap:4px; margin-top:6px; }}
.images img {{ width:100%; height:105px; object-fit:cover; border:1px solid #777; }}
.notice {{ margin-top:6px; padding:5px 8px; border:1px solid #b98; background:#fff8e8; }}
.notice ul {{ margin:3px 0 0 18px; padding:0; }}
.recommendations {{ margin-top:6px; padding:6px 8px; border:1px solid #8aa6b8; background:#f3f8fb; page-break-inside:avoid; }}
.recommendations ul {{ margin:3px 0 0 18px; padding:0; }}
.footer {{ margin-top:5px; display:flex; justify-content:space-between; color:#555; }}
@media print {{ .toolbar {{ display:none; }} body {{ font-size:10.5px; }} }}
</style></head><body>
<div class="toolbar"><button onclick="window.print()">打印 / 保存PDF</button></div>
<h1>甲襞微循环检测报告</h1>
<div class="meta">
 <span>姓名：{html.escape(str(patient.get("name","")))}</span>
 <span>性别：{html.escape(str(patient.get("sex","")))}</span>
 <span>年龄：{html.escape(str(patient.get("age","")))}</span>
 <span>病例号：{html.escape(str(patient.get("patient_id",case_id)))}</span>
 <span>检查日期：{html.escape(str(patient.get("exam_date","")))}</span>
 <span>检查手指：{html.escape(str(patient.get("finger","")))}</span>
 <span>设备：Windows PC + UVC</span>
 <span>报告版本：{html.escape(str(report.get("model_version","")))}</span>
</div>
<table><thead><tr><th>测量项目</th><th>测量值</th><th>单位</th><th>正常值</th><th>积分</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="summary">
 <div>形态积分：<b>{scores.get("morphology_score","—")}</b></div>
 <div>流态积分：<b>{scores.get("flow_score","—")}</b></div>
 <div>袢周积分：<b>{scores.get("periloop_score","—")}</b></div>
 <div>总积分：<b>{scores.get("total_score","—")}</b></div>
</div>
<div class="summary"><div style="grid-column:span 4">综合判断：
 <b>{html.escape(str(scores.get("overall_assessment","—")))}</b></div></div>
<div class="images">{''.join(pictures)}</div>
<div class="notice"><b>质量与复核提示：</b>
 <ul>{warning_html or "<li>无自动质量警告；模型结果仍须由专业人员结合原图解释。</li>"}</ul>
</div>
<div class="recommendations"><b>健康提示（非诊断建议）：</b>
 <ul>{recommendation_html or "<li>保持规律作息、适度活动和手部保暖；如有持续不适，请咨询专业医生。</li><li>以上内容不构成诊断、处方或保健品推荐。</li>"}</ul>
</div>
<div class="footer"><span>本报告为研发辅助分析结果，不替代临床诊断。</span>
<span>检查编号：{html.escape(str(case_id))}</span></div>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def render_report_file(report_json: Path, output: Path | None = None) -> Path:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    return render_report_html(
        report, output or report_json.with_name("report_print.html")
    )
