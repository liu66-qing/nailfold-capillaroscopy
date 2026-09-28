"""Printable health-observation page. Self-contained HTML: no network, no LLM.

This is a health-management product, not a medical device, so the page is laid
out as a personal health card rather than a clinical report sheet:
  - a summary banner
  - the image strip
  - the 20 observations as a numbered two-column list (DISPLAY_ORDER);
    no row is highlighted
  - one card per theme: what was seen, what it is commonly seen with, what to do
    (each action says which finding it is for)
  - how to read a single capture, and when to see a doctor
No emoji: the readers are mostly middle-aged and older.
The page never shows a reference column, a confidence mark, a patient ID or a
signature line.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Any

from .final_advice import check_text
from .final_registry import FIELDS

esc = html.escape

# Printed order. Left column: image and loop shape, as on the paper sheet.
# Right column: the rows that vary little between people are interleaved with rows
# that do vary, so no two of them stand next to each other.
DISPLAY_ORDER = (
    "clarity", "capillary_count", "afferent_diameter", "efferent_diameter",
    "output_input_ratio", "apex_diameter", "loop_length", "crossing_ratio",
    "malformation_ratio", "vasomotion",
    "flow_state", "microthrombus", "rbc_aggregation", "blood_color", "wbc_count",
    "exudation", "subpapillary_venous_plexus", "hemorrhage", "papilla", "sweat_duct",
)
LOW_VARIATION = ("flow_state", "vasomotion", "rbc_aggregation", "wbc_count", "hemorrhage",
                 "sweat_duct")

# one-line plain explanations printed under technical item names in the table
TIPS = {
    "afferent_diameter": "血液流入的一侧",
    "efferent_diameter": "血液流出的一侧",
    "apex_diameter": "管袢顶端的弯曲处",
    "subpapillary_venous_plexus": "皮肤浅层的细小回流血管网",
    "microthrombus": "血细胞短暂聚在一起的样子",
    "papilla": "甲襞皮肤表面的纹理",
}


def _line(n: int, f: str, d: dict[str, Any]) -> str:
    # no highlight: every row is printed the same way
    tip = "<small>%s</small>" % esc(TIPS[f]) if f in TIPS else ""
    return ("<li><span class='n'>%02d</span><span class='k'>%s%s</span>"
            "<span class='v'>%s</span></li>" % (n, esc(d["item"]), tip, esc(d["value"])))


def _thumbs(images: list[bytes] | None) -> str:
    if not images:
        return ""
    cells = "".join("<img alt='甲襞图像 %d' src='data:image/jpeg;base64,%s'>"
                    % (i + 1, base64.b64encode(b).decode()) for i, b in enumerate(images[:4]))
    return "<div class='imgs'>%s</div>" % cells


def thumbnail_bytes(paths: list[Path], width: int = 360) -> list[bytes]:
    import cv2
    import numpy as np
    out = []
    for p in paths[:4]:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h = int(im.shape[0] * width / im.shape[1])
        ok, buf = cv2.imencode(".jpg", cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            out.append(buf.tobytes())
    return out


def _section(s: dict[str, Any]) -> str:
    acts = "".join("<li>%s%s</li>" % (esc(a["text"]), "<span class='for'>针对：%s</span>"
                                      % esc(a["for"]) if a["for"] else "")
                   for a in s["actions"])
    todo = f"<div class='todo'><b>可以这样做</b><ul>{acts}</ul></div>" if acts else ""
    return (f"<section class='theme t-{esc(s['theme'])}'>"
            f"<h3>{esc(s['title'])}</h3>"
            f"<p><b>看到了</b>{esc(s['seen'])}</p>"
            f"<p><b>这意味着</b>{esc(s['meaning'])}</p>"
            f"{todo}</section>")


def render_html(report: dict[str, Any], output: Path | None = None,
                patient: dict[str, Any] | None = None,
                images: list[bytes] | None = None) -> str:
    patient = patient or {}
    fields, adv = report["fields"], report["advice"]
    ids = list(DISPLAY_ORDER)
    assert sorted(ids) == sorted(f[0] for f in FIELDS)
    half = (len(ids) + 1) // 2
    col = lambda part, start: "<ol class='obs'>%s</ol>" % "".join(  # noqa: E731
        _line(start + i, f, fields[f]) for i, f in enumerate(part))
    groups = col(ids[:half], 1) + col(ids[half:], half + 1)
    sections = "".join(_section(s) for s in adv["sections"])
    who = "　".join(esc(str(patient[k])) + (" 岁" if k == "age" else "")
                   for k in ("name", "sex", "age") if patient.get(k))
    date = esc(str(patient.get("exam_date", "")))
    # where the result comes from: images, when, which finger, room temperature
    src = ["基于本次上传的 %d 张图像" % report["images"]]
    when = " ".join(str(patient[k]) for k in ("exam_date", "capture_time") if patient.get(k))
    if when:
        src.append("拍摄于 " + when)
    if patient.get("finger"):
        src.append("手指：" + str(patient["finger"]))
    if patient.get("room_temp"):
        src.append("室温约 %s ℃" % patient["room_temp"])
    source = esc("，".join(src) + "。")
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>甲襞微循环健康观察 {esc(report["exam_id"])}</title>
<style>
@page {{ size:A4; margin:10mm 12mm; }}
* {{ box-sizing:border-box; }}
body {{ font-family:"PingFang SC","Microsoft YaHei",sans-serif; color:#2b2a33; margin:0 auto;
 max-width:190mm; font-size:12.5px; line-height:1.65; background:#fff; }}
.hero {{ background:linear-gradient(135deg,#fff1e8 0%,#fde3ea 55%,#efe6fb 100%);
 border-radius:16px; padding:14px 18px 12px; }}
.hero .top {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; }}
.hero h1 {{ font-size:20px; margin:0; color:#8a3552; font-weight:700; }}
.hero .who {{ color:#7a6470; font-size:11.5px; text-align:right; }}
.hero .sum {{ font-size:14px; margin:8px 0 0; color:#4a3140; }}
.imgs {{ display:flex; gap:8px; margin:10px 0 2px; }}
.imgs img {{ flex:1; min-width:0; height:88px; object-fit:cover; border-radius:10px; }}
h2 {{ font-size:14.5px; margin:14px 0 6px; color:#8a3552; }}
.grps {{ display:grid; grid-template-columns:1fr 1fr; gap:0 14px; background:#faf8fb;
 border-radius:14px; padding:6px 12px; }}
.obs {{ list-style:none; margin:0; padding:0; }}
.obs li {{ display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:8px;
 border-bottom:1px dashed #ebe4ee; }}
.obs li:last-child {{ border-bottom:0; }}
.obs .n {{ color:#c2b3c9; font-size:10.5px; width:18px; }}
.obs .k {{ flex:1; color:#5d5563; }} .obs .v {{ font-weight:600; text-align:right; }}
.obs small {{ display:block; color:#aaa0ae; font-size:10px; line-height:1.2; }}
.for {{ display:block; color:#9a8791; font-size:10.5px; }}
.src {{ color:#7a6470; font-size:11px; margin:4px 0 0; }}
.read {{ color:#7a6470; font-size:11.5px; margin:8px 2px 0; }}

.theme {{ border-radius:14px; padding:10px 14px; margin-top:9px; background:#faf8fb;
 border-left:5px solid #c9b6d6; page-break-inside:avoid; }}
.theme h3 {{ margin:0 0 4px; font-size:13.5px; }}
.theme p {{ margin:2px 0; }}
.theme b {{ display:inline-block; min-width:58px; color:#7a6470; font-weight:600; margin-right:4px; }}
.todo ul {{ margin:2px 0 0 18px; padding:0; }} .todo li {{ margin:1px 0; }}
.t-photo {{ border-color:#8fb3d9; background:#f3f7fc; }}
.t-cold {{ border-color:#e6a57e; background:#fff6f0; }}
.t-slow {{ border-color:#86c3a8; background:#f2faf6; }}
.t-shape {{ border-color:#b9c77a; background:#f8faef; }}
.t-env {{ border-color:#8cc5d4; background:#f1f9fb; }}
.t-keep {{ border-color:#86c3a8; background:#f2faf6; }}
.doc {{ margin-top:10px; border-radius:12px; padding:8px 14px; background:#fff8e6; color:#6b5520; }}
footer {{ margin-top:10px; color:#9a939c; font-size:10.5px; text-align:center; }}
@media (max-width:600px) {{ .grps {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ font-size:11.5px; }} .imgs img {{ height:76px; }} }}
</style></head><body>
<div class="hero"><div class="top"><h1>甲襞微循环健康观察</h1>
<div class="who">{who}{"<br>" if who and date else ""}{date}</div></div>
<p class="sum">{esc(adv["headline"])}</p>
<p class="src">{source}</p></div>
{_thumbs(images)}
<h2>这次看到的</h2>
<div class="grps">{groups}</div>
<h2>解读与建议</h2>
{sections}
<p class="read">{esc(adv["how_to_read"])}</p>
<div class="doc">{esc(adv["see_doctor"])}</div>
<footer>{esc(adv["disclaimer"])}　·　编号 {esc(report["exam_id"])}</footer>
</body></html>"""
    check_text(re.sub(r"<img[^>]*>", "", doc.split("</style>", 1)[1]))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(doc, encoding="utf-8")
    return doc
