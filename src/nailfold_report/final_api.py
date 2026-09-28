"""HTTP wrapper for the final report.

    uvicorn nailfold_report.final_api:app --host 127.0.0.1 --port 8000

POST /v1/reports  multipart: exam_id, device_id (optional), images (1..n files)
    -> report JSON (schema release/final_v1/schemas/report.schema.json);
       ?format=html returns the printable page instead
GET  /v1/health   -> release id and asset sha256

There is no authentication. Bind to localhost or put it behind the platform's
gateway; do not expose it directly. Uploaded images are written to a
temporary directory and deleted after the request.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .final_inference import FinalReportPredictor
from .final_registry import (BUNDLE_SHA256, DET_SHA256, ENCODER_SHA256, RELEASE_ID,
                             SEG_SHA256)
from .final_render import render_html, thumbnail_bytes

ROOT = Path(os.environ.get("NAILFOLD_ROOT", Path(__file__).resolve().parents[2]))
MAX_IMAGES = 64
MAX_BYTES = 20 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FINGERS = {"左手无名指", "右手无名指", "左手中指", "右手中指", "其他"}

app = FastAPI(title="nailfold final report", version=RELEASE_ID)
_predictor: FinalReportPredictor | None = None


def predictor() -> FinalReportPredictor:
    global _predictor
    if _predictor is None:
        # NAILFOLD_DEVICE=cpu keeps the GPU free for other jobs
        _predictor = FinalReportPredictor(ROOT, device=os.environ.get("NAILFOLD_DEVICE"))
    return _predictor


UPLOAD_PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>甲襞微循环健康观察 · 本地测试</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
label{display:block;margin:12px 0 4px}input{font-size:14px}
button{margin-top:16px;padding:8px 20px;font-size:14px}#msg{margin-top:12px;color:#555}</style>
</head><body><h2>甲襞微循环健康观察 · 本地测试</h2>
<form id="f" action="/v1/reports?format=html" method="post" enctype="multipart/form-data">
<label for="exam_id">检查编号</label><input id="exam_id" name="exam_id" value="local-test" required>
<label for="device_id">设备编号（可选）</label><input id="device_id" name="device_id" value="unknown">
<label for="finger">拍摄手指（可选）</label>
<select id="finger" name="finger"><option value="">不填</option><option>左手无名指</option>
<option>右手无名指</option><option>左手中指</option><option>右手中指</option><option>其他</option></select>
<label for="room_temp">室温，℃（可选）</label><input id="room_temp" name="room_temp" type="number" min="0" max="45" step="1">
<label for="images">同一次检查的图像（可多选）</label>
<input id="images" name="images" type="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" multiple required>
<button type="submit">生成健康观察</button></form>
<p id="msg" role="status"></p>
<script>document.getElementById('f').onsubmit=function(){document.getElementById('msg').textContent=
'正在分析，CPU 上每张图约需 5~10 秒，请稍候……'};</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return UPLOAD_PAGE


@app.get("/examples/{n}", response_class=HTMLResponse)
def example(n: int) -> str:
    f = ROOT / "release/final_v1/examples" / ("example_%d.html" % n)
    if not 1 <= n <= 9 or not f.exists():
        raise HTTPException(404, "no such example")
    return f.read_text(encoding="utf-8")


@app.get("/v1/health")
def health() -> dict:
    return dict(release_id=RELEASE_ID, bundle_sha256=BUNDLE_SHA256,
                encoder_sha256=ENCODER_SHA256, seg_sha256=SEG_SHA256, det_sha256=DET_SHA256)


@app.post("/v1/reports")
async def create_report(exam_id: str = Form(..., min_length=1, max_length=128),
                        device_id: str = Form("unknown", max_length=128),
                        finger: str = Form("", max_length=16),
                        room_temp: str = Form("", max_length=4),
                        images: list[UploadFile] = File(...),
                        format: str = "json"):
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(422, "need 1..%d images" % MAX_IMAGES)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, up in enumerate(images):
            ext = Path(up.filename or "").suffix.lower()
            if ext not in ALLOWED:
                raise HTTPException(415, "unsupported image type: %s" % ext)
            data = await up.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "image too large")
            p = Path(tmp) / ("%03d%s" % (i, ext))   # never trust the client filename
            p.write_bytes(data)
            paths.append(p)
        try:
            report = predictor().predict(paths, exam_id=exam_id, device_id=device_id)
            thumbs = thumbnail_bytes(paths) if format == "html" else None
        except (OSError, ValueError) as e:
            raise HTTPException(422, "could not read images: %s" % type(e).__name__)
    if format == "html":
        from datetime import datetime
        now = datetime.now()
        temp = room_temp if room_temp.isdigit() and 0 <= int(room_temp) <= 45 else ""
        info = dict(exam_date=now.strftime("%Y-%m-%d"),
                    capture_time="上午" if now.hour < 12 else "下午",
                    finger=finger if finger in FINGERS else "", room_temp=temp)
        return HTMLResponse(render_html(report, images=thumbs, patient=info))
    return report
