from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path


FIELD_LABELS = {
    "clarity": "清晰度", "capillary_count": "管袢数",
    "afferent_diameter": "输入枝管径", "efferent_diameter": "输出枝管径",
    "output_input_ratio": "输出/输入枝管径", "apex_diameter": "袢顶直径",
    "loop_length": "管袢长", "crossing_ratio": "交叉管袢数",
    "malformation_ratio": "畸形管袢数", "flow_state": "流态",
    "vasomotion": "血管运动性", "rbc_aggregation": "红细胞聚集",
    "wbc_count": "白细胞数", "microthrombus": "白微栓",
    "blood_color": "血色", "exudation": "渗出", "hemorrhage": "出血",
    "subpapillary_venous_plexus": "乳头下静脉丛", "papilla": "乳头",
    "sweat_duct": "汗腺导管",
}


def field_names(fields: list[str]) -> str:
    return "、".join(FIELD_LABELS.get(field, field) for field in fields)


def capture(
    camera_index: int, output: Path, seconds: float, fps: float
) -> tuple[list[Path], Path]:
    import cv2

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        raise SystemExit(f"cannot open UVC camera index {camera_index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    camera.set(cv2.CAP_PROP_FPS, 30)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    interval = 1.0 / fps
    dynamic_interval = 0.1
    started = next_capture = next_dynamic = time.monotonic()
    decoded = 0
    writer = None
    video_path = output / "uvc_10fps.avi"
    try:
        while time.monotonic() - started < seconds:
            ok, frame = camera.read()
            if not ok:
                continue
            decoded += 1
            now = time.monotonic()
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height)
                )
            if now >= next_dynamic:
                writer.write(frame)
                next_dynamic += dynamic_interval
            if now >= next_capture:
                path = output / f"frame_{len(paths):04d}.jpg"
                if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                    paths.append(path)
                next_capture += interval
    finally:
        camera.release()
        if writer is not None:
            writer.release()
    if len(paths) < 4:
        raise SystemExit(f"only {len(paths)} frames captured from {decoded} decoded")
    return paths, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/windows_model_v3")
    )
    parser.add_argument("--output-root", type=Path, default=Path("captures"))
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--videos", type=Path, nargs="*")
    parser.add_argument(
        "--video-model-dir", type=Path, default=Path("artifacts/video_v3_final")
    )
    parser.add_argument(
        "--geometry-model-dir", type=Path, default=Path("artifacts/geometry_final_v3")
    )
    parser.add_argument("--patient-name", default="")
    parser.add_argument("--patient-id", default="")
    parser.add_argument("--sex", default="")
    parser.add_argument("--age", default="")
    parser.add_argument("--finger", default="")
    args = parser.parse_args()
    from nailfold_report.windows_inference import WindowsCasePredictor

    case_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = args.output_root / case_id
    if args.images:
        paths, videos = args.images, (args.videos or [])
    else:
        paths, video = capture(
            args.camera, case_dir / "frames", args.seconds, args.sample_fps
        )
        videos = [video]
    predictor = WindowsCasePredictor(args.model_dir)
    result = predictor.predict(paths)
    if (args.geometry_model_dir / "metadata.json").exists():
        from nailfold_report.geometry_inference import GeometryPredictor

        geometry = GeometryPredictor(args.geometry_model_dir).predict(
            [Path(path) for path in result["evidence_images"]]
        )
        result["fields"].update(geometry)
        afferent = result["fields"]["afferent_diameter"]["value"]
        efferent = result["fields"]["efferent_diameter"]["value"]
        result["fields"]["output_input_ratio"] = {
            "value": efferent / afferent if afferent and afferent > 0 else None,
            "confidence": None,
            "status": "derived_from_geometry_routed_diameters",
        }
    if videos and (args.video_model_dir / "metadata.json").exists():
        from nailfold_report.video_inference import DynamicPredictor
        try:
            result["fields"].update(
                DynamicPredictor(args.video_model_dir).predict(videos, predictor)
            )
        except Exception as error:
            result["quality"]["needs_review"] = True
            result["quality"]["warnings"].append(
                "视频无法解析，已保留静态报告；5个动态字段未生成："
                + str(error)
            )
        else:
            low_support = [
                field for field, value in result["fields"].items()
                if value.get("status") == "predicted_low_support"
            ]
            if low_support:
                result["quality"]["needs_review"] = True
                result["quality"]["warnings"].append(
                    "以下动态字段训练类别支持不足，不可作性能宣称："
                    + field_names(low_support)
                )
    if predictor.score_rules is not None:
        from nailfold_report.windows_inference import deterministic_scores

        result["scores"] = deterministic_scores(
            result["fields"], predictor.score_rules
        )
    result["quality"]["warnings"] = [
        warning for warning in result["quality"]["warnings"]
        if not warning.startswith("以下分类字段低于未校准置信度阈值")
    ]
    low_confidence = [
        field for field, details in result["fields"].items()
        if details.get("confidence") is not None
        and details["confidence"] < 0.55
        and not details.get("status", "").endswith("low_support")
    ]
    if low_confidence:
        result["quality"]["warnings"].insert(
            0,
            "以下分类字段低于未校准置信度阈值："
            + field_names(low_confidence),
        )
    result["quality"]["needs_review"] = bool(result["quality"]["warnings"])
    result["case_id"] = case_id
    result["patient"] = {
        "name": args.patient_name,
        "patient_id": args.patient_id or case_id,
        "sex": args.sex,
        "age": args.age,
        "finger": args.finger,
        "exam_date": date.today().isoformat(),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from nailfold_report.print_report import render_report_html

    render_report_html(result, case_dir / "report_print.html")
    # ensure_ascii keeps the subprocess protocol independent of the Windows code page.
    print(json.dumps({"report_path": str(report_path.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
