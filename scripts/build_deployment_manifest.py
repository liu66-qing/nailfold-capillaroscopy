from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/deployment_manifest_v3.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    sources = (
        "artifacts/windows_model_v3", "artifacts/geometry_final_v3",
        "artifacts/video_v3_final",
    )
    files = []
    for source in sources:
        for path in sorted((root / source).glob("*")):
            if path.is_file() and path.name != "train.log":
                files.append(path)
    for relative in (
        "artifacts/labels/multisource_confidence_v3.csv",
        "artifacts/labels/multisource_confidence_v3.summary.json",
        "artifacts/labels/score_rules_v3.json",
        "artifacts/labelv2_locked_metrics.json",
        "artifacts/geometry_deploy_v3_locked/metrics.json",
        "artifacts/video_v3_cv_metrics.json",
        "scripts/windows_uvc_report.py",
        "scripts/windows_report_app.py",
        "scripts/render_print_report.py",
        "src/nailfold_report/print_report.py",
        "启动甲襞分析系统.bat",
    ):
        path = root / relative
        if path.exists():
            files.append(path)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    payload = {
        "created_at": date.today().isoformat(),
        "model_version": "siglip2-naflex-multisource-v3-20260802",
        "label_version": "multisource_confidence_v3",
        "deployment": "Windows PC + USB Type-C/UVC",
        "clinical_gate": "NO-GO",
        "reason": "回顾性算法锁定集未达到医生一致性门禁；缺少前瞻盲法验证",
        "files": entries,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"files": len(entries), "bytes": sum(x["bytes"] for x in entries)}))


if __name__ == "__main__":
    main()
