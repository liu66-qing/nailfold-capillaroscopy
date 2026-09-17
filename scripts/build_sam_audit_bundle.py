"""Build a self-contained private 50-image SAM audit bundle."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = ["traditional", "sam2_tiny", "sam2_base_plus", "medsam"]
    for directory in ["original", *methods]:
        (args.output / directory).mkdir(exist_ok=True)
    samples = pd.read_csv(args.experiment / "sample_manifest_50.csv")
    html = ["<html><head><meta charset='utf-8'><style>body{font-family:Arial;background:#eee}.r{display:grid;grid-template-columns:180px repeat(5,256px);gap:6px;margin:10px;padding:8px;background:#fff}img{width:256px;height:192px;object-fit:contain;background:#222}.id{font-size:12px}</style></head><body><h1>Private 50-image segmentation audit</h1>"]
    labels = ["Original", "Traditional", "SAM2 Tiny", "SAM2 Base+", "MedSAM"]
    for row in samples.to_dict("records"):
        stem = row["exam_case_id"].replace("/", "__") + "__" + Path(row["image_path"]).stem
        original_name = f"{stem}.jpg"
        shutil.copy2(args.image_root / row["image_path"], args.output / "original" / original_name)
        paths = [f"original/{original_name}"]
        for method in methods:
            source = args.experiment / method / "overlays" / original_name
            shutil.copy2(source, args.output / method / original_name)
            paths.append(f"{method}/{original_name}")
        cells = "".join(f"<div><small>{label}</small><br><img src='{path}'></div>" for label, path in zip(labels, paths))
        html.append(f"<section class='r'><div class='id'><b>{row['exam_case_id']}</b><br>fold {row['fold']}<br>{row['image_path']}</div>{cells}</section>")
    html.append("</body></html>")
    (args.output / "index.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
