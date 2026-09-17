from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def select_images(folds: Path, image_root: Path, count: int) -> list[tuple[str, Path]]:
    table = pd.read_csv(folds)
    table = table[(table["split"] == "train") & table["report_label_available"].astype(bool)]
    selected: list[tuple[str, Path]] = []
    # One image per deterministic case avoids a single case dominating the audit.
    for row in table.sort_values("exam_case_id").itertuples(index=False):
        case_dir = image_root / row.exam_case_id
        files = sorted(case_dir.glob("CAPorg*.jpg"))
        if files:
            selected.append((row.exam_case_id, files[len(files) // 2]))
        if len(selected) >= count:
            break
    return selected


def choose_mask(candidates: list[dict], image: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    rgb = image.astype(np.float32)
    red_excess = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
    global_red = float(red_excess.mean())
    choices = []
    for candidate in candidates:
        mask = candidate["segmentation"].astype(bool)
        coverage = float(mask.mean())
        if not 0.0002 <= coverage <= 0.20:
            continue
        mean_red = float(red_excess[mask].mean()) if mask.any() else global_red
        quality = float(candidate.get("predicted_iou", 0.0)) * float(candidate.get("stability_score", 0.0))
        score = quality * (1.0 + max(0.0, mean_red - global_red) / 50.0)
        choices.append((score, mask, candidate))
    if not choices:
        return np.zeros((h, w), dtype=bool), {"reason": "no_candidate_after_filters"}
    _, mask, candidate = max(choices, key=lambda item: item[0])
    return mask, candidate


def summarize(mask: np.ndarray) -> dict:
    binary = mask.astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([])
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {
        "coverage": float(mask.mean()),
        "components_ge_16px": int((component_areas >= 16).sum()),
        "largest_component_px": int(component_areas.max()) if len(component_areas) else 0,
        "perimeter_px": int(sum(len(contour) for contour in contours)),
    }


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rendered = image.copy()
    rendered[mask] = (0.35 * rendered[mask] + 0.65 * np.array([0, 255, 128])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rendered, contours, -1, (255, 255, 0), 1)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    (args.output / "overlays").mkdir(exist_ok=True)
    selected = select_images(args.folds, args.image_root, args.count)
    with (args.output / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["exam_case_id", "image_path"])
        writer.writerows((case_id, str(path)) for case_id, path in selected)

    torch.cuda.reset_peak_memory_stats()
    start_load = time.perf_counter()
    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(args.checkpoint), device="cuda")
    model.eval()
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=16, points_per_batch=64, pred_iou_thresh=0.70,
        stability_score_thresh=0.80, min_mask_region_area=16, output_mode="binary_mask",
    )
    load_seconds = time.perf_counter() - start_load
    rows: list[dict] = []
    for case_id, path in selected:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        with torch.inference_mode():
            candidates = generator.generate(image)
        elapsed = time.perf_counter() - started
        mask, picked = choose_mask(candidates, image)
        stats = summarize(mask)
        stem = f"{case_id.replace('/', '__')}__{path.stem}"
        np.save(args.output / "masks" / f"{stem}.npy", mask)
        cv2.imwrite(str(args.output / "overlays" / f"{stem}.jpg"), cv2.cvtColor(overlay(image, mask), cv2.COLOR_RGB2BGR))
        rows.append({
            "exam_case_id": case_id, "image_path": str(path), "candidate_count": len(candidates),
            "selected_predicted_iou": float(picked.get("predicted_iou", 0.0)),
            "selected_stability": float(picked.get("stability_score", 0.0)),
            "seconds": elapsed, **stats,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    metadata = {
        "method": "SAM2 automatic grid candidates; selected by model quality, stability, 0.02%-20% coverage, and red-excess contrast",
        "prompt_strategy": "16x16 automatic point grid; no report image or report text is supplied",
        "load_seconds": load_seconds,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "sample_count": len(rows),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
