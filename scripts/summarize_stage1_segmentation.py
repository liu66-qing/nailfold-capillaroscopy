"""Create audit triage and pixel-domain geometry candidates for Stage 1 masks."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

METHODS = ("traditional", "sam2_tiny", "sam2_base_plus", "medsam")

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Morphological skeleton; avoids adding an unreviewed dependency."""
    work = (binary.astype(np.uint8) * 255).copy()
    skel = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skel |= cv2.subtract(work, opened)
        work = cv2.erode(work, element)
    return skel.astype(bool)

def mask_geometry(mask: np.ndarray) -> dict:
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=np.int64)
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    maxima = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-5)
    widths = 2 * dt[maxima]
    skel = skeletonize(mask)
    nbr = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.uint8)) - skel.astype(np.int16)
    return {
        "coverage": float(mask.mean()), "area_px": int(mask.sum()),
        "component_count_ge_16px": int((areas >= 16).sum()),
        "largest_component_px": int(areas.max()) if len(areas) else 0,
        "component_area_median_px": float(np.median(areas)) if len(areas) else 0.,
        "perimeter_px": float(sum(cv2.arcLength(c, True) for c in cont)),
        "width_px_median": float(np.median(widths)) if len(widths) else 0.,
        "width_px_p90": float(np.quantile(widths, .9)) if len(widths) else 0.,
        "skeleton_length_px": int(skel.sum()),
        "skeleton_endpoints": int((skel & (nbr == 1)).sum()),
        "skeleton_branchpoints": int((skel & (nbr >= 3)).sum()),
    }

def triage(row: dict) -> list[str]:
    flags = []
    if row["coverage"] > .20 or row["largest_component_px"] > row["area_px"] * .80 and row["coverage"] > .10:
        flags.append("background_connected_or_overfill")
    if row["width_px_median"] > 12:
        flags.append("overwide_mask")
    if row["component_count_ge_16px"] > 40:
        flags.append("fragmented_or_texture_response")
    if row["coverage"] < .015:
        flags.append("low_coverage_or_missed_vessels")
    return flags or ["no_automatic_red_flag"]

def main() -> None:
    root = Path("/root/autodl-tmp/nailfold")
    out = root / "artifacts/experiments/sam_stage2"
    samples = pd.read_csv(out / "sample_manifest_50.csv")
    geom_dir = out / "mask_geometry_candidates"; geom_dir.mkdir(exist_ok=True)
    all_rows = []
    for method in METHODS:
        rows = []
        for s in samples.to_dict("records"):
            stem = s["exam_case_id"].replace("/", "__") + "__" + Path(s["image_path"]).stem
            mask = np.load(out / method / "masks" / f"{stem}.npy")
            row = {**s, "method": method, **mask_geometry(mask)}
            row["auto_quality_flags"] = "|".join(triage(row))
            row["loop_completeness_status"] = "manual_review_required_no_reference_mask"
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(geom_dir / f"{method}_geometry_candidates.csv", index=False)
        all_rows.extend(rows)
    audit = pd.DataFrame(all_rows)
    audit.to_csv(out / "quality_audit_auto.csv", index=False)
    summary = {}
    for method, group in audit.groupby("method"):
        exploded = group.assign(flag=group.auto_quality_flags.str.split("|")).explode("flag")
        summary[method] = {
            "images": int(len(group)),
            "flag_counts": {str(k): int(v) for k, v in exploded.flag.value_counts().items()},
            "review_priority_cases": group[group.auto_quality_flags.ne("no_automatic_red_flag")][["exam_case_id", "image_path", "auto_quality_flags"]].to_dict("records"),
        }
    # JSON keeps the triage compact; the CSV retains one record per method/image.
    (out / "quality_audit_summary.json").write_text(json.dumps({
        "purpose": "automatic triage only; visual review remains required and this is not segmentation ground truth",
        "methods": summary,
        "field_limitations": {
            "candidate_only": ["pixel width distribution", "pixel skeleton length", "component/branch/end-point counts", "coverage", "perimeter"],
            "not_reliable_yet": ["medical capillary count", "afferent/efferent/apex diameter in um", "loop length in um", "crossing ratio", "malformation ratio"],
            "reason": "no calibrated scale, no validated loop-instance association, and no pixel-level reference masks; raw skeleton branch counts are retained as exploratory pixels-only features, not quality flags"
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
