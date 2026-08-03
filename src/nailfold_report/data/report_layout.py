from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ReportField:
    name: str
    center_y: int
    modality: str


# Fixed rows verified against the 680x750 report template.
REPORT_FIELDS: tuple[ReportField, ...] = (
    ReportField("clarity", 30, "image"),
    ReportField("capillary_count", 49, "image"),
    ReportField("afferent_diameter", 68, "image"),
    ReportField("efferent_diameter", 87, "image"),
    ReportField("output_input_ratio", 106, "image"),
    ReportField("apex_diameter", 125, "image"),
    ReportField("loop_length", 144, "image"),
    ReportField("crossing_ratio", 163, "image"),
    ReportField("malformation_ratio", 182, "image"),
    ReportField("flow_state", 201, "video"),
    ReportField("flow_speed_um_s", 220, "unsupported"),
    ReportField("vasomotion", 239, "video"),
    ReportField("rbc_aggregation", 258, "video"),
    ReportField("wbc_count", 277, "video"),
    ReportField("microthrombus", 296, "video"),
    ReportField("blood_color", 315, "image"),
    ReportField("exudation", 334, "image"),
    ReportField("hemorrhage", 353, "image"),
    ReportField("subpapillary_venous_plexus", 372, "image"),
    ReportField("papilla", 391, "image"),
    ReportField("sweat_duct", 410, "image"),
)


def select_reference_report(report_paths: list[Path]) -> Path | None:
    """Select a deterministic OCR reference; all copies remain available for voting."""
    if not report_paths:
        return None
    by_name = {path.name: path for path in report_paths}
    return by_name.get("rep.jpg.jpg", sorted(report_paths)[0])


def dark_pixel_count(
    image: Image.Image,
    center_y: int,
    x0: int = 220,
    x1: int = 335,
    half_height: int = 7,
    threshold: int = 128,
) -> int:
    gray = image.convert("L")
    crop = gray.crop((x0, center_y - half_height, x1, center_y + half_height + 1))
    return sum(pixel < threshold for pixel in crop.getdata())


def report_field_presence(report_path: Path, min_dark_pixels: int = 5) -> dict[str, bool]:
    with Image.open(report_path) as image:
        if image.size != (680, 750):
            return {field.name: False for field in REPORT_FIELDS}
        return {
            field.name: dark_pixel_count(image, field.center_y) >= min_dark_pixels
            for field in REPORT_FIELDS
        }
