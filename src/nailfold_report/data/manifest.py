from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageStat

from .report_layout import REPORT_FIELDS, report_field_presence, select_reference_report


ARCHIVE_PATTERN = re.compile(r"^recovered_archive\d+$")
CAP_PATTERN = re.compile(r"^CAPorg\d+\.jpg$", re.IGNORECASE)
REPORT_PATTERN = re.compile(r"^rep.*\.jpg\.jpg$", re.IGNORECASE)


@dataclass
class FileRecord:
    exam_case_id: str
    archive_id: str
    numeric_directory: int
    role: str
    path: str
    filename: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    gray_mean: float | None = None
    gray_std: float | None = None
    is_black_placeholder: bool = False


@dataclass
class CaseRecord:
    exam_case_id: str
    patient_id: None
    archive_id: str
    numeric_directory: int
    cap_count_raw: int
    cap_count_usable: int
    report_copy_count: int
    video_count: int
    has_rep_rtf: bool
    has_prn_rtf: bool
    reference_report: str | None
    report_present_field_count: int
    report_label_available: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_stats(path: Path) -> tuple[int, int, float, float, bool]:
    with Image.open(path) as image:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        std = float(stat.stddev[0])
        return image.width, image.height, mean, std, mean <= 1.0 and std <= 1.0


def iter_exam_directories(data_root: Path) -> Iterable[tuple[Path, Path]]:
    for archive in sorted(data_root.iterdir()):
        if not archive.is_dir() or not ARCHIVE_PATTERN.match(archive.name):
            continue
        for case_dir in sorted(
            (item for item in archive.iterdir() if item.is_dir() and item.name.isdigit()),
            key=lambda item: int(item.name),
        ):
            yield archive, case_dir


def build_manifest(data_root: Path) -> tuple[list[CaseRecord], list[FileRecord]]:
    case_records: list[CaseRecord] = []
    file_records: list[FileRecord] = []

    for archive, case_dir in iter_exam_directories(data_root):
        exam_case_id = f"{archive.name}/{case_dir.name}"
        files = [item for item in case_dir.iterdir() if item.is_file()]
        caps = sorted((item for item in files if CAP_PATTERN.match(item.name)), key=lambda p: p.name)
        reports = sorted((item for item in files if REPORT_PATTERN.match(item.name)), key=lambda p: p.name)
        videos = sorted((item for item in files if item.suffix.lower() == ".avi"), key=lambda p: p.name)
        reference = select_reference_report(reports)
        presence = report_field_presence(reference) if reference else {}

        usable_caps = 0
        for path in caps:
            width, height, mean, std, is_black = image_stats(path)
            usable_caps += int(not is_black)
            file_records.append(
                FileRecord(
                    exam_case_id=exam_case_id,
                    archive_id=archive.name,
                    numeric_directory=int(case_dir.name),
                    role="cap_image",
                    path=str(path.relative_to(data_root)).replace("\\", "/"),
                    filename=path.name,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    width=width,
                    height=height,
                    gray_mean=round(mean, 6),
                    gray_std=round(std, 6),
                    is_black_placeholder=is_black,
                )
            )

        for role, paths in (("report_image", reports), ("video", videos)):
            for path in paths:
                file_records.append(
                    FileRecord(
                        exam_case_id=exam_case_id,
                        archive_id=archive.name,
                        numeric_directory=int(case_dir.name),
                        role=role,
                        path=str(path.relative_to(data_root)).replace("\\", "/"),
                        filename=path.name,
                        size_bytes=path.stat().st_size,
                        sha256=sha256_file(path) if role == "report_image" else "",
                    )
                )

        case_records.append(
            CaseRecord(
                exam_case_id=exam_case_id,
                patient_id=None,
                archive_id=archive.name,
                numeric_directory=int(case_dir.name),
                cap_count_raw=len(caps),
                cap_count_usable=usable_caps,
                report_copy_count=len(reports),
                video_count=len(videos),
                has_rep_rtf=(case_dir / "rep_rch1.rtf").exists(),
                has_prn_rtf=(case_dir / "prn_rch1.rtf").exists(),
                reference_report=(
                    str(reference.relative_to(data_root)).replace("\\", "/") if reference else None
                ),
                report_present_field_count=sum(presence.values()),
                report_label_available=any(presence.values()),
            )
        )

    return case_records, file_records


def write_manifest(
    case_records: list[CaseRecord],
    file_records: list[FileRecord],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, records: list[object]) -> None:
        rows = [asdict(record) for record in records]
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "cases.csv", case_records)
    write_csv(output_dir / "files.csv", file_records)

    summary = {
        "exam_directories": len(case_records),
        "cap_files": sum(record.role == "cap_image" for record in file_records),
        "black_cap_files": sum(
            record.role == "cap_image" and record.is_black_placeholder for record in file_records
        ),
        "report_files": sum(record.role == "report_image" for record in file_records),
        "video_files": sum(record.role == "video" for record in file_records),
        "image_report_directories": sum(
            case.cap_count_raw > 0 and case.report_copy_count > 0 for case in case_records
        ),
        "image_report_label_directories": sum(
            case.cap_count_usable > 0 and case.report_label_available for case in case_records
        ),
        "video_label_directories": sum(
            case.video_count > 0 and case.report_label_available for case in case_records
        ),
        "note": "patient_id is unknown; exam_case_id is an archive/numeric-directory identifier",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
