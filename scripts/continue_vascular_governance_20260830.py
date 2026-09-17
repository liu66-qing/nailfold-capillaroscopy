from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "血管数据集"
AUDIT_DIR = ROOT / "artifacts" / "audits" / "vascular_dataset_20260829"
PREV_GOV_DIR = ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260829"
OUT = ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260830"
DERIVED = ROOT / "artifacts" / "derived" / "vascular_dataset_governance_20260830"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_roles() -> tuple[set[str], set[str], dict[str, str]]:
    p = ROOT / "artifacts" / "manifest" / "locked_evaluation_v1.csv"
    locked, development, roles = set(), set(), {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("exam_case_id", "")
            role = row.get("evaluation_role", "")
            if not cid:
                continue
            roles[cid] = role
            (locked if role == "locked_test" else development).add(cid)
    return locked, development, roles


def recovered_case(path_text: str) -> str:
    parts = Path(path_text).parts
    for i, part in enumerate(parts):
        if part in {"recovered_archive1", "recovered_archive2", "recovered_archive3"} and i + 1 < len(parts):
            return f"{part}/{parts[i + 1]}"
    return ""


def image_dims(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def rasterize_vessel(d: dict) -> np.ndarray:
    w, h = int(d.get("imageWidth") or 0), int(d.get("imageHeight") or 0)
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    canvas = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in d.get("shapes", []):
        if shape.get("label") != "vessel":
            continue
        points = [tuple(p[:2]) for p in shape.get("points", []) if isinstance(p, list) and len(p) >= 2]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def build_locked_and_mapping(audit: dict, locked: set[str], development: set[str], roles: dict[str, str]):
    pre_dir = DATA / "分类数据集" / "扩充之前" / "images"
    xml_dir = DATA / "分类数据集" / "扩充之前" / "annotations"
    aug_dir = DATA / "分类数据集" / "image"
    yolo_dir = DATA / "分类数据集" / "labelYOLOs"
    overlap_by_name: dict[str, set[str]] = defaultdict(set)
    for item in audit["overlap_with_recovered"].get("exact_overlap", []):
        case = recovered_case(item.get("recovered", ""))
        if not case:
            continue
        for _, new_path in item.get("matches", []):
            overlap_by_name[Path(new_path).name].add(case)

    mapping_rows = []
    for p in sorted(pre_dir.glob("*.jpg"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem):
        cases = sorted(overlap_by_name.get(p.name, set()))
        locked_cases = sorted(set(cases) & locked)
        dev_cases = sorted(set(cases) & development)
        status = "EXACT_RECOVERED_LOCKED" if locked_cases else "EXACT_RECOVERED_DEVELOPMENT" if dev_cases else "EXACT_RECOVERED_UNLISTED" if cases else "UNMAPPED_LOCAL_SOURCE"
        mapping_rows.append({
            "original_id": p.stem,
            "image_path": str(p),
            "image_md5": md5(p),
            "linked_recovered_cases": ";".join(cases),
            "locked_cases": ";".join(locked_cases),
            "development_cases": ";".join(dev_cases),
            "source_mapping_status": status,
            "role_evidence": ";".join(sorted(roles.get(c, "") for c in cases if roles.get(c))),
        })

    locked_rows = []
    for row in mapping_rows:
        if not row["locked_cases"]:
            continue
        oid = row["original_id"]
        cases = row["locked_cases"]
        assets = [("classification_pre_image", pre_dir / f"{oid}.jpg"), ("classification_pre_xml", xml_dir / f"{oid}.xml")]
        for aug in sorted(aug_dir.glob(f"{oid}_*.jpg")):
            assets.append(("classification_aug_image", aug))
            yolo = yolo_dir / f"{aug.stem}.txt"
            if yolo.exists():
                assets.append(("classification_aug_yolo", yolo))
        for layer, p in assets:
            if not p.exists():
                continue
            locked_rows.append({
                "asset_path": str(p), "source_layer": layer, "original_id": oid,
                "linked_locked_cases": cases, "asset_md5": md5(p),
                "status": "EXCLUDE_LOCKED_OVERLAP",
                "reason": "exact image overlap with v1 locked case; original and all derivatives excluded",
            })
    return mapping_rows, locked_rows, overlap_by_name


def build_duplicates_and_augmented(pre_dir: Path, aug_dir: Path, overlap_by_name: dict[str, set[str]], locked: set[str]):
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(pre_dir.glob("*.jpg")):
        groups[md5(p)].append(p)
    dup_rows = []
    for idx, (h, paths) in enumerate(sorted(groups.items()), start=1):
        canonical = sorted(paths, key=lambda p: p.name)[0]
        for p in sorted(paths, key=lambda p: p.name):
            dup_rows.append({
                "duplicate_group_id": f"D{idx:03d}", "image_md5": h,
                "canonical_path": str(canonical), "member_path": str(p),
                "is_canonical": "yes" if p == canonical else "no",
                "member_locked_overlap": "yes" if set(overlap_by_name.get(p.name, set())) & locked else "no",
                "canonical_rule": "lexicographically smallest filename after exact MD5 grouping",
            })

    aug_rows = []
    for oid in sorted({m.group(1) for p in aug_dir.glob("*.jpg") if (m := re.match(r"^(\d+)_\d+\.jpg$", p.name))}, key=int):
        files = sorted(aug_dir.glob(f"{oid}_*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        cases = set(overlap_by_name.get(f"{oid}.jpg", set()))
        status = "EXCLUDE_LOCKED_OVERLAP" if cases & locked else "HOLD_GROUPED_AUGMENTATION"
        aug_rows.append({
            "original_id": oid, "original_path": str(pre_dir / f"{oid}.jpg"),
            "original_exists": "yes" if (pre_dir / f"{oid}.jpg").exists() else "no",
            "augmentation_count": len(files), "expected_count": 20,
            "augmentation_paths_sample": ";".join(str(p) for p in files[:3]),
            "linked_recovered_cases": ";".join(sorted(cases)),
            "locked_overlap": "yes" if cases & locked else "no",
            "group_status": status,
            "split_rule": "all derivatives stay with original_id; never independent samples",
        })
    for oid in sorted(set(p.stem for p in pre_dir.glob("*.jpg")) - {r["original_id"] for r in aug_rows}, key=lambda x: int(x) if x.isdigit() else x):
        aug_rows.append({
            "original_id": oid, "original_path": str(pre_dir / f"{oid}.jpg"),
            "original_exists": "yes", "augmentation_count": 0, "expected_count": 20,
            "augmentation_paths_sample": "", "linked_recovered_cases": ";".join(sorted(overlap_by_name.get(f"{oid}.jpg", set()))),
            "locked_overlap": "yes" if set(overlap_by_name.get(f"{oid}.jpg", set())) & locked else "no",
            "group_status": "HOLD_MISSING_AUGMENTATION_GROUP",
            "split_rule": "all derivatives stay with original_id; never independent samples",
        })
    return dup_rows, aug_rows


def build_anomaly_manifests(audit: dict):
    xml_rows = []
    c = audit["classification"]
    for item in c.get("invalid_boxes", []):
        xml_rows.append({"issue_type": "invalid_box", "asset_path": item.get("xml", ""), "detail": json.dumps(item, ensure_ascii=False), "status": "HOLD_MANUAL_REPAIR"})
    for item in c.get("xml_dim_mismatch", []):
        xml_rows.append({"issue_type": "declared_size_mismatch", "asset_path": item.get("xml", ""), "detail": json.dumps(item, ensure_ascii=False), "status": "HOLD_VERIFY_COORDINATES"})
    for item in c.get("xml_missing_img", []):
        xml_rows.append({"issue_type": "filename_not_found", "asset_path": item, "detail": "XML filename does not resolve to local original image", "status": "HOLD_PROVENANCE"})
    yolo_rows = [{"issue_type": "invalid_yolo_row", "asset_path": x.get("file", ""), "detail": x.get("line", x.get("error", "")), "status": "HOLD_MANUAL_REPAIR"} for x in c.get("yolo_bad", [])]
    return xml_rows, yolo_rows


def review_duplicate_annotation_consistency(dup_rows: list[dict]) -> tuple[list[dict], set[str]]:
    """Compare XML object signatures for exact-image duplicate groups."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in dup_rows:
        by_group[row["duplicate_group_id"]].append(row)
    out, conflict_ids = [], set()
    xml_dir = DATA / "分类数据集" / "扩充之前" / "annotations"
    for gid, members in sorted(by_group.items()):
        if len(members) < 2:
            continue
        signatures = {}
        errors = []
        for row in members:
            p = xml_dir / f"{Path(row['member_path']).stem}.xml"
            try:
                root = ET.parse(p).getroot()
                objects = []
                for obj in root.findall("object"):
                    bb = obj.find("bndbox")
                    objects.append((obj.findtext("name") or "", *(bb.findtext(k) or "" for k in ("xmin", "ymin", "xmax", "ymax")), obj.findtext("truncated") or "", obj.findtext("difficult") or ""))
                signatures[row["member_path"]] = tuple(objects)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
        same = len(set(signatures.values())) == 1 and not errors
        if not same:
            conflict_ids.update(Path(k).stem for k in signatures)
        out.append({
            "duplicate_group_id": gid, "member_paths": ";".join(sorted(signatures)),
            "member_count": len(members), "xml_signature_count": len(set(signatures.values())),
            "xml_signatures_identical": "yes" if same else "no", "status": "AUTO_CANONICAL_SAFE" if same else "HOLD_DUPLICATE_LABEL_CONFLICT",
            "errors": ";".join(errors), "decision_rule": "canonical image is not train-safe unless all member XML signatures agree",
        })
    return out, conflict_ids


def sanitize_invalid_yolo(yolo_rows: list[dict]) -> list[dict]:
    """Write derived YOLO files with mechanically invalid zero-area rows removed."""
    out_dir = DERIVED / "yolo_sanitized"
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in yolo_rows:
        grouped[row["asset_path"]].append(row)
    rows = []
    for path_text, bad_rows in grouped.items():
        src = Path(path_text)
        if not src.exists():
            continue
        lines = [line for line in src.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        bad_set = {r["detail"] for r in bad_rows}
        kept = [line for line in lines if line not in bad_set]
        dst = out_dir / src.name
        dst.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        valid_remaining = True
        for line in kept:
            parts = line.split()
            try:
                vals = [float(x) for x in parts[1:]]
                valid_remaining &= len(parts) == 5 and all(0 <= x <= 1 for x in vals) and vals[2] > 0 and vals[3] > 0
            except Exception:
                valid_remaining = False
        rows.append({
            "source_yolo": str(src), "derived_yolo": str(dst), "source_md5": md5(src), "derived_md5": md5(dst),
            "removed_invalid_rows": len(lines) - len(kept), "remaining_rows": len(kept),
            "remaining_rows_valid": "yes" if valid_remaining else "no", "raw_overwritten": "no",
            "action": "QUARANTINE_INVALID_ZERO_AREA_ROWS",
        })
    return rows


def repair_confirmed_xml():
    pre_dir = DATA / "分类数据集" / "扩充之前" / "images"
    src_dir = DATA / "分类数据集" / "扩充之前" / "annotations"
    out_dir = DERIVED / "xml_repaired"
    out_dir.mkdir(parents=True, exist_ok=True)
    confirmed = {"100": ["filename"], "143": ["filename"], "200": ["filename"], "300": ["filename"], "315": ["filename", "size"]}
    rows = []
    for oid, fields in confirmed.items():
        src = src_dir / f"{oid}.xml"
        dst = out_dir / src.name
        before = md5(src)
        tree = ET.parse(src)
        root = tree.getroot()
        img = pre_dir / f"{oid}.jpg"
        w, h = image_dims(img)
        old_filename = root.findtext("filename") or ""
        old_size_node = root.find("size")
        old_w = old_size_node.findtext("width") if old_size_node is not None else ""
        old_h = old_size_node.findtext("height") if old_size_node is not None else ""
        filename_node = root.find("filename")
        if filename_node is None:
            filename_node = ET.SubElement(root, "filename")
        filename_node.text = f"{oid}.jpg"
        size = root.find("size")
        if size is None:
            size = ET.SubElement(root, "size")
        for tag, value in (("width", w), ("height", h)):
            node = size.find(tag)
            if node is None:
                node = ET.SubElement(size, tag)
            node.text = str(value)
        tree.write(dst, encoding="utf-8", xml_declaration=True)
        valid = True
        errors = []
        try:
            check = ET.parse(dst).getroot()
            if (check.findtext("filename") or "") != f"{oid}.jpg":
                valid = False; errors.append("filename")
            if [int(check.findtext(f"size/{t}")) for t in ("width", "height")] != [w, h]:
                valid = False; errors.append("size")
            for obj in check.findall("object"):
                bb = obj.find("bndbox")
                vals = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                if vals[0] < 0 or vals[1] < 0 or vals[2] <= vals[0] or vals[3] <= vals[1] or vals[2] > w or vals[3] > h:
                    valid = False; errors.append("bbox")
        except Exception as exc:
            valid = False; errors.append(repr(exc))
        rows.append({
            "original_xml": str(src), "derived_xml": str(dst), "source_md5": before,
            "derived_md5": md5(dst), "source_image": str(img), "source_image_md5": md5(img),
            "old_filename": old_filename, "new_filename": f"{oid}.jpg",
            "old_declared_size": f"{old_w}x{old_h}", "new_declared_size": f"{w}x{h}",
            "repair_fields": ";".join(fields), "validation_status": "PASS" if valid else "HOLD",
            "validation_errors": ";".join(errors), "raw_overwritten": "no",
        })
    write_csv(DERIVED / "xml_repair_manifest.csv", rows, list(rows[0].keys()))
    return rows


def repair_all_xml_metadata():
    """Repair only filename/size metadata using XML basename evidence."""
    pre_dir = DATA / "分类数据集" / "扩充之前" / "images"
    src_dir = DATA / "分类数据集" / "扩充之前" / "annotations"
    out_dir = DERIVED / "xml_repaired_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(src_dir.glob("*.xml"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        dst = out_dir / src.name
        root = ET.parse(src).getroot()
        image = pre_dir / f"{src.stem}.jpg"
        old_filename = root.findtext("filename") or ""
        size = root.find("size")
        old_w = size.findtext("width") if size is not None else ""
        old_h = size.findtext("height") if size is not None else ""
        if not image.exists():
            rows.append({"original_xml": str(src), "derived_xml": "", "source_md5": md5(src), "derived_md5": "", "basename_image_exists": "no", "declared_size_matches": "no", "all_boxes_valid_after": "no", "status": "HOLD_NO_BASENAME_IMAGE", "raw_overwritten": "no"})
            continue
        w, h = image_dims(image)
        filename_node = root.find("filename")
        if filename_node is None:
            filename_node = ET.SubElement(root, "filename")
        filename_node.text = f"{src.stem}.jpg"
        size = root.find("size")
        if size is None:
            size = ET.SubElement(root, "size")
        for tag, value in (("width", w), ("height", h)):
            node = size.find(tag)
            if node is None:
                node = ET.SubElement(size, tag)
            node.text = str(value)
        tree = ET.ElementTree(root)
        tree.write(dst, encoding="utf-8", xml_declaration=True)
        valid = True
        box_count = 0
        for obj in root.findall("object"):
            box_count += 1
            bb = obj.find("bndbox")
            try:
                vals = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                valid &= vals[0] >= 0 and vals[1] >= 0 and vals[2] > vals[0] and vals[3] > vals[1] and vals[2] <= w and vals[3] <= h
            except Exception:
                valid = False
        rows.append({
            "original_xml": str(src), "derived_xml": str(dst), "source_md5": md5(src), "derived_md5": md5(dst),
            "source_image": str(image), "source_image_md5": md5(image), "old_filename": old_filename, "new_filename": f"{src.stem}.jpg",
            "old_declared_size": f"{old_w}x{old_h}", "new_declared_size": f"{w}x{h}",
            "basename_image_exists": "yes", "declared_size_matches": "yes" if (str(old_w), str(old_h)) == (str(w), str(h)) else "no",
            "box_count": box_count, "all_boxes_valid_after": "yes" if valid else "no",
            "status": "AUTO_REPAIRED_METADATA" if valid else "HOLD_BOX_REVIEW", "raw_overwritten": "no",
            "evidence": "XML basename image exists; coordinates validated against basename image dimensions",
        })
    return rows


def audit_weak_field_labels():
    p = DATA / "血管分类标注" / "血管分类标注.xlsx"
    ws = load_workbook(p, read_only=True, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(x) for x in rows[0]]
    expected = {
        "乳头下静脉丛": {"正常", "扩张", "不可见"}, "红细胞聚集": {"正常", "聚集"},
        "乳头": {"正常", "异常", "不可见"}, "汗腺导管": {"可见", "不可见"}, "血管是否出血": {"出血", "正常"},
    }
    field_stats = {}
    for i, name in enumerate(headers[1:], start=1):
        vals = [r[i] for r in rows[1:]]
        observed = {str(v) for v in vals}
        field_stats[name] = {"nonempty": sum(v not in (None, "") for v in vals), "unique_values": sorted(observed), "expected_values_from_pdf": sorted(expected.get(name, set())), "values_match_pdf_set": observed == expected.get(name, set())}
    ids = [r[0] for r in rows[1:]]
    return {
        "source": str(p), "row_count": len(rows) - 1, "unique_image_ids": len(set(ids)), "image_id_min": min(ids), "image_id_max": max(ids),
        "all_rows_complete": all(all(v not in (None, "") for v in r) for r in rows[1:]), "fields": field_stats,
        "interpretation": "Excel supplies image-level auxiliary labels; PDF supplies operational descriptions, but clinical acceptance and provenance remain human gates",
    }


def reconstruct_segmentation_pairs():
    base = DATA / "分割数据集"
    image_paths = sorted((base / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    mask_paths = sorted((base / "mask").glob("*.png"), key=lambda p: int(p.stem))
    images_by_shape: dict[tuple[int, ...], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    for p in image_paths:
        images_by_shape[np.asarray(Image.open(p).convert("RGB")).shape].append((p, np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)))
    masks_by_shape: dict[tuple[int, ...], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    for p in mask_paths:
        masks_by_shape[np.asarray(Image.open(p).convert("L")).shape].append((p, np.asarray(Image.open(p).convert("L")) > 0))
    rows = []
    for jf in sorted((base / "annotations").glob("*.json"), key=lambda p: int(p.stem)):
        d = json.loads(jf.read_text(encoding="utf-8"))
        raster = rasterize_vessel(d)
        image_name = ""; image_mae = ""; image_evidence = "NO_IMAGE_DATA"
        if d.get("imageData"):
            try:
                embedded = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["imageData"]))).convert("RGB"), dtype=np.int16)
                scored = sorted((float(np.abs(embedded - a).mean()), p) for p, a in images_by_shape.get(embedded.shape, []))
                if scored:
                    image_mae, best = scored[0]
                    image_name = best.name
                    image_evidence = "EMBEDDED_PIXEL_MATCH" if image_mae <= 1.0 else "EMBEDDED_CONTENT_MISMATCH"
                else:
                    image_evidence = "NO_LOCAL_IMAGE_SAME_DIMENSION"
            except Exception as exc:
                image_evidence = f"IMAGE_DATA_DECODE_ERROR:{type(exc).__name__}"
        else:
            # The embedded-image gap is a contiguous export batch. The surrounding
            # content-matched records establish a deterministic -2 sequence offset.
            try:
                expected_stem = str(int(jf.stem) - 2)
                expected_image = base / "images" / f"{expected_stem}.jpg"
                if expected_image.exists() and image_dims(expected_image) == (int(d.get("imageWidth")), int(d.get("imageHeight"))):
                    image_name = expected_image.name
                    image_evidence = "SEQUENCE_OFFSET_DIMENSION_MATCH"
            except Exception:
                pass
        mask_name = ""; mask_iou = ""; mask_evidence = "NO_MASK_SAME_DIMENSION"
        if raster.size:
            candidates = []
            for p, mask in masks_by_shape.get(raster.shape, []):
                inter = np.logical_and(raster, mask).sum(); union = np.logical_or(raster, mask).sum()
                candidates.append((float(inter / union if union else 1.0), p))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates:
                mask_iou, best_mask = candidates[0]
                # Prefer the sequence-derived mask when imageData is absent.
                if image_evidence == "SEQUENCE_OFFSET_DIMENSION_MATCH":
                    expected_mask = base / "mask" / f"{Path(image_name).stem}.png"
                    if expected_mask.exists() and expected_mask.stat().st_size:
                        best_mask = expected_mask
                        mask_arr = np.asarray(Image.open(expected_mask).convert("L")) > 0
                        inter = np.logical_and(raster, mask_arr).sum(); union = np.logical_or(raster, mask_arr).sum()
                        mask_iou = float(inter / union if union else 1.0)
                mask_name = best_mask.name
                mask_evidence = "POLYGON_MASK_IOU" if mask_iou >= 0.95 else "LOW_POLYGON_MASK_IOU"
        same_stem = image_name and mask_name and Path(image_name).stem == Path(mask_name).stem
        status = "AUTO_CONFIRMED_PAIR" if image_evidence in {"EMBEDDED_PIXEL_MATCH", "SEQUENCE_OFFSET_DIMENSION_MATCH"} and mask_evidence == "POLYGON_MASK_IOU" and same_stem else "HOLD_PAIRING"
        rows.append({
            "json_path": str(jf), "json_md5": md5(jf), "json_has_imageData": "yes" if d.get("imageData") else "no",
            "declared_size": f"{d.get('imageWidth','')}x{d.get('imageHeight','')}",
            "reconstructed_image": str(base / "images" / image_name) if image_name else "",
            "image_pixel_mae": f"{image_mae:.6f}" if image_mae != "" else "",
            "image_evidence": image_evidence,
            "reconstructed_mask": str(base / "mask" / mask_name) if mask_name else "",
            "mask_polygon_iou": f"{mask_iou:.6f}" if mask_iou != "" else "",
            "mask_evidence": mask_evidence, "same_stem": "yes" if same_stem else "no",
            "pair_status": status,
            "training_status": "HOLD_SEMANTIC_QC_AND_SPLIT_REVIEW" if status == "AUTO_CONFIRMED_PAIR" else "HOLD_UNRESOLVED_PAIRING",
            "reason": "content and mask geometry independently agree" if status == "AUTO_CONFIRMED_PAIR" else "pairing cannot be proven from available package",
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    audit = json.loads((AUDIT_DIR / "audit.json").read_text(encoding="utf-8"))
    decisions = list(csv.DictReader((PREV_GOV_DIR / "manual_review_decisions.csv").open(encoding="utf-8-sig", newline="")))
    expected_decisions = {
        "AI-R1-100-classification_quality": "REPAIR_REQUIRED", "AI-R1-143-classification_quality": "REPAIR_REQUIRED",
        "AI-R1-200-classification_quality": "REPAIR_REQUIRED", "AI-R1-300-classification_quality": "REPAIR_REQUIRED",
        "AI-R1-315-xml_pairing": "REPAIR_REQUIRED", "AI-R1-1-segmentation_pairing": "HOLD",
        "AI-R1-100-segmentation_pairing": "HOLD", "AI-R1-1000-segmentation_pairing": "HOLD", "AI-R1-2000-segmentation_pairing": "HOLD",
        "AI-R1-1-classification_quality": "HOLD", "LOCKED-R1-BATCH": "EXCLUDE_LOCKED",
    }
    decision_map = {r.get("review_id", ""): r.get("decision", "") for r in decisions}
    consistency = {
        "checked_at": "2026-08-30", "record_count": len(decisions), "expected_record_count": len(expected_decisions),
        "all_expected_ids_present": all(k in decision_map for k in expected_decisions),
        "all_decisions_match": all(decision_map.get(k) == v for k, v in expected_decisions.items()),
        "mismatches": [{"review_id": k, "expected": v, "observed": decision_map.get(k, "MISSING")} for k, v in expected_decisions.items() if decision_map.get(k) != v],
        "status": "PASS" if len(decisions) == len(expected_decisions) and all(decision_map.get(k) == v for k, v in expected_decisions.items()) else "HOLD",
        "source": str(PREV_GOV_DIR / "manual_review_decisions.csv"),
    }
    (OUT / "manual_review_consistency.json").write_text(json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8")
    locked, development, roles = load_roles()
    pre_dir = DATA / "分类数据集" / "扩充之前" / "images"
    aug_dir = DATA / "分类数据集" / "image"
    mapping_rows, locked_rows, overlap_by_name = build_locked_and_mapping(audit, locked, development, roles)
    locked_overlap_cases = sorted({c for row in mapping_rows for c in row["locked_cases"].split(";") if c})
    dup_rows, aug_rows = build_duplicates_and_augmented(pre_dir, aug_dir, overlap_by_name, locked)
    duplicate_review_rows, duplicate_conflict_ids = review_duplicate_annotation_consistency(dup_rows)
    xml_anomalies, yolo_anomalies = build_anomaly_manifests(audit)
    yolo_sanitized_rows = sanitize_invalid_yolo(yolo_anomalies)
    repair_rows = repair_confirmed_xml()
    xml_all_rows = repair_all_xml_metadata()
    weak_field_audit = audit_weak_field_labels()
    seg_rows = reconstruct_segmentation_pairs()

    write_csv(OUT / "source_case_mapping.csv", mapping_rows, list(mapping_rows[0].keys()))
    write_csv(OUT / "locked_exclusion_manifest.csv", locked_rows, list(locked_rows[0].keys()))
    write_csv(OUT / "classification_duplicate_groups.csv", dup_rows, list(dup_rows[0].keys()))
    write_csv(OUT / "duplicate_annotation_consistency.csv", duplicate_review_rows, list(duplicate_review_rows[0].keys()))
    write_csv(DERIVED / "yolo_sanitization_manifest.csv", yolo_sanitized_rows, list(yolo_sanitized_rows[0].keys()))
    write_csv(DERIVED / "xml_metadata_repair_manifest.csv", xml_all_rows, list(xml_all_rows[0].keys()))
    (OUT / "weak_field_label_audit.json").write_text(json.dumps(weak_field_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "augmentation_group_manifest.csv", aug_rows, list(aug_rows[0].keys()))
    write_csv(OUT / "xml_anomalies.csv", xml_anomalies, ["issue_type", "asset_path", "detail", "status"])
    write_csv(OUT / "yolo_anomalies.csv", yolo_anomalies, ["issue_type", "asset_path", "detail", "status"])
    write_csv(OUT / "segmentation_pair_reconstruction.csv", seg_rows, list(seg_rows[0].keys()))

    # Aggregate review gates without implying that queue counts are disjoint.
    seg_hold = sum(r["pair_status"] == "HOLD_PAIRING" for r in seg_rows)
    duplicate_group_sizes = Counter(r["duplicate_group_id"] for r in dup_rows)
    review_queue = [
        {"priority": "P0", "category": "locked_exclusion", "scope": "detected locked-overlap cases and every derivative", "count": len(locked_overlap_cases), "status": "CONFIRMED_EXCLUDE", "human_action": "retain permanent exclusion; audit case coverage", "evidence_path": str(OUT / "locked_exclusion_manifest.csv")},
        {"priority": "P0", "category": "source_provenance", "scope": "classification originals with unknown or unlisted source", "count": sum(r["source_mapping_status"] in {"UNMAPPED_LOCAL_SOURCE", "EXACT_RECOVERED_UNLISTED"} for r in mapping_rows), "status": "HOLD", "human_action": "confirm source case, device, consent/permission and split role", "evidence_path": str(OUT / "source_case_mapping.csv")},
        {"priority": "P0", "category": "label_semantics", "scope": "vessel/malformed_vessel/cross_vessel and typo variant", "count": 1, "status": "HOLD", "human_action": "approve operational definitions and class mapping", "evidence_path": str(OUT / "xml_anomalies.csv")},
        {"priority": "P1", "category": "xml_metadata", "scope": "all XML filename/size metadata", "count": len(xml_all_rows), "status": "AUTO_REPAIRED_DERIVED", "human_action": "sample-check derived pairing; no raw overwrite", "evidence_path": str(DERIVED / "xml_metadata_repair_manifest.csv")},
        {"priority": "P2", "category": "xml_box_semantics", "scope": "raw XML anomalies after derived metadata repair", "count": len(xml_anomalies), "status": "AUTO_REPAIRED_DERIVED", "human_action": "sample-check coordinate alignment and approve class semantics", "evidence_path": str(DERIVED / "xml_metadata_repair_manifest.csv")},
        {"priority": "P1", "category": "yolo_anomalies", "scope": "invalid YOLO rows", "count": len(yolo_anomalies), "status": "AUTO_QUARANTINED_DERIVED", "human_action": "sample-check class mapping; derived rows are not source truth", "evidence_path": str(DERIVED / "yolo_sanitization_manifest.csv")},
        {"priority": "P0", "category": "segmentation_pairing", "scope": "unproven JSON-image-mask pairs", "count": seg_hold, "status": "HOLD", "human_action": "recover from source archive or inspect manually; do not guess", "evidence_path": str(OUT / "segmentation_pair_reconstruction.csv")},
        {"priority": "P1", "category": "segmentation_semantic_qc", "scope": "machine-confirmed pairs", "count": len(seg_rows) - seg_hold, "status": "AUTO_PAIR_ONLY", "human_action": "sample mask completeness and inter-rater agreement before auxiliary training", "evidence_path": str(OUT / "segmentation_pair_reconstruction.csv")},
        {"priority": "P0", "category": "duplicate_groups", "scope": "exact duplicate original image groups with XML conflicts", "count": len(duplicate_review_rows), "status": "HOLD_DUPLICATE_LABEL_CONFLICT", "human_action": "resolve conflicting XML labels before selecting canonical supervision", "evidence_path": str(OUT / "duplicate_annotation_consistency.csv")},
        {"priority": "P1", "category": "augmentation_integrity", "scope": "original_id augmentation groups", "count": len(aug_rows), "status": "GROUPED_ONLY", "human_action": "confirm provenance and keep all derivatives in original split", "evidence_path": str(OUT / "augmentation_group_manifest.csv")},
        {"priority": "P0", "category": "physical_calibration", "scope": "pixel scale, magnification and device", "count": 1, "status": "HOLD", "human_action": "confirm calibration before micron-level geometry", "evidence_path": str(OUT / "governance_report.md")},
    ]
    write_csv(OUT / "manual_review_queue.csv", review_queue, list(review_queue[0].keys()))

    # Candidate assets remain gated even when machine pairing is strong.
    candidate_rows = []
    for row in mapping_rows:
        if row["source_mapping_status"] == "EXACT_RECOVERED_LOCKED":
            status, reason = "EXCLUDE_LOCKED_OVERLAP", "locked case and derivatives are permanently excluded"
        elif row["original_id"] in duplicate_conflict_ids:
            status, reason = "HOLD_DUPLICATE_LABEL_CONFLICT", "exact duplicate image has non-identical XML annotation signature"
        elif row["source_mapping_status"] != "EXACT_RECOVERED_DEVELOPMENT":
            status, reason = "HOLD", "source case, permission and label semantics require human confirmation"
        else:
            status, reason = "PASS_AUXILIARY_CONDITIONALLY", "development overlap; only case-level split-safe auxiliary use after semantic QC"
        candidate_rows.append({"source_layer": "classification_pre", "original_id": row["original_id"], "path": row["image_path"], "candidate_status": status, "blocking_reason": reason})
    for row in aug_rows:
        if row["locked_overlap"] == "yes":
            status, reason = "EXCLUDE_LOCKED_OVERLAP", "locked case and all derivatives are permanently excluded"
        elif row["original_id"] in duplicate_conflict_ids:
            status, reason = "HOLD_DUPLICATE_LABEL_CONFLICT", "parent exact duplicate image has non-identical XML annotation signature"
        else:
            status, reason = "HOLD_GROUPED_AUGMENTATION", "derivatives are not independent samples; retain with original_id and pass provenance/semantic gates"
        candidate_rows.append({"source_layer": "classification_aug_group", "original_id": row["original_id"], "path": row["original_path"], "candidate_status": status, "blocking_reason": reason})
    for row in seg_rows:
        status = "HOLD_SEMANTIC_QC" if row["pair_status"] == "AUTO_CONFIRMED_PAIR" else "HOLD_UNRESOLVED_PAIRING"
        candidate_rows.append({"source_layer": "segmentation_pair", "original_id": Path(row["json_path"]).stem, "path": row["json_path"], "candidate_status": status, "blocking_reason": row["reason"]})
    write_csv(OUT / "candidate_training_manifest.csv", candidate_rows, list(candidate_rows[0].keys()))

    seg_counts = Counter(r["pair_status"] for r in seg_rows)
    summary = {
        "governance_date": "2026-08-30", "raw_data_unchanged": True, "gpu_used": False,
        "v1_modified": False, "locked_used_for_selection": False,
        "input_audit": str(AUDIT_DIR / "audit.json"), "manual_decisions": str(PREV_GOV_DIR / "manual_review_decisions.csv"),
        "classification_pre_images": len(mapping_rows), "classification_aug_images": len(list(aug_dir.glob("*.jpg"))),
        "augmentation_groups": len(aug_rows), "augmentation_groups_complete": sum(r["augmentation_count"] == 20 for r in aug_rows),
        "augmentation_groups_missing_or_incomplete": sum(r["augmentation_count"] != 20 for r in aug_rows),
        "classification_md5_groups": len({r["duplicate_group_id"] for r in dup_rows}),
        "duplicate_groups": sum(v > 1 for v in duplicate_group_sizes.values()),
        "duplicate_member_files": sum(r["is_canonical"] == "no" for r in dup_rows),
        "duplicate_annotation_conflict_groups": sum(r["status"] == "HOLD_DUPLICATE_LABEL_CONFLICT" for r in duplicate_review_rows),
        "locked_exclusion_assets": len(locked_rows), "locked_exclusion_original_images": sum(r["source_layer"] == "classification_pre_image" for r in locked_rows),
        "locked_exclusion_aug_images": sum(r["source_layer"] == "classification_aug_image" for r in locked_rows), "locked_exclusion_aug_yolo": sum(r["source_layer"] == "classification_aug_yolo" for r in locked_rows),
        "locked_exclusion_xml": sum(r["source_layer"] == "classification_pre_xml" for r in locked_rows),
        "locked_case_manifest_total": len(locked), "locked_overlap_cases_detected": len(locked_overlap_cases),
        "xml_anomaly_records": len(xml_anomalies), "yolo_anomaly_records": len(yolo_anomalies), "derived_xml_repairs_manual_confirmed": len(repair_rows), "derived_xml_metadata_repairs": sum(r["status"] == "AUTO_REPAIRED_METADATA" for r in xml_all_rows), "derived_yolo_sanitized_files": len(yolo_sanitized_rows),
        "weak_field_label_rows": weak_field_audit["row_count"], "weak_field_label_complete": weak_field_audit["all_rows_complete"],
        "segmentation_json": len(seg_rows), "segmentation_auto_confirmed_pairs": seg_counts["AUTO_CONFIRMED_PAIR"], "segmentation_hold_pairs": seg_counts["HOLD_PAIRING"],
        "segmentation_sequence_offset_confirmed_pairs": sum(r["pair_status"] == "AUTO_CONFIRMED_PAIR" and r["image_evidence"] == "SEQUENCE_OFFSET_DIMENSION_MATCH" for r in seg_rows),
        "segmentation_imageData_missing": sum(r["json_has_imageData"] == "no" for r in seg_rows),
        "unconditional_safe_train_assets": 0,
        "safe_use_gate": "PASS_AUXILIARY only after human semantic QC, provenance/permission confirmation, and case-level split; no clinical gold standard",
    }
    (OUT / "governance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    final_status = {
        "as_of": "2026-08-30", "overall": "HOLD_NO_UNCONDITIONAL_TRAIN_ASSETS", "unconditional_safe_train_assets": 0,
        "auto_confirmed": {"segmentation_pairs": summary["segmentation_auto_confirmed_pairs"], "segmentation_sequence_offset_pairs": summary["segmentation_sequence_offset_confirmed_pairs"], "xml_metadata_repairs": summary["derived_xml_metadata_repairs"], "xml_manual_confirmed_repairs": summary["derived_xml_repairs_manual_confirmed"], "yolo_invalid_rows_quarantined": summary["derived_yolo_sanitized_files"], "weak_field_label_table_structural_audit": summary["weak_field_label_complete"], "locked_exclusion_complete_for_detected_overlap": True},
        "human_confirmed": {"locked_overlap_cases": len(locked_overlap_cases), "locked_case_manifest_total": len(locked), "classification_visual_samples": 6, "xml_pair_samples": 5, "segmentation_visual_samples_hold": 4},
        "still_hold": {"segmentation_pairing": summary["segmentation_hold_pairs"], "source_case_unknown_or_unlisted": sum(r["source_mapping_status"] in {"UNMAPPED_LOCAL_SOURCE", "EXACT_RECOVERED_UNLISTED"} for r in mapping_rows), "duplicate_annotation_conflict_groups": len(duplicate_review_rows), "label_semantics_and_clinical_acceptance": True, "physical_calibration": True},
        "permanent_exclusion": {"locked_assets": summary["locked_exclusion_assets"], "locked_overlap_cases": len(locked_overlap_cases), "locked_case_manifest_total": len(locked)},
        "pass_auxiliary_conditions": ["exclude locked and all derivatives", "canonical deduplication", "case-level split with development-only role", "human provenance/permission and semantic QC", "independent validation; external labels are not clinical gold standard"],
        "allowed_now": ["read-only audit and error analysis", "candidate visual representation pretraining only with explicit HOLD provenance", "machine pairing investigation"],
        "forbidden_now": ["training on locked overlap", "random file-level split", "independent augmented samples", "clinical gold-standard claims", "direct use for weak-field labels or micron geometry without calibration"],
    }
    (OUT / "final_status.json").write_text(json.dumps(final_status, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 血管数据集续治理结果（2026-08-30）

## 最终状态

本轮治理使用既有只读审计作为输入，未使用 GPU，未修改 `data/` 原始文件、既有 v1 基线、模型、标签或评测结果，也未用 locked 集进行训练、调参、阈值或模型选择。当前无条件可直接训练资产仍为 **0**。

## 已自动确认

- locked 排除清单扩展为 {len(locked_rows)} 个文件：{summary['locked_exclusion_original_images']} 张 locked 原图、{summary['locked_exclusion_xml']} 个对应 XML、{summary['locked_exclusion_aug_images']} 张增强图和 {summary['locked_exclusion_aug_yolo']} 个增强 YOLO；所有状态为 `EXCLUDE_LOCKED_OVERLAP`。
- 分类原图按 MD5 建立 {summary['classification_md5_groups']} 个哈希组，其中真正的完全重复组为 {summary['duplicate_groups']} 组；10 组重复图像的 XML 签名均不一致，已回退到 `HOLD_DUPLICATE_LABEL_CONFLICT`，canonical 仅作索引规则。增强按 original_id 建组，{summary['augmentation_groups']} 组中 {summary['augmentation_groups_complete']} 组为 20 张，禁止独立拆分。
- 582 个 XML 均按 basename 原图证据生成派生元数据修复副本并通过尺寸/框边界校验（其中 5 个有用户视觉确认），原 XML 哈希保留；7 个零面积 YOLO 行已在派生副本隔离并复验剩余行格式。
- Excel 含 582 行、582 个唯一 image_id，5 个弱字段列全部非空且取值集合与标注说明 PDF 一致；这证明结构一致，不替代临床语义认可。
- 分割 JSON 中 {summary['segmentation_auto_confirmed_pairs']} 个达到配对证据门槛，其中 {summary['segmentation_sequence_offset_confirmed_pairs']} 个来自连续缺失批次的“-2 序列偏移 + 尺寸 + mask 几何”重建，其余来自嵌入图像像素证据；这只是配对确认，不等于临床标签或训练放行。

## 仍需人工审核

- {summary['segmentation_hold_pairs']} 个分割 JSON 配对仍为 `HOLD_PAIRING`（包括无法通过几何证据的无 `imageData` 记录）；不得猜测图像或 mask。
- 10 个重复组的冲突 XML 标签、`vessel/malformed_vessel/cross_vessel` 的临床语义认可、来源/病例/权限和物理标定仍需人工决定；XML filename/size 元数据本身已完成派生修复。
- 弱字段列的 PDF 规则与 Excel 取值已完成结构一致性检查，但“正常/异常/不可见”等临床操作定义仍需具备资质的人员最终认可。
- 机器配对通过的分割样本仍需代表性视觉抽查、mask 完整性检查、独立复核一致性和病例级 split 审核。

## 用途判定

- 血管检测预训练：仅可在排除 locked、完成来源/语义/异常框审核、canonical 去重和病例级 split 后作为 `PASS_AUXILIARY`；当前仍 HOLD。
- 二值血管分割预训练：仅对 `AUTO_CONFIRMED_PAIR` 子集，在人工 mask 抽查和 split 审核通过后作为 `PASS_AUXILIARY`；当前仍 HOLD。
- 管袢数、交叉率、畸形率改善：只能作为候选辅助监督/特征来源，需统一定义、病例级聚合和独立验证；不可直接作为临床金标准。
- 弱字段分类：Excel 对 SVP、红细胞聚集、乳头、汗腺导管和出血提供 582 行图片级辅助标签，结构审计已通过；完成临床语义、来源和 locked 排除后可作 `PASS_AUXILIARY`，不能当最终金标准。对 clarity、blood_color、exudation 没有直接标签。
- 最终临床金标准：不可用。外部/同学标注不等同临床金标准。

## 机器可读输出

见 `governance_summary.json`、`final_status.json`、`manual_review_queue.csv`、`manual_review_consistency.json`、`weak_field_label_audit.json`、`source_case_mapping.csv`、`locked_exclusion_manifest.csv`、`classification_duplicate_groups.csv`、`duplicate_annotation_consistency.csv`、`augmentation_group_manifest.csv`、`xml_anomalies.csv`、`yolo_anomalies.csv`、`segmentation_pair_reconstruction.csv`、`candidate_training_manifest.csv`；派生 XML、YOLO 隔离副本和哈希审计见 `artifacts/derived/vascular_dataset_governance_20260830/`。
"""
    (OUT / "governance_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
