from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "血管数据集"
GOV = ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260830"
DERIVED = ROOT / "artifacts" / "derived" / "vascular_dataset_governance_20260830"
OUT = DERIVED / "train_ready"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_detection_and_weak():
    candidates = list(csv.DictReader((GOV / "candidate_training_manifest.csv").open(encoding="utf-8-sig", newline="")))
    ids = [r["original_id"] for r in candidates if r["source_layer"] == "classification_pre" and r["candidate_status"] == "PASS_AUXILIARY_CONDITIONALLY"]
    # Include only explicitly adjudicated A/B duplicate members; HOLD and MERGE
    # groups remain excluded. Locked members are filtered below.
    decisions = {r["duplicate_group_id"]: r for r in csv.DictReader((GOV / "duplicate_group_decisions.csv").open(encoding="utf-8-sig", newline=""))}
    locked_ids = {r["original_id"] for r in csv.DictReader((GOV / "source_case_mapping.csv").open(encoding="utf-8-sig", newline="")) if r.get("source_mapping_status") == "EXACT_RECOVERED_LOCKED"}
    dup_cons = list(csv.DictReader((GOV / "duplicate_annotation_consistency.csv").open(encoding="utf-8-sig", newline="")))
    adjudicated = {"D340":"530", "D486":"522", "D494":"364"}
    for g in dup_cons:
        d = decisions.get(g["duplicate_group_id"], {})
        chosen = adjudicated.get(g["duplicate_group_id"])
        if not chosen and d.get("decision") in {"A", "B"}:
            chosen = Path(d["member_a"] if d["decision"] == "A" else d["member_b"]).stem
        if chosen and chosen not in locked_ids:
            ids.append(chosen)
    ids = sorted(set(ids), key=lambda x: int(x))
    xml_dir = DERIVED / "xml_repaired_all"
    image_dir = DATA / "分类数据集" / "扩充之前" / "images"
    det_rows, weak_rows = [], []
    wb = load_workbook(DATA / "血管分类标注" / "血管分类标注.xlsx", read_only=True, data_only=True)
    ws = wb.active
    headers = [str(x) for x in next(ws.iter_rows(values_only=True))]
    weak_by_id = {str(row[0]): row for row in ws.iter_rows(min_row=2, values_only=True)}
    class_map = {"vessel": 0, "malformed_vessel": 1, "cross_vessel": 2}
    for oid in sorted(ids, key=lambda x: int(x)):
        src_img = image_dir / f"{oid}.jpg"; src_xml = xml_dir / f"{oid}.xml"
        if not src_img.exists() or not src_xml.exists():
            continue
        # Use the final adjudicated merged XML where applicable.
        for gid, chosen in {"D340":"530", "D486":"522", "D494":"364"}.items():
            if oid == chosen and (DERIVED / "xml_adjudicated_final" / f"{gid}.xml").exists():
                src_xml = DERIVED / "xml_adjudicated_final" / f"{gid}.xml"
        root = ET.parse(src_xml).getroot(); w, h = Image.open(src_img).size
        objects = root.findall("object")
        if any((o.findtext("name") or "") not in class_map for o in objects):
            continue
        lines = []
        for obj in objects:
            bb = obj.find("bndbox"); vals = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if vals[2] <= vals[0] or vals[3] <= vals[1] or vals[0] < 0 or vals[1] < 0 or vals[2] > w or vals[3] > h:
                continue
            cx, cy = ((vals[0] + vals[2]) / 2 / w, (vals[1] + vals[3]) / 2 / h)
            bw, bh = ((vals[2] - vals[0]) / w, (vals[3] - vals[1]) / h)
            lines.append(f"{class_map[obj.findtext('name')]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        copy_file(src_img, OUT / "detection" / "images" / src_img.name)
        (OUT / "detection" / "labels").mkdir(parents=True, exist_ok=True)
        (OUT / "detection" / "labels" / f"{oid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        det_rows.append({"asset_id": oid, "image_path": str(OUT / "detection" / "images" / src_img.name), "label_path": str(OUT / "detection" / "labels" / f"{oid}.txt"), "class_map": "vessel=0;malformed_vessel=1;cross_vessel=2", "status": "PASS_AUXILIARY_CONDITIONAL", "gates": "no locked; no duplicate XML conflict; derived XML metadata; case-level split; human semantic QC pending"})
        if oid in weak_by_id:
            row = weak_by_id[oid]
            out = {"asset_id": oid, "image_path": str(OUT / "weak_field" / "images" / src_img.name), "status": "PASS_AUXILIARY_CONDITIONAL"}
            out.update({headers[i]: row[i] for i in range(1, len(headers))})
            weak_rows.append(out)
            copy_file(src_img, OUT / "weak_field" / "images" / src_img.name)
    return det_rows, weak_rows


def build_segmentation():
    rows = list(csv.DictReader((GOV / "segmentation_pair_reconstruction.csv").open(encoding="utf-8-sig", newline="")))
    out_rows = []
    for row in rows:
        if row["pair_status"] != "AUTO_CONFIRMED_PAIR":
            continue
        image = Path(row["reconstructed_image"]); mask = Path(row["reconstructed_mask"])
        if not image.exists() or not mask.exists():
            continue
        stem = Path(row["json_path"]).stem
        copy_file(image, OUT / "segmentation" / "images" / f"{stem}.jpg")
        copy_file(mask, OUT / "segmentation" / "masks" / f"{stem}.png")
        source_json = Path(row["json_path"])
        d = json.loads(source_json.read_text(encoding="utf-8"))
        d["imagePath"] = f"../images/{stem}.jpg"
        d["imageData"] = None
        d["imageWidth"], d["imageHeight"] = Image.open(image).size
        target_json = OUT / "segmentation" / "annotations" / f"{stem}.json"
        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_json.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        out_rows.append({"asset_id": stem, "image_path": str(OUT / "segmentation" / "images" / f"{stem}.jpg"), "mask_path": str(OUT / "segmentation" / "masks" / f"{stem}.png"), "annotation_path": str(target_json), "pair_evidence": row["image_evidence"] + ";" + row["mask_evidence"], "mask_polygon_iou": row["mask_polygon_iou"], "status": "PASS_AUXILIARY_CONDITIONAL", "gates": "no known locked overlap; case identity unavailable; human mask QC and split review required"})
    return out_rows


def main() -> None:
    det_rows, weak_rows = build_detection_and_weak()
    seg_rows = build_segmentation()
    write_csv(OUT / "detection_train_manifest.csv", det_rows)
    write_csv(OUT / "weak_field_train_manifest.csv", weak_rows)
    write_csv(OUT / "segmentation_train_manifest.csv", seg_rows)
    summary = {
        "created_at": "2026-08-30", "gpu_used": False, "raw_data_modified": False, "v1_modified": False,
        "detection_images": len(det_rows), "weak_field_images": len(weak_rows), "segmentation_pairs": len(seg_rows),
        "all_status": "PASS_AUXILIARY_CONDITIONAL", "locked_assets_included": 0, "augmented_assets_as_independent_samples": 0,
        "required_before_training": ["human semantic approval", "case-level split and provenance/permission approval", "independent QC", "no locked overlap"],
    }
    (OUT / "train_asset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "TRAINING_README.md").write_text("""# 派生可训练辅助资产\n\n这些目录只包含由治理证据筛出的辅助监督候选，不是临床金标准。\n\n- `detection/`：分类原图和由派生 XML 转换的 YOLO 框。\n- `weak_field/`：与 Excel image_id 对齐的图片级字段。\n- `segmentation/`：机器证实 JSON-image-mask 配对。\n\n所有条目仍需病例级拆分、临床语义认可和独立质量抽查后才能训练。增强图未作为独立样本，locked 重叠未纳入。\n""", encoding="utf-8")


if __name__ == "__main__":
    main()
