from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "血管数据集"
AUDIT_DIR = ROOT / "artifacts" / "audits" / "vascular_dataset_20260829"
OUT = ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260829"
OUT.mkdir(parents=True, exist_ok=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_manifest() -> tuple[set[str], set[str], dict[str, str]]:
    p = ROOT / "artifacts" / "manifest" / "locked_evaluation_v1.csv"
    locked, development, roles = set(), set(), {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row.get("exam_case_id", "")
            role = row.get("evaluation_role", "")
            roles[cid] = role
            (locked if role == "locked_test" else development).add(cid)
    return locked, development, roles


def case_from_recovered(path_text: str) -> str | None:
    p = Path(path_text)
    parts = p.parts
    for i, part in enumerate(parts):
        if part in {"recovered_archive1", "recovered_archive2", "recovered_archive3"} and i + 1 < len(parts):
            return f"{part}/{parts[i + 1]}"
    return None


def main() -> None:
    audit = json.loads((AUDIT_DIR / "audit.json").read_text(encoding="utf-8"))
    locked, development, roles = load_manifest()

    pre_dir = DATA / "分类数据集" / "扩充之前" / "images"
    pre_xml_dir = DATA / "分类数据集" / "扩充之前" / "annotations"
    aug_dir = DATA / "分类数据集" / "image"
    aug_yolo_dir = DATA / "分类数据集" / "labelYOLOs"
    seg_img_dir = DATA / "分割数据集" / "images"
    seg_mask_dir = DATA / "分割数据集" / "mask"
    seg_json_dir = DATA / "分割数据集" / "annotations"

    # Exact duplicate and group metadata for classification originals.
    pre_files = sorted(pre_dir.glob("*.jpg"))
    pre_hash_groups: dict[str, list[Path]] = defaultdict(list)
    for p in pre_files:
        pre_hash_groups[md5(p)].append(p)
    canonical_pre: set[Path] = set()
    duplicate_pre: set[Path] = set()
    for paths in pre_hash_groups.values():
        canonical_pre.add(sorted(paths)[0])
        duplicate_pre.update(sorted(paths)[1:])

    aug_groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(aug_dir.glob("*.jpg")):
        m = re.match(r"^(\d+)_([0-9]+)\.jpg$", p.name)
        aug_groups[m.group(1) if m else p.stem].append(p)

    # Map exact overlap files to recovered cases. The audit already performed the
    # expensive hash comparison; we only consume its auditable result here.
    overlap_by_new: dict[str, set[str]] = defaultdict(set)
    for item in audit["overlap_with_recovered"].get("exact_overlap", []):
        case_id = case_from_recovered(item.get("recovered", ""))
        if not case_id:
            continue
        for _, new_path in item.get("matches", []):
            overlap_by_new[str(Path(new_path))].add(case_id)

    # Normalize paths for Windows/JSON path differences.
    overlap_by_name: dict[str, set[str]] = defaultdict(set)
    for ptext, cases in overlap_by_new.items():
        overlap_by_name[Path(ptext).name].update(cases)
    locked_overlap_cases = {c for cs in overlap_by_new.values() for c in cs if c in locked}
    dev_overlap_cases = {c for cs in overlap_by_new.values() for c in cs if c in development}
    unlisted_overlap_cases = {c for cs in overlap_by_new.values() for c in cs if c not in roles}

    manifest_rows: list[dict] = []

    def add_asset(path: Path, source: str, original_id: str = "", aug_id: str = "") -> None:
        name = path.name
        cases = set(overlap_by_name.get(name, set())) if source == "classification_pre" else set()
        # For an augmented group, inherit exclusion from its original image group.
        if source == "classification_aug" and original_id:
            pre = pre_dir / f"{original_id}.jpg"
            cases = set(overlap_by_name.get(pre.name, set()))
        lock = sorted(cases & locked)
        status = "HOLD"
        reason = "requires provenance, semantic QC and split review"
        if lock:
            status = "EXCLUDE_LOCKED_OVERLAP"
            reason = "exact overlap with v1 locked case; all derivatives excluded"
        elif source == "classification_pre" and path in duplicate_pre:
            status = "EXCLUDE_EXACT_DUPLICATE"
            reason = "duplicate of canonical original image"
        elif source == "segmentation":
            status = "HOLD_SEGMENTATION_PAIRING"
            reason = "JSON-image-mask mapping and source identity unresolved"
        elif source == "classification_aug":
            reason = "augmentation inherits original-group status; never split independently"
        elif cases:
            reason = "exact overlap with known development case; fold-safe auxiliary use only"
        elif source == "classification_pre":
            reason = "source case not recovered; external provenance not yet confirmed"
        manifest_rows.append({
            "asset_path": str(path),
            "source_layer": source,
            "original_id": original_id,
            "augmentation_id": aug_id,
            "linked_recovered_cases": ";".join(sorted(cases)),
            "locked_overlap": "yes" if lock else "no",
            "development_overlap": "yes" if cases & development else "no",
            "status": status,
            "reason": reason,
        })

    for p in pre_files:
        add_asset(p, "classification_pre", p.stem)
    for p in sorted(aug_dir.glob("*.jpg")):
        m = re.match(r"^(\d+)_([0-9]+)\.jpg$", p.name)
        add_asset(p, "classification_aug", m.group(1) if m else p.stem, m.group(2) if m else "")
    for p in sorted(seg_img_dir.glob("*")):
        add_asset(p, "segmentation", p.stem)

    fields = [
        "asset_path", "source_layer", "original_id", "augmentation_id",
        "linked_recovered_cases", "locked_overlap", "development_overlap",
        "status", "reason",
    ]
    write_csv(OUT / "governance_manifest.csv", manifest_rows, fields)

    # Candidate manifest is intentionally conservative: no file is unconditionally
    # safe until human semantic/provenance review is complete.
    candidate_rows = [
        {
            "source_layer": "classification_pre",
            "original_id": p.stem,
            "path": str(p),
            "candidate_status": "HOLD",
            "blocking_reason": "provenance and annotation semantics not yet human-approved",
        }
        for p in pre_files if p not in duplicate_pre and not (overlap_by_name.get(p.name, set()) & locked)
    ]
    write_csv(
        OUT / "candidate_training_manifest.csv",
        candidate_rows,
        ["source_layer", "original_id", "path", "candidate_status", "blocking_reason"],
    )

    # Summarized human-review queue. Mechanical checks are recorded as queues, but
    # the queue explicitly states the human decision that remains necessary.
    c = audit["classification"]
    s = audit["segmentation"]
    queue: list[dict] = []

    def q(priority: str, category: str, scope: str, count: int, examples: str, decision: str, automated: str) -> None:
        queue.append({
            "priority": priority, "category": category, "scope": scope,
            "count": count, "example_paths": examples,
            "human_decision_required": decision, "automated_check_status": automated,
        })

    q("P0", "locked_overlap", "recovered cases", len(locked_overlap_cases), 
      ";".join(sorted(locked_overlap_cases)[:20]),
      "confirm exclusion of each case and every augmentation derivative", "detected; exclusion list generated")
    q("P0", "unlisted_overlap", "recovered cases", len(unlisted_overlap_cases),
      ";".join(sorted(unlisted_overlap_cases)),
      "resolve provenance or quarantine", "detected; quarantine required")
    q("P0", "source_provenance", "classification originals not exactly mapped", 
      max(0, c["pre_images"] - len(overlap_by_name)), "classification_pre/images/*.jpg",
      "confirm source case, acquisition device and permission to use", "not inferable from files")
    q("P0", "label_semantics", "classification XML/Excel labels", 1,
      "classification_pre/annotations/*.xml; 血管分类标注.xlsx",
      "approve operational definitions for cross_vessel, malformed_vessel, hemorrhage, SVP, papilla and sweat duct",
      "format parsed; clinical meaning not decidable")
    q("P0", "annotation_quality", "classification XML invalid boxes", len(c["invalid_boxes"]),
      ";".join(x.get("xml", "") for x in c["invalid_boxes"][:10]),
      "repair/delete each box after visual review", "mechanically detected")
    q("P0", "annotation_quality", "YOLO invalid rows", len(c["yolo_bad"]),
      ";".join(x.get("file", "") for x in c["yolo_bad"][:10]),
      "repair/delete each row and verify class mapping", "mechanically detected")
    q("P1", "coordinate_mapping", "XML size mismatch", len(c["xml_dim_mismatch"]),
      ";".join(x.get("xml", "") for x in c["xml_dim_mismatch"][:10]),
      "verify source image and decide whether coordinate rescaling is valid", "mechanically detected")
    q("P1", "filename_mapping", "XML filename not found", len(c["xml_missing_img"]),
      ";".join(c["xml_missing_img"][:10]),
      "recover original image mapping from source archive or quarantine", "mechanically detected")
    q("P1", "duplicate_review", "exact duplicate original groups", c["exact_duplicate_image_hash_groups"],
      "124.jpg=330.jpg; 337.jpg=57.jpg; see audit.json",
      "choose one canonical file and confirm labels are identical", "hash groups detected")
    q("P1", "augmentation_integrity", "augmentation groups", c["augmentation_group_count"],
      "分类数据集/image/<original_id>_<aug_id>.jpg",
      "confirm augmentation provenance; split by original_id before training", "groups counted")
    q("P1", "augmentation_missing_group", "originals without augmentation", 2,
      "分类数据集/扩充之前/images/418.jpg; 分类数据集/扩充之前/images/423.jpg",
      "confirm whether augmentation was intentionally skipped or assets are missing", "detected by group-to-original join")
    q("P0", "segmentation_pairing", "JSON without literal image path", s["missing_image_count"],
      "分割数据集/annotations/*.json",
      "rebuild JSON-image-mask pairing from imageData/original zip", "mechanically detected; all paths unresolved")
    q("P0", "segmentation_pairing", "JSON-image/mask dimension drift", s["stem_dimension_mismatch_count"],
      "分割数据集/annotations/*.json",
      "approve corrected pairing and inspect representative masks", "mechanically detected")
    q("P1", "segmentation_missing_asset", "JSON without imageData", s["json_stats"].get("no_imageData", 0),
      "分割数据集/annotations/*.json",
      "recover source image or quarantine annotation", "mechanically detected")
    q("P1", "segmentation_extra_json", "JSON without image/mask stem", s["json_without_image_stem_count"],
      "3160.json; 3161.json",
      "recover or exclude", "mechanically detected")
    q("P1", "mask_semantics", "background polygon and binary mask meaning", 1,
      "分割数据集/annotations + mask",
      "human-confirm background semantics and foreground completeness on sampled images", "not decidable from format")
    q("P0", "gold_standard", "sampled semantic QC", 200,
      "100 classification originals + 100 segmentation image/mask pairs",
      "approve representative labels, visibility, completeness and ambiguity; record inter-rater agreement", "requires human review")
    q("P0", "physical_calibration", "scale and magnification", 1,
      "classification image dimensions; segmentation crop dimensions",
      "confirm device, pixel scale and whether micron measurements are valid", "not present in dataset package")

    write_csv(
        OUT / "manual_review_queue.csv",
        queue,
        ["priority", "category", "scope", "count", "example_paths", "human_decision_required", "automated_check_status"],
    )

    # Keep local-file counts separate from recovered-side overlap records.  The
    # audit count is the number of recovered files; one recovered file can match
    # more than one local image (for example duplicated originals).
    local_locked_pre = sum(
        1 for row in manifest_rows
        if row["source_layer"] == "classification_pre" and row["locked_overlap"] == "yes"
    )
    local_locked_aug = sum(
        1 for row in manifest_rows
        if row["source_layer"] == "classification_aug" and row["locked_overlap"] == "yes"
    )
    local_dev_pre = sum(
        1 for row in manifest_rows
        if row["source_layer"] == "classification_pre" and row["development_overlap"] == "yes"
        and row["locked_overlap"] == "no"
    )

    summary = {
        "governance_date": "2026-08-29",
        "raw_data_unchanged": True,
        "gpu_used": False,
        "v1_modified": False,
        "locked_used_for_selection": False,
        "classification_pre_images": c["pre_images"],
        "classification_aug_images": c["aug_images"],
        "classification_aug_groups": c["augmentation_group_count"],
        "segmentation_images": s["images"],
        "segmentation_masks": s["masks"],
        "segmentation_json": s["json_annotations"],
        "exact_overlap_files": audit["overlap_with_recovered"]["exact_overlap_count"],
        "exact_overlap_local_files": len({
            new_path
            for item in audit["overlap_with_recovered"].get("exact_overlap", [])
            for _, new_path in item.get("matches", [])
        }),
        "locked_overlap_cases": len(locked_overlap_cases),
        "locked_overlap_local_originals": local_locked_pre,
        "locked_overlap_local_augmentations": local_locked_aug,
        "development_overlap_cases": len(dev_overlap_cases),
        "development_overlap_local_originals_excluding_locked": local_dev_pre,
        "unlisted_overlap_cases": len(unlisted_overlap_cases),
        "exact_duplicate_groups": c["exact_duplicate_image_hash_groups"],
        "xml_invalid_boxes": len(c["invalid_boxes"]),
        "yolo_invalid_rows": len(c["yolo_bad"]),
        "xml_size_mismatch": len(c["xml_dim_mismatch"]),
        "xml_missing_filename": len(c["xml_missing_img"]),
        "seg_json_without_imageData": s["json_stats"].get("no_imageData", 0),
        "seg_dimension_drift": s["stem_dimension_mismatch_count"],
        "unconditional_safe_train_assets": 0,
    }
    (OUT / "governance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 血管数据集数字治理结果（2026-08-29）

## 结论

本次治理只读执行，未使用 GPU，未修改原始数据、v1 基线或任何评测文件；没有使用 locked 集做模型选择。治理输出是状态清单和人工审核队列，不是对原始文件的覆盖修复。

**当前无条件可直接训练的资产数为 0。** 这不是说数据没有价值，而是说明在来源、标签语义和配对关系完成人工确认前，不能把它当作金标准训练集。

## 已确认事实

- 分类扩充前：{c['pre_images']} 张原图、{c['pre_xml']} 个 XML；扩充后：{c['aug_images']} 张增强图，按 {c['augmentation_group_count']} 个原图组、每组 20 张管理。
- 原图 418、423 没有对应增强组，不能假定扩充完整。
- 分割：{s['images']} 张图、{s['masks']} 张 mask、{s['json_annotations']} 个 JSON。
- 与 recovered 原始病例精确重叠：recovered 侧 {audit['overlap_with_recovered']['exact_overlap_count']} 条文件记录，对应本地 {len({new_path for item in audit['overlap_with_recovered'].get('exact_overlap', []) for _, new_path in item.get('matches', [])})} 个分类原图；涉及 {len(locked_overlap_cases)} 个 locked 病例、{len(dev_overlap_cases)} 个 development 病例和 {len(unlisted_overlap_cases)} 个未入清单病例。
- locked 重叠在本地表现为 {local_locked_pre} 个原图及其 {local_locked_aug} 个增强派生物；这些文件全部永久排除。
- 分类原图内部完全重复组：{c['exact_duplicate_image_hash_groups']} 组；XML 异常框 {len(c['invalid_boxes'])} 个；YOLO 异常行 {len(c['yolo_bad'])} 个。
- 分割 3,161 个 JSON 的 imagePath 均无法在本机直接解析；其中无 imageData {s['json_stats'].get('no_imageData', 0)} 个；图像/mask/JSON 尺寸或索引漂移 {s['stem_dimension_mismatch_count']} 个；JSON 与图像/mask 资产数量不一致。

## 当前资产状态

| 资产层 | 当前状态 | 允许用途 |
|---|---|---|
| 分类原图 + 图片级 Excel 标签 | HOLD | 完成来源、语义和抽样复核后，做图片级辅助监督 |
| XML/YOLO 血管框类别 | HOLD | 修复坐标和类别定义后，做血管检测/候选计数 |
| 分类增强图 | HOLD | 只能跟随原图组进入同一 split，不能独立拆分 |
| 分割图 + mask + JSON | HOLD_SEGMENTATION_PAIRING | 重建一一对应并抽查后，做二值血管分割预训练 |
| 与 17 个 locked 病例重叠的文件及派生物 | EXCLUDE | 永久排除训练、调参和模型选择 |

## 必须人工审核的内容

人工审核队列见 `manual_review_queue.csv`。其中 P0 项不得自动放行：

队列是按审核范围汇总的，条目之间可能重叠（例如同一 JSON 同时存在路径缺失和尺寸漂移）；各行数量不可直接相加。

1. {len(locked_overlap_cases)} 个 locked 病例及其所有增强派生物是否完整排除；
2. 未入清单病例和其余未知来源图像的真实来源、病例归属、设备和使用权限；
3. `cross_vessel`、`malformed_vessel`、出血、乳头下静脉丛、乳头和汗腺导管的操作性定义；
4. {len(c['invalid_boxes'])} 个坏 XML 框、{len(c['yolo_bad'])} 行坏 YOLO，以及 {len(c['xml_dim_mismatch'])} 个 XML 尺寸不一致样本；
5. 分割 JSON、图像和 mask 的真实配对，尤其是 {s['stem_dimension_mismatch_count']} 个漂移样本和 {s['json_stats'].get('no_imageData', 0)} 个无内嵌图像样本；
6. 原图 418、423 的增强缺失原因，以及 580 个增强组是否确实由对应原图派生；
7. 至少 100 张分类原图和 100 对分割图/mask 的代表性抽查，并记录标注完整性、歧义和复核一致性；
8. 像素尺度、放大倍数和设备标定是否足以支持微米级几何量。

AI/脚本已经完成文件清点、hash 去重、增强组识别、格式异常检测和 locked 重叠筛选；但不能替代上述临床语义、边界样本、物理标定和金标准确认。

## 输出文件

- `governance_manifest.csv`：15341 个资产的逐文件状态；
- `candidate_training_manifest.csv`：当前候选训练原图，全部标记 HOLD；
- `manual_review_queue.csv`：按优先级汇总的人工审核队列；
- `governance_summary.json`：机器可读汇总。
"""
    (OUT / "governance_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
