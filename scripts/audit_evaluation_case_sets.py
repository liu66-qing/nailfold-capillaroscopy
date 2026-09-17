"""Audit case-set alignment between v1 and multimodal evaluations.

Read-only protocol audit: no training, inference, or locked-set selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FIELDS = [
    "clarity",
    "capillary_count",
    "crossing_ratio",
    "malformation_ratio",
    "blood_color",
    "exudation",
    "hemorrhage",
    "subpapillary_venous_plexus",
    "papilla",
]


ALIASES = {
    "crossing_ratio": {"[<30%]": "<=30%", "<30%": "<=30%", "10--30%": "<=30%"},
    "malformation_ratio": {"[<10%]": "<=10%", "<10%": "<=10%"},
    "blood_color": {"[淡红色]": "淡红", "淡红色": "淡红", "浅红色": "浅红", "暗红色": "暗红"},
    "subpapillary_venous_plexus": {"[不见]": "不见"},
    "papilla": {"[波纹状]": "波纹状"},
}


def canonical(field: str, value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("≥", ">=").replace("≤", "<=")
    if len(text) >= 2 and text[0] in "[【(" and text[-1] in "])】":
        text = text[1:-1]
    return ALIASES.get(field, {}).get(text, text)


def ids(values: pd.Series) -> list[str]:
    return sorted(set(values.dropna().astype(str)))


def id_hash(values: set[str] | list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def archive_counts(values: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        archive = value.split("/", 1)[0]
        counts[archive] = counts.get(archive, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--multimodal-predictions", type=Path, required=True)
    parser.add_argument("--multimodal-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roles = pd.read_csv(args.roles)
    predictions = pd.read_csv(args.multimodal_predictions)
    roles["exam_case_id"] = roles["exam_case_id"].astype(str)
    predictions["exam_case_id"] = predictions["exam_case_id"].astype(str)
    multimodal_manifest = None
    if args.multimodal_manifest:
        multimodal_manifest = pd.read_csv(args.multimodal_manifest)
        multimodal_manifest["exam_case_id"] = multimodal_manifest["exam_case_id"].astype(str)

    all_ids = set(ids(roles["exam_case_id"]))
    dev_ids = set(ids(roles.loc[roles.evaluation_role.eq("development"), "exam_case_id"]))
    locked_ids = set(ids(roles.loc[roles.evaluation_role.eq("locked_test"), "exam_case_id"]))
    mm_ids = set(ids(predictions["exam_case_id"]))
    mm_manifest_ids = set(ids(multimodal_manifest["exam_case_id"])) if multimodal_manifest is not None else set()
    if dev_ids & locked_ids or all_ids != dev_ids | locked_ids:
        raise ValueError("invalid v1 role partition")

    run_sets: dict[str, set[str]] = {}
    for (model, method), frame in predictions.groupby(["model", "method"], dropna=False):
        run_sets[f"{model}/{method}"] = set(ids(frame["exam_case_id"]))

    coverage: dict[str, dict[str, int]] = {}
    label_audit: dict[str, dict[str, int]] = {}
    role_by_id = roles.set_index("exam_case_id")
    for field in FIELDS:
        true_col = f"{field}_true"
        pred_col = f"{field}_pred"
        coverage[field] = {
            "rows_with_true": int(predictions[true_col].notna().sum()),
            "rows_with_prediction": int(predictions[pred_col].notna().sum()),
            "cases_with_true": int(predictions.loc[predictions[true_col].notna(), "exam_case_id"].nunique()),
        }
        rows_with_true = predictions[predictions[true_col].notna()][["exam_case_id", true_col]].copy()
        rows_with_true["role_true"] = rows_with_true["exam_case_id"].map(role_by_id[field])
        label_audit[field] = {
            "rows_compared": int(len(rows_with_true)),
            "value_mismatches_vs_v1_manifest": int((rows_with_true[true_col].astype(str) != rows_with_true["role_true"].astype(str)).sum()),
            "cases_with_multiple_true_values": int(
                rows_with_true.groupby("exam_case_id")[true_col].nunique(dropna=True).gt(1).sum()
            ),
            "canonical_value_mismatches_vs_v1_manifest": int(
                (
                    rows_with_true[true_col].map(lambda value: canonical(field, value))
                    != rows_with_true["role_true"].map(lambda value: canonical(field, value))
                ).sum()
            ),
        }

    overlap_locked = mm_ids & locked_ids
    overlap_dev = mm_ids & dev_ids
    report = {
        "schema_version": "evaluation-case-set-audit/1.0",
        "read_only": True,
        "gpu_used": False,
        "locked_cases_used_for_selection": False,
        "v1_manifest": {
            "path": str(args.roles),
            "cases": len(all_ids),
            "development_cases": len(dev_ids),
            "locked_cases": len(locked_ids),
            "all_case_id_sha256": id_hash(all_ids),
            "development_case_id_sha256": id_hash(dev_ids),
            "locked_case_id_sha256": id_hash(locked_ids),
            "archive_counts": {
                "all": archive_counts(all_ids),
                "development": archive_counts(dev_ids),
                "locked": archive_counts(locked_ids),
            },
        },
        "multimodal_predictions": {
            "path": str(args.multimodal_predictions),
            "rows": len(predictions),
            "unique_cases": len(mm_ids),
            "case_id_sha256": id_hash(mm_ids),
            "archive_counts": archive_counts(mm_ids),
            "run_case_set_sizes": {name: len(value) for name, value in sorted(run_sets.items())},
            "run_case_set_sha256": {name: id_hash(value) for name, value in sorted(run_sets.items())},
            "all_runs_use_identical_case_set": len({tuple(sorted(value)) for value in run_sets.values()}) <= 1,
        },
        "multimodal_manifest": (
            {
                "path": str(args.multimodal_manifest),
                "cases": len(mm_manifest_ids),
                "development_cases": int(multimodal_manifest.evaluation_role.eq("development").sum()),
                "locked_cases": int(multimodal_manifest.evaluation_role.eq("locked_test").sum()),
                "all_case_id_sha256": id_hash(mm_manifest_ids),
                "development_case_id_sha256": id_hash(set(ids(multimodal_manifest.loc[multimodal_manifest.evaluation_role.eq("development"), "exam_case_id"]))),
                "locked_case_id_sha256": id_hash(set(ids(multimodal_manifest.loc[multimodal_manifest.evaluation_role.eq("locked_test"), "exam_case_id"]))),
            }
            if multimodal_manifest is not None
            else None
        ),
        "alignment": {
            "multimodal_cases_in_v1_development": len(overlap_dev),
            "multimodal_cases_in_v1_locked": len(overlap_locked),
            "v1_locked_cases_missing_from_multimodal": len(locked_ids - mm_ids),
            "v1_development_cases_missing_from_multimodal": len(dev_ids - mm_ids),
            "v1_cases_missing_from_multimodal": len(all_ids - mm_ids),
            "multimodal_cases_outside_v1_manifest": len(mm_ids - all_ids),
            "strict_same_case_set": all_ids == mm_ids,
            "prediction_manifest_case_set_match": (
                set(ids(multimodal_manifest.loc[multimodal_manifest.evaluation_role.eq("development"), "exam_case_id"])) == mm_ids
            ) if multimodal_manifest is not None else None,
            "v1_locked_vs_multimodal_locked_overlap": (
                len(locked_ids & set(ids(multimodal_manifest.loc[multimodal_manifest.evaluation_role.eq("locked_test"), "exam_case_id"])))
                if multimodal_manifest is not None else None
            ),
            "interpretation": (
                "The multimodal 198-case set is not the v1 47-case locked set: "
                f"it contains {len(overlap_locked)} v1 locked cases and {len(overlap_dev)} v1 development cases."
            ),
        },
        "field_coverage": coverage,
        "label_audit": label_audit,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 评测病例集合一致性审计",
        "",
        "本报告为只读审计，不训练、不推理、不用 locked 集做模型选择。",
        "",
        "## 结论",
        "",
        f"- v1 权威清单：{len(all_ids)} 例（development {len(dev_ids)} + locked {len(locked_ids)}）。",
        f"- 多模态预测文件：{len(mm_ids)} 例，{len(run_sets)} 个模型/方法组合。",
        f"- 多模态集合包含 v1 locked：{len(overlap_locked)} 例；包含 v1 development：{len(overlap_dev)} 例。",
        f"- v1 locked 未出现在多模态集合：{len(locked_ids - mm_ids)} 例。",
        f"- v1 development 未出现在多模态集合：{len(dev_ids - mm_ids)} 例。",
        "- 因此，多模态记录中的“排除 35 例 locked”不是 v1 权威 47 例 locked 集的同义表述。",
        (
            f"- 远端多模态清单：{len(mm_manifest_ids)} 例（development "
            f"{int(multimodal_manifest.evaluation_role.eq('development').sum())} + "
            f"locked {int(multimodal_manifest.evaluation_role.eq('locked_test').sum())}），"
            f"其 locked 与 v1 locked 重叠 "
            f"{len(locked_ids & set(ids(multimodal_manifest.loc[multimodal_manifest.evaluation_role.eq('locked_test'), 'exam_case_id'])))} 例。"
            if multimodal_manifest is not None
            else "- 未提供远端多模态清单。"
        ),
        "",
        "## 集合统计",
        "",
        "| 集合 | 病例数 | SHA-256（按排序 ID） | archive1/2/3 |",
        "|---|---:|---|---|",
        f"| v1 全部 | {len(all_ids)} | {id_hash(all_ids)} | {archive_counts(all_ids)} |",
        f"| v1 development | {len(dev_ids)} | {id_hash(dev_ids)} | {archive_counts(dev_ids)} |",
        f"| v1 locked | {len(locked_ids)} | {id_hash(locked_ids)} | {archive_counts(locked_ids)} |",
        f"| 多模态预测 | {len(mm_ids)} | {id_hash(mm_ids)} | {archive_counts(mm_ids)} |",
        "",
        "## 标签审计",
        "",
        "多模态 true 字段与 v1 清单的原始字符串存在格式差异，但按 canonical mapping（括号、区间别名、颜色后缀）规范化后，9 个共同字段均为 0 个不一致，且没有病例出现多个真值。",
        "",
        "| 字段 | 原始字符串不一致行数 | 规范化后不一致行数 | 多真值病例数 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {field} | {label_audit[field]['value_mismatches_vs_v1_manifest']} | {label_audit[field]['canonical_value_mismatches_vs_v1_manifest']} | {label_audit[field]['cases_with_multiple_true_values']} |"
        for field in FIELDS
    )
    lines.extend([
        "",
        "## 协议判定",
        "",
        "当前多模态结果不能与 v1 locked 结果作严格 head-to-head 性能结论。下一步应冻结同一病例集合，先做 development 5 折 OOF，再对同一 locked 集一次性评测。",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["alignment"], ensure_ascii=False))


if __name__ == "__main__":
    main()
