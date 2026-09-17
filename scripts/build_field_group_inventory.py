"""Build a field-group experiment inventory from development artifacts.

The report keeps classification BA, numeric normalized scores, and compatibility
default hit rates separate. It never reads locked predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    binary = load(root / "artifacts/evaluation/current_dual_seg_binary_v1/metrics.json")
    step12 = load(root / "artifacts/pipeline/step12_r7/step12_report.json")
    count = load(root / "artifacts/evaluation/step7_count_gbt.json")
    video = load(root / "artifacts/video/step11/step11_report.json")
    fallback = load(root / "artifacts/video/step11/fallback.json")
    geometry = load(root / "artifacts/geometry_exact/step8_report.json")
    step10 = load(root / "artifacts/features/step10_oof/step10_report.json")

    b = binary["fields"]
    s = step12["field_scores"]
    g = geometry["metrics"]
    rows = [
        {
            "group": "A 整体质量与颜色",
            "fields": "clarity; blood_color",
            "best": f"正式二级 dual_seg OOF BA: clarity {pct(b['clarity']['balanced_accuracy'])}; blood_color {pct(b['blood_color']['balanced_accuracy'])}",
            "attempts": "三等级 SigLIP/GBT; DINOv2/HuluMed/dual/dual_seg; robust mean/std/median; video-Huber与帧选择; case-mean latent; 严格域适应; MIL; geometry 拼接; LoRA; TTA; 阈值/类别权重",
            "decision": "低分/中间组。二级定义已保留，但未达到稳定 70% BA；停止泛化模型搜索，优先做标签审查、颜色/曝光可测性门控。",
        },
        {
            "group": "B 管袢实例与形态",
            "fields": "capillary_count; crossing_ratio; malformation_ratio",
            "best": f"count 专用 GBT 三分类 BA {pct(count['balanced_accuracy'])}; step12_r7 crossing {pct(s['crossing_ratio'])}, malformation {pct(s['malformation_ratio'])}",
            "attempts": "YOLO 多类分割(v1/plus-pseudo); 阈值扫描; no-expand/skeleton; count GBT; mask统计; DINO/Hulu/seg分类; MIL/视频质量/增强; 交叉与畸形比例由结构特征路由",
            "decision": "真正低分组。计数仍受漏检/碎片和病例级区间标签影响；形态比例缺实例级真值。只允许后处理/分割改动通过全字段回归门。",
        },
        {
            "group": "C 单袢物理几何",
            "fields": "afferent_diameter; efferent_diameter; apex_diameter; loop_length; output_input_ratio",
            "best": f"step12_r7 归一化分数: afferent {pct(s['afferent_diameter'])}; efferent {pct(s['efferent_diameter'])}; apex {pct(s['apex_diameter'])}; length {pct(s['loop_length'])}; ratio 由管径派生。该分数不是分类准确率。",
            "attempts": f"geometry-v2 mask/centerline/scale; visual/structured/all ExtraTrees/Ridge; Hulu/SigLIP; video增强路由。geometry_exact MAE: afferent {g['afferent_diameter']['model_mae']:.2f}, efferent {g['efferent_diameter']['model_mae']:.2f}, apex {g['apex_diameter']['model_mae']:.2f}, length {g['loop_length']['model_mae']:.2f}",
            "decision": "高分保护组（按综合归一化指标），冻结为回归保护；不能把 88--98% 归一化分数解释成医生一致性或绝对准确率。后续只在有结构金标准和设备标定时改。",
        },
        {
            "group": "D 局部病灶",
            "fields": "exudation; hemorrhage",
            "best": f"正式二级 exudation BA {pct(b['exudation']['balanced_accuracy'])}; hemorrhage 旧三等级 BA {pct(step10['fields']['hemorrhage']['balanced_accuracy'])}，阳性极少。",
            "attempts": "四级/二级合并; SigLIP/Hulu/DINO/dual_seg; MIL与固定帧; Qwen/QLoRA; geometry/增强; 阈值与置信权重",
            "decision": "exudation 为低/中间组，二级可交付但需复核；hemorrhage 低支持探索性/异常触发，不得用 94% 多数类普通准确率宣称学会。",
        },
        {
            "group": "E 背景组织结构",
            "fields": "subpapillary_venous_plexus; papilla; sweat_duct",
            "best": f"正式二级 SVP BA {pct(b['subpapillary_venous_plexus']['balanced_accuracy'])}; papilla 探索性 BA {pct(b['papilla']['balanced_accuracy'])}; sweat_duct 默认命中率 {pct(s['sweat_duct'])}（非模型能力）。",
            "attempts": "多等级与二级合并; DINO/Hulu/dual_seg; robust pooling/MIL/video-Huber; case-mean latent; 域适应; classifier chain; Qwen质量选择; MedSAM/SigLIP2筛选",
            "decision": "SVP 是中间组；papilla 低分且探索性；汗腺导管采用默认+异常触发。不要把背景字段与几何组共享模型或目标。",
        },
        {
            "group": "F 视频动态与细胞事件",
            "fields": "flow_state; vasomotion; rbc_aggregation; wbc_count; microthrombus; flow_speed",
            "best": f"step11 flow_state BA {pct(video['balanced_accuracy'])}，未过 0.35 门；fallback flow_state 普通命中 {pct(fallback['fields']['flow_state']['ordinary_accuracy'])}、RBC {pct(fallback['fields']['rbc_aggregation']['ordinary_accuracy'])}。vasomotion/WBC 为默认兼容率约 98%，flow_speed 无有效标签。",
            "attempts": "连续视频光流/Hulu特征; stable segment; K8/K12 video-Huber; Qwen候选帧; dynamic fallback",
            "decision": "低分且视频覆盖受限（95/186 开发病例）。动态字段必须保持视频证据和观察时长；当前只可兼容性降级，不能当静态分类问题继续调参。",
        },
    ]
    lines = [
        "# Field-Group Experiment Inventory", "",
        "Generated from development artifacts only. All cited OOF artifacts require `locked_cases_seen=0`; the historical 47-case locked cohort is not used for selection.", "",
        "## Protocol boundary", "",
        "- Formal weak-field comparator: `current_dual_seg_binary_v1`, 186 development cases, fixed five case folds, DINOv2 + HuluMed + segmentation features, ExtraTrees (300 trees, leaf=2, balanced), pooled OOF balanced accuracy.",
        "- Historical multi-class and step12 values are diagnostic context only and are not numerically interchangeable with the formal binary comparator.",
        "- Numeric geometry scores are `1 - MAE / field_range`; compatibility scores are default-value hit rates. Neither is classification balanced accuracy.",
        "- A simple frame-after-aggregation pipeline is not called MIL. MIL is reserved for explicit case-bag supervision.", "",
        "| 组 | 字段 | 当前最佳可比结果 | 已尝试方向 | 判定与下一步 |", "|---|---|---|---|---|",
    ]
    for row in rows:
        vals = [row[k].replace("|", "\\|").replace("\n", " ") for k in ("group", "fields", "best", "attempts", "decision")]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## Freeze / reopen policy", "", "- Freeze Group C geometry routes as protected regression components; any segmentation change must report all four geometry MAEs and stop on predeclared degradation.", "- Treat compatibility defaults (sweat duct, vasomotion, WBC, flow speed) as product policy with anomaly/review triggers, not evidence of learned recognition.", "- Reopen only Groups B, D, E and F with group-specific evidence: instance masks/centerlines for B/C, lesion ROI evidence for D, background-region/evaluability labels for E, and continuous timestamped clips for F.", "- For Groups A/D/E, the next defensible no-new-doctor-label work is calibration/evaluability and error-queue review; another global encoder or pooling sweep is not supported by current evidence."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
