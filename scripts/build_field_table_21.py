# -*- coding: utf-8 -*-
"""Assemble the 21-field delivery table, each cell traced to a frozen artifact.

Rule that shapes the whole file: the locked-47 columns can only be filled from
locked reads that ALREADY happened. The standing rule is that locked-47 must not
be read again for tuning, model selection, threshold choice, field choice or
output semantics, and the one post-freeze read in this round is spent. So:

  * five fields have real locked numbers (locked_delivered_20260918, A0 config)
  * malformation_ratio additionally has the A2x read (locked_consumed_20260924)
  * every other field's locked cells are NOT_AUTHORISED_TO_READ -- not "unknown",
    not zero, and not silently filled from development

BA for the 2026-09-18 fields is DERIVED from the confusion matrices stored in that
run, which is arithmetic on an existing artifact, not a new read. AUROC and the
70%-coverage accuracy cannot be derived that way: that run stored no per-case
probabilities. Those cells stay empty with the reason attached.
"""
import io
import json
import os

import numpy as np
import pandas as pd

OUT = "artifacts/audits/field_table_21_20260925"
os.makedirs(OUT, exist_ok=True)

DELIVERED = "artifacts/experiments/locked_delivered_20260918/locked_delivered.json"
CONSUMED = "artifacts/experiments/locked_consumed_20260924/locked_consumed_evaluation.json"
MATRIX = "artifacts/experiments/medical_encoder_transfer_20260921/field_matrix_three_arms.csv"
LADDER = "artifacts/experiments/rescue_external_20260922/ladder_local/ladder_local.csv"
FROZEN_A2X = ("artifacts/experiments/rescue_external_20260922/"
              "frozen_candidate_malformation_a2x/frozen_candidate.json")
V1_LOCKED = "artifacts/evaluation/final_locked_evaluation_v1.json"
ROUTING = "artifacts/experiments/product_contract_20260924/field_routing.csv"
NEWFIELD = "artifacts/audits/new_field_probe_20260925/new_field_probe.json"
REV = "server_code_audit/locked_evaluation_v1_reviewed.csv"


def read_json(p):
    return json.load(io.open(p, encoding="utf-8"))


def write(name, obj):
    io.open(os.path.join(OUT, name), "w", encoding="utf-8").write(
        json.dumps(obj, ensure_ascii=False, indent=1))


delivered = read_json(DELIVERED)
consumed = read_json(CONSUMED)
frozen = read_json(FROZEN_A2X)
v1 = read_json(V1_LOCKED)
newfield = read_json(NEWFIELD)
matrix = pd.read_csv(MATRIX, encoding="utf-8")
ladder = pd.read_csv(LADDER, encoding="utf-8")
routing = pd.read_csv(ROUTING, encoding="utf-8").set_index("field")
rev = pd.read_csv(REV, dtype={"exam_case_id": str})

anchor = matrix[matrix.arm == "anchor_dinov2b_deployed"].set_index("field")

NOT_AUTH = "NOT_AUTHORISED_TO_READ"
NO_PROB = "NOT_DERIVABLE_no_stored_probabilities"


def ba_from_confusion(c):
    r0 = c["tn"] / (c["tn"] + c["fp"])
    r1 = c["tp"] / (c["tp"] + c["fn"])
    return round((r0 + r1) / 2, 4), round(r0, 4), round(r1, 4)


def dev_cell(field):
    """development OOF under the single shipped configuration."""
    if field not in anchor.index:
        return {}
    r = anchor.loc[field]
    return {
        "dev_n": int(r.n),
        "dev_classes": int(r.n_classes),
        "dev_acc": round(float(r.accuracy), 4),
        "dev_baseline": round(float(r.baseline_constant), 4),
        "dev_delta": round(float(r.delta), 4),
        "dev_ba": round(float(r.balanced_accuracy), 4),
        "dev_auroc": (round(float(r.auroc_macro_ovr), 4)
                      if not pd.isna(r.auroc_macro_ovr) else None),
        "dev_collapsed": bool(r.collapsed_to_one_class),
        "dev_smallest_class": int(r.smallest_class),
        "unit_of_observation": str(r.unit_of_observation),
        "observable_from_static_image": bool(r.observable_from_static_image),
    }


def locked_from_delivered(field):
    """Real locked numbers from the 2026-09-18 read. BA/recalls derived from the
    stored confusion matrix; AUROC and the coverage curve are not derivable."""
    v = delivered["fields"][field]
    L, LM, c = v["locked"], v["locked_vs_locked_majority"], v["confusion"]
    ba, r0, r1 = ba_from_confusion(c)
    return {
        "locked_source": "locked_delivered_20260918 (A0 shipped config)",
        "locked_n": L["n"],
        "locked_acc": L["accuracy"],
        "locked_acc_ci95": L["acc_ci95"],
        "locked_own_mode_baseline": LM["baseline_dev_majority_answer"],
        "locked_delta_vs_own_mode": LM["delta"],
        "locked_delta_ci95": LM["ci95"],
        "locked_delta_ci_excludes_zero": bool(LM["passes"]),
        "locked_carried_dev_baseline": L["baseline_dev_majority_answer"],
        "locked_delta_vs_carried_dev_baseline": L["delta"],
        "prevalence_flip": v["prevalence_shift"]["dev_majority_answer_is_wrong_on_locked"],
        "locked_ba_derived": ba,
        "locked_recall_class0": r0,
        "locked_recall_class1": r1,
        "locked_confusion_rows_true_cols_pred": [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
        "locked_auroc": NO_PROB,
        "locked_acc_at_70pct_coverage": NO_PROB,
        "locked_collapsed": v["collapsed_to_one_class"],
    }


def locked_unread(field):
    n = v1["fields"].get(field, {}).get("cases")
    return {
        "locked_source": None,
        "locked_n_labelled_available": n,
        "locked_n_provenance": ("label availability counted in "
                               "final_locked_evaluation_v1.json (a 2026-08 read, "
                               "already spent); no image was opened for this table"),
        "locked_acc": NOT_AUTH,
        "locked_delta": NOT_AUTH,
        "locked_ba": NOT_AUTH,
        "locked_auroc": NOT_AUTH,
        "locked_recalls": NOT_AUTH,
        "locked_confusion": NOT_AUTH,
        "locked_ci95": NOT_AUTH,
        "locked_acc_at_70pct_coverage": NOT_AUTH,
        "why": ("locked-47 may no longer be read for field choice, model choice, "
                "threshold choice or output semantics; the one post-freeze read in "
                "this round is spent on malformation_ratio"),
    }


def fixed_mode_field(field, value):
    """Fields shipped as a fixed constant. The v1 'score' for these is a
    COMPATIBILITY rate -- how often the constant matches the report -- and must
    never be read as model accuracy."""
    dist = rev.loc[rev.development_fold.notna(), field].value_counts(dropna=False)
    dist = {str(k): int(n) for k, n in dist.items()}
    top = max(dist, key=lambda k: dist[k] if k != "nan" else -1)
    tot = sum(n for k, n in dist.items() if k != "nan")
    return {
        "model": "none -- fixed constant output",
        "fixed_value": value,
        "dev_distribution": dist,
        "dev_share_of_fixed_value": round(dist.get(value, 0) / tot, 4) if tot else None,
        "dev_mode_value": top,
        "locked_compatibility_rate": v1["fields"].get(field, {}).get("score"),
        "locked_n_labelled_available": v1["fields"].get(field, {}).get("cases"),
        "compatibility_is_not_accuracy": ("this rate is how often a constant matches "
                                         "the printed report, with no model involved; "
                                         "it must not be quoted as model accuracy"),
        "not_a_patient_measurement": True,
        "disposition": "fixed",
        "rag": "not_modelled / unknown -- must not be presented as a fact about "
               "this patient",
    }


ROWS = []


def row(**kw):
    ROWS.append(kw)


DINO = "DINOv2-B/14 整图特征 + 5 池化 + PCA64 + LogisticRegression(C=0.03)"

# ---- 1. clarity -------------------------------------------------------------
row(field="clarity",
    model=DINO + "（A0 部署配置）",
    target="二分类 清晰 vs 欠清/模糊；输入 static 显微原图，病例级",
    **dev_cell("clarity"), **locked_from_delivered("clarity"),
    disposition="可交付（Tier 1）；但 locked 上对自身众数的 delta 仅 +0.087，CI 含 0",
    alternative_info="无替代量；这是图像质量门，不是临床发现",
    rag="reliable_observation，可进建议、可作为其他字段的门控")

# ---- 2. exudation -----------------------------------------------------------
row(field="exudation",
    model=DINO + "（A0 部署配置）",
    target="二分类 无 vs +/++/+++；输入 static 显微原图，病例级",
    **dev_cell("exudation"), **locked_from_delivered("exudation"),
    disposition="⚠️ development 上 +0.240，locked 上对自身众数 delta 恰为 0.000 —— "
                "未在留出上复现",
    alternative_info="无",
    rag="image_correlation，可进建议但须按 locked 结果降级陈述")

# ---- 3. subpapillary_venous_plexus -----------------------------------------
row(field="subpapillary_venous_plexus",
    model=DINO + "（A0 部署配置）",
    target="二分类 未见/可见；输入 static 显微原图，病例级",
    **dev_cell("subpapillary_venous_plexus"),
    **locked_from_delivered("subpapillary_venous_plexus"),
    disposition="仅有信号（Tier 2）；locked delta +0.128，CI [-0.043, 0.298] 含 0",
    alternative_info="无",
    rag="image_correlation，仅研究提示，不进建议")

# ---- 4. blood_color ---------------------------------------------------------
row(field="blood_color",
    model=DINO + "（A0 部署配置）",
    target="二分类 正常 vs 暗红/淡红；输入 static 显微原图，病例级",
    **dev_cell("blood_color"), **locked_from_delivered("blood_color"),
    disposition="仅有信号（Tier 2）；locked delta +0.114，CI 含 0",
    alternative_info="无；颜色受光源和白平衡影响，未做设备间校正",
    rag="image_correlation，仅研究提示，不进建议")

# ---- 5. malformation_ratio (A2x, the user's pre-filled row) ------------------
ab70 = [x for x in consumed["coverage_accuracy_curve"]
        if x.get("target_coverage") == 0.7][0]
dev70 = [x for x in frozen["abstention"] if x.get("target_coverage") == 0.7][0]
a2x, a0c = consumed["a2x"], consumed["a0"]
row(field="malformation_ratio",
    model="A2x = DINOv2-B 整图特征 + HF 形态检测器自动局部特征（832 维）",
    target="二分类 ≤10% vs >10%；**不得输出实际百分比**；局部区域由训练折产生的"
           "检测器自动提取",
    **dev_cell("malformation_ratio"),
    dev_a2x_acc=0.7284, dev_a2x_delta=0.1605, dev_a2x_ba=0.7216,
    dev_a2x_auroc=0.7766,
    dev_a2x_acc_at_70pct_coverage=dev70["accuracy"],
    dev_a2x_70pct_gate_passes=False,
    locked_source="locked_consumed_20260924（A2x 冻结后单次读取）",
    locked_n=consumed["n"], locked_excluded_no_label=consumed["excluded_no_label"],
    locked_acc=a2x["accuracy"],
    locked_own_mode_baseline=consumed["baselines"]["locked_own_mode"]["accuracy"],
    locked_delta_vs_own_mode=round(a2x["accuracy"]
                                   - consumed["baselines"]["locked_own_mode"]["accuracy"], 4),
    locked_carried_dev_baseline=consumed["baselines"]["development_mode"]["accuracy"],
    locked_ba=a2x["balanced_accuracy"], locked_auroc=a2x["auroc"],
    locked_recall_class0=a2x["per_class_recall"]["0"],
    locked_recall_class1=a2x["per_class_recall"]["1"],
    locked_confusion_rows_true_cols_pred=a2x["confusion_matrix"]["rows_true_cols_pred"],
    locked_ece=a2x["ece"],
    locked_acc_at_70pct_coverage=ab70["accuracy"],
    locked_a2x_minus_a0_ba=consumed["a2x_minus_a0"]["ba_gain"],
    locked_a2x_minus_a0_ba_ci=consumed["a2x_minus_a0"]["ba_ci"],
    locked_a2x_minus_a0_ci_excludes_zero=consumed["a2x_minus_a0"]["ba_ci_excludes_zero"],
    locked_a0_acc=a0c["accuracy"], locked_a0_ba=a0c["balanced_accuracy"],
    disposition="强研发候选，未达原定交付门槛（70% 覆盖 + 85% 准确率：development "
                "0.7876、locked 0.6667，均不过；提高拒答率也到不了）",
    alternative_info="自动检测形态统计是**相关特征，不等价于医生百分比**（人工框只 "
                     "17/44 落在自己档内，约 3 倍尺度偏移）",
    rag="最多 image_correlation；**不得用于阴性排除**；只允许四值 "
        "low_band_correspondence / high_band_correspondence / unknown / rejected")

# ---- 6. papilla -------------------------------------------------------------
pap = ladder[(ladder.field == "papilla")].set_index("rung")
row(field="papilla",
    model=DINO + "（A0）；**不得采用已失败的 A1+A2 融合**（A1+A2 比 A1 差 −0.0224，"
          "三条件全不过）",
    target="三分类 乳头形态（唯一必须保留 3 类的字段）；输入 static 显微原图",
    **dev_cell("papilla"),
    dev_a2_ba=float(pap.loc["A2", "balanced_accuracy"]),
    dev_a2x_ba=float(pap.loc["A2x", "balanced_accuracy"]),
    dev_note="四臂 BA 全在 0.40~0.46，基线 0.4162；A0 delta 仅 +0.027",
    **locked_unread("papilla"),
    disposition="仅有信号；**模型侧暂停**",
    alternative_info="检测器局部特征无增益（A2x 为 −0.015）",
    rag="image_correlation，仅研究提示，不进建议")

# ---- 7. crossing_ratio ------------------------------------------------------
cro = ladder[(ladder.field == "crossing_ratio")].set_index("rung")
row(field="crossing_ratio",
    model=DINO + "（A0）",
    target="二分类 ≤30% vs >30%；**不得输出实际比例**",
    **dev_cell("crossing_ratio"),
    dev_a2x_ba=float(cro.loc["A2x", "balanced_accuracy"]),
    dev_note="A0 delta −0.0113（负）；最好的 A2x 也只有 +0.0169，BA 0.5812",
    **locked_unread("crossing_ratio"),
    disposition="无信号；转入**重新定义参考标准**（cross_vessel 检测器 AP 在五折上"
                "上限 0.3156，瓶颈在我们自己的标注）",
    alternative_info="自动 crossing 框数须作为**新字段另评估**，见 new_field_1；"
                     "实测与临床档 rho 0.105（CI 含 0），本地检测器给 −0.036，"
                     "两个检测器符号相反 → **不等价**",
    rag="unknown，不进建议")

# ---- 8. capillary_count -----------------------------------------------------
cap = ladder[(ladder.field == "capillary_count")].set_index("rung")
row(field="capillary_count",
    model=DINO + "（A0）",
    target="三分类 条/mm 档位；**不得输出条/mm 数值**（设备 UNCALIBRATED）",
    **dev_cell("capillary_count"),
    dev_a2_ba=float(cap.loc["A2", "balanced_accuracy"]),
    dev_note="A0 delta +0.011 而 BA 仅 0.4523（低于随机）；A2 的 BA 增益 +0.110 已被 "
             "L2 对照和标定四条否掉",
    **locked_unread("capillary_count"),
    disposition="无信号；被**设备标定禁令**阻断，转入标定与分母定义",
    alternative_info="“画面内可见管袢数”须作为**新字段另评估**，见 new_field_2；"
                     "实测 rho 0.370（CI 排除 0）、本地检测器 0.334 同向 → 有关联但"
                     "**不等价于条/mm：缺长度分母、缺标定、缺视野元数据**",
    rag="unknown，不进建议")

# ---- 9~12. the four measurements -------------------------------------------
for f, cn in (("afferent_diameter", "输入枝管径"), ("efferent_diameter", "输出枝管径"),
              ("apex_diameter", "顶端管径"), ("loop_length", "管袢长度")):
    d = dev_cell(f)
    row(field=f, model=DINO + "（A0）",
        target="三分类档位；**不得输出微米值**（device_calibration_status.json = "
               "UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION）",
        **d,
        dev_note="BA %.4f，%s；标签只有 3~5 档且档间距 < 我们的 MAE" % (
            d["dev_ba"], "低于随机" if d["dev_ba"] < 0.5 else "接近随机"),
        **locked_unread(f),
        disposition="无信号；四个测量字段全关",
        alternative_info="无；比值 output_input_ratio 是恒等式反算值，不是独立观测",
        rag="unknown / not_modelled，不进建议")

# ---- 13. flow_state ---------------------------------------------------------
row(field="flow_state",
    model=DINO + "（A0）",
    target="二分类 流态；**单位是时间基**，静态图不可观测",
    **dev_cell("flow_state"),
    dev_note="delta −0.0112（负），BA 0.4928，AUROC 0.5090 —— 与随机不可区分",
    **locked_unread("flow_state"),
    video_reality="视频只覆盖 51.1% 病例；静态相关性**不能冒充流态观测**",
    disposition="无信号；需时间观测",
    alternative_info="无等价静态量",
    rag="unknown，不进建议")

# ---- 14. microthrombus ------------------------------------------------------
row(field="microthrombus",
    model=DINO + "（A0）",
    target="二分类 无 vs 有；**单位是时间基上的事件**（个/min），静态图不可观测",
    **dev_cell("microthrombus"),
    dev_note="development delta +0.077 / BA 0.6542 是**静态外观相关**，不是事件计数",
    **locked_unread("microthrombus"),
    disposition="**暂停**；**不能固定为“无”**（development 分布 108 无 / 45 “1--2” / "
                "29 “>2”，固定为无会在 74 例上错）",
    alternative_info="无；这是 v2 未覆盖的唯一空档",
    rag="unknown，且**不得驱动任何血栓相关建议**")

# ---- 15. rbc_aggregation ----------------------------------------------------
row(field="rbc_aggregation",
    model=DINO + "（A0）",
    target="二分类 红细胞聚集；原始定义由**流动上下文**分级",
    **dev_cell("rbc_aggregation"),
    dev_note="delta −0.011（负），BA 0.4933 —— 准确率 0.8177 全部来自 0.8287 的众数",
    **locked_unread("rbc_aggregation"),
    observability_note="静态图能否观察该目标：聚集是在流动中判读的，静态帧只能看到"
                       "血柱外观，**不是同一个观测**",
    disposition="无信号；固定为众数",
    alternative_info="无",
    rag="unknown / not_modelled，不进建议")

# ---- 16~19. fixed-mode fields ----------------------------------------------
row(field="vasomotion", target="舒缩活动；固定输出", **fixed_mode_field("vasomotion", "0--1"),
    locked_n_note="v1 兼容率 0.9149（47 例）")
row(field="wbc_count", target="白细胞数；固定输出", **fixed_mode_field("wbc_count", "1--30"),
    locked_n_note="v1 兼容率 0.9783（46 例）")
row(field="sweat_duct", target="汗腺导管；固定输出", **fixed_mode_field("sweat_duct", "0--2"),
    locked_n_note="v1 兼容率 1.0（44 例）；n=186 下算术上不可能建模")
row(field="hemorrhage", target="出血；固定输出", **fixed_mode_field("hemorrhage", u"无"),
    locked_n_note="v1 兼容率 0.5（47 例，二分类退化）",
    correction=("⚠️ development 上**并非恒为“无”**：168 无 / 12 “1--2” / 5 单位串。"
                "固定为“无”会在 12 例上错，这是覆盖率取舍，不是实测准确率"),
    alternative_info="“局部可疑红点”须作为**新字段另评估**，见 new_field_3："
                     "磁盘上**没有任何检测器输出红点类别**，无量可评；要做需要区域标注，"
                     "而我们持有 0 份")

# ---- 20. flow_speed_um_s ----------------------------------------------------
row(field="flow_speed_um_s", model="无模型", target="流速；**原标签为空**",
    dev_n=0, locked_n=0, locked_source=None,
    disposition="删除", alternative_info="无", rag="not_modelled")

# ---- 21. output_input_ratio -------------------------------------------------
row(field="output_input_ratio", model="无独立模型",
    target="输出/输入管径比；**派生量**，由两个管径反算，不独立预测",
    dev_n=None, locked_n=None, locked_source=None,
    disposition="不独立预测", alternative_info="恒等式，非独立观测", rag="not_modelled")

assert len(ROWS) == 21, "expected 21 rows, got %d" % len(ROWS)

# ---- the three new fields, as separately assessed quantities ---------------
NEW_FIELDS = {
    "new_field_1_auto_crossing_box_count": newfield["auto_crossing_box_count"],
    "new_field_2_visible_loop_count_in_frame": newfield["visible_loop_count_in_frame"],
    "new_field_3_locally_suspicious_red_dots": newfield["locally_suspicious_red_dots"],
    "second_opinion_local_detector": newfield["second_opinion_local_detector"],
}

out = {
    "run": "field_table_21_20260925",
    "what_this_is": "研发结果汇总表，逐格可追溯到冻结产物。**不是上线声明**",
    "locked_cases_seen_this_run": 0,
    "locked_budget": {
        "reads_so_far": 7,
        "runs_that_selected_a_model_on_locked": 3,
        "status": "已消费的内部留出，不是干净最终测试集",
        "post_freeze_read_in_this_round": "已用于 malformation_ratio (n=39)",
        "consequence": ("表中 14 个字段的 locked 列标为 %s。这不是“未知”，是"
                        "“当前授权下不可评估”" % NOT_AUTH),
    },
    "shipped_configuration": {
        "poolings": ["mean", "topk_mean", "max", "cls", "std"],
        "C": 0.03, "pca_dim": 64, "seed": 20260917, "n_boot": 2000,
        "baseline": "single development-mode constant answer, never class_weight=balanced",
        "split": REV + " :development_fold (NaN == locked-47)",
    },
    "rows": ROWS,
    "new_fields_assessed_separately": NEW_FIELDS,
    "sources": {
        "development_15_field_matrix": MATRIX,
        "local_feature_ladder": LADDER,
        "a2x_frozen_candidate": FROZEN_A2X,
        "locked_read_20260918_five_fields": DELIVERED,
        "locked_read_20260924_malformation_a2x": CONSUMED,
        "locked_label_availability_only": V1_LOCKED,
        "frozen_field_routing": ROUTING,
        "new_field_probe": NEWFIELD,
    },
    "limitations": [
        "development 结果**不得**作为产品能力报告",
        "BA 对 2026-09-18 那五个字段是从已存混淆矩阵**推算**的，不是新读取",
        "AUROC 和 70% 覆盖准确率对那五个字段**不可推算**：该次运行未存逐病例概率",
        "n=39~47 时 bootstrap CI 半宽约 ±0.14~0.21，CI 含 0 意味着**未判定**，"
        "不是“接近通过”",
        "单中心单队列，全部不是外部验证",
        "禁止输出微米、条/mm、个/min 等未标定数值",
    ],
}

write("field_table_21.json", out)

flat = pd.DataFrame(ROWS)
flat.to_csv(os.path.join(OUT, "field_table_21.csv"), index=False, encoding="utf-8-sig")
print("rows:", len(ROWS), " columns:", len(flat.columns))
print("locked filled from real reads:",
      sum(1 for r in ROWS if r.get("locked_source")))
print("locked NOT_AUTHORISED:",
      sum(1 for r in ROWS if r.get("locked_acc") == NOT_AUTH))
print("wrote", OUT)
