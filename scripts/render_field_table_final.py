# -*- coding: utf-8 -*-
"""Render the FINAL 21-row delivery table with the 22 columns requested.

Presentation only. This script reads the already-built field_table_21.json plus
the frozen LOAO artifacts and writes a display table. It opens no locked image,
trains nothing, and rewrites no existing evaluation.

Column discipline, which is the point of the file:
  * a locked cell may only hold a number that came from a locked read that already
    happened; five fields + malformation_ratio A2x qualify
  * a locked read that stored no per-case probabilities cannot yield AUROC or a
    coverage curve -> NOT_AVAILABLE_FROM_STORED_ARTIFACT
  * a field never read on locked -> n is NOT_AUTHORISED_TO_READ, metrics NOT_AVAILABLE
  * fixed / deleted / derived fields -> NOT_APPLICABLE
  * development and LOAO numbers are labelled as such and never migrate into a
    locked column
"""
import io
import json
import os

import pandas as pd

SRC = "artifacts/audits/field_table_21_20260925/field_table_21.json"
BY_ARCH = ("artifacts/experiments/medical_encoder_transfer_20260921/"
           "field_matrix_three_arms_by_archive.csv")
FROZEN_A2X = ("artifacts/experiments/rescue_external_20260922/"
              "frozen_candidate_malformation_a2x/frozen_candidate.json")
OUT = "artifacts/audits/field_table_21_final_20260925"
os.makedirs(OUT, exist_ok=True)

NA_STORE = "NOT_AVAILABLE_FROM_STORED_ARTIFACT"
NA = "NOT_AVAILABLE"
NOT_AUTH = "NOT_AUTHORISED_TO_READ"
NOT_APP = "NOT_APPLICABLE"

src = json.load(io.open(SRC, encoding="utf-8"))
rows_in = {r["field"]: r for r in src["rows"]}
probes = src["new_fields_assessed_separately"]
frozen = json.load(io.open(FROZEN_A2X, encoding="utf-8"))

arch = pd.read_csv(BY_ARCH, encoding="utf-8")
arch = arch[arch.arm == "anchor_dinov2b_deployed"]


def loao_dev(field):
    """LOAO as the pooled-OOF partition by archive: every archive was in training
    when its own cases were predicted, so this is NOT held-out-archive training."""
    d = arch[arch.field == field]
    if d.empty:
        return NA
    parts = ["%s n=%d BA=%.4f" % (r.archive.replace("recovered_archive", "A"),
                                  r.n, r.ba) for _, r in d.iterrows()]
    return "development LOAO(池化OOF按档案切分): " + "; ".join(parts)


def loao_a2x():
    r = frozen["loao_refit_archive_held_out"]["by_held_out_archive"]
    parts = []
    for k, v in r.items():
        parts.append("%s n=%d BA=%.4f Δ=%+.4f" % (
            k.replace("recovered_archive", "A"), v["n"],
            v["balanced_accuracy"], v["delta_vs_training_archive_baseline"]))
    return "development 重训式LOAO(档案完全退出训练): " + "; ".join(parts)


def dev_str(r):
    if r.get("dev_n") in (None, 0):
        return NOT_APP
    bits = ["n=%s" % r["dev_n"]]
    if r.get("dev_acc") is not None:
        bits.append("Acc=%.4f" % r["dev_acc"])
    if r.get("dev_baseline") is not None:
        bits.append("基线=%.4f" % r["dev_baseline"])
    if r.get("dev_delta") is not None:
        bits.append("Δ=%+.4f" % r["dev_delta"])
    if r.get("dev_ba") is not None:
        bits.append("BA=%.4f" % r["dev_ba"])
    if r.get("dev_auroc") is not None:
        bits.append("AUROC=%.4f" % r["dev_auroc"])
    return "development OOF(非locked): " + " ".join(bits)


COLS = ["字段", "locked可评估n", "locked结果来源", "最终模型/结构", "目标定义",
        "Acc", "locked众数基线", "Acc−基线", "BA", "AUROC", "每类召回",
        "混淆矩阵", "95%CI", "70%覆盖准确率", "development OOF", "LOAO",
        "字段状态", "是否进入RAG", "RAG身份", "允许建议", "禁止建议",
        "不可替代原因/可替代信息"]

T = []


def add(**kw):
    T.append({c: kw.get(c, "") for c in COLS})


DINO = "DINOv2-B/14 整图 + 5池化 + PCA64 + LogReg(C=0.03)"
LOCKED18 = "locked_delivered_20260918（A0部署配置，历史已消费读取）"
FORBID_ALL = "不得作出任何疾病诊断、排除或分期断言"

# ============ 1. clarity =====================================================
r = rows_in["clarity"]
add(**{"字段": "clarity（清晰度）", "locked可评估n": 46, "locked结果来源": LOCKED18,
       "最终模型/结构": DINO, "目标定义": "二分类 清晰 vs 欠清/模糊；静态显微原图，病例级",
       "Acc": "0.7391", "locked众数基线": "0.6522", "Acc−基线": "+0.0870",
       "BA": "0.7271（由已存混淆矩阵推算）", "AUROC": NA_STORE,
       "每类召回": "类0(清晰)=0.767 / 类1(异常)=0.688",
       "混淆矩阵": "[[23,7],[5,11]]", "95%CI": "Δ的CI=[-0.0875,+0.2609]（含0，未判定）",
       "70%覆盖准确率": NA_STORE, "development OOF": dev_str(r),
       "LOAO": loao_dev("clarity"),
       "字段状态": "signal_present_not_clinically_validated（Tier1；locked上Δ的CI含0）",
       "是否进入RAG": "yes", "RAG身份": "reliable_observation",
       "允许建议": "仅说明本次图像可解读性；可作为其他字段的门控（图像不清时令其拒答）",
       "禁止建议": FORBID_ALL + "；不得把“清晰”表述为健康、把“模糊”表述为病变",
       "不可替代原因/可替代信息": "无替代量。这是图像质量门，不是临床发现，"
                                "因此不存在等价临床替代字段"})

# ============ 2. exudation ===================================================
r = rows_in["exudation"]
add(**{"字段": "exudation（渗出）", "locked可评估n": 47, "locked结果来源": LOCKED18,
       "最终模型/结构": DINO, "目标定义": "二分类 无 vs +/++/+++；静态显微原图，病例级",
       "Acc": "0.5319", "locked众数基线": "0.5319", "Acc−基线": "+0.0000",
       "BA": "0.5355（由已存混淆矩阵推算）", "AUROC": NA_STORE,
       "每类召回": "类0=0.591 / 类1=0.480", "混淆矩阵": "[[13,9],[13,12]]",
       "95%CI": "Δ的CI=[-0.1915,+0.2128]（含0，未判定）",
       "70%覆盖准确率": NA_STORE, "development OOF": dev_str(r),
       "LOAO": loao_dev("exudation"),
       "字段状态": "development_signal_not_reproduced_on_locked"
                   "（development Δ=+0.2404，locked Δ=0.0000）",
       "是否进入RAG": "correlation_only", "RAG身份": "image_correlation",
       "允许建议": "仅可作为图像观察陈述，须同时标注“留出集上未复现”",
       "禁止建议": FORBID_ALL + "；不得据此提示炎症、感染或任何病因",
       "不可替代原因/可替代信息": "无替代量；无其他字段可代为表达渗出"})

# ============ 3. SVP =========================================================
r = rows_in["subpapillary_venous_plexus"]
add(**{"字段": "subpapillary_venous_plexus（乳头下静脉丛）", "locked可评估n": 47,
       "locked结果来源": LOCKED18, "最终模型/结构": DINO,
       "目标定义": "二分类 未见 vs 可见；静态显微原图，病例级",
       "Acc": "0.7872", "locked众数基线": "0.6596", "Acc−基线": "+0.1277",
       "BA": "0.7782（由已存混淆矩阵推算）", "AUROC": NA_STORE,
       "每类召回": "类0=0.750 / 类1=0.807", "混淆矩阵": "[[12,4],[6,25]]",
       "95%CI": "Δ的CI=[-0.0426,+0.2979]（含0，未判定）",
       "70%覆盖准确率": NA_STORE, "development OOF": dev_str(r),
       "LOAO": loao_dev("subpapillary_venous_plexus"),
       "字段状态": "signal_present_not_clinically_validated（Tier2；CI含0）",
       "是否进入RAG": "correlation_only", "RAG身份": "image_correlation",
       "允许建议": "无；仅研究提示，不进入面向用户的建议",
       "禁止建议": FORBID_ALL + "；不得提示静脉回流障碍或循环疾病",
       "不可替代原因/可替代信息": "无替代量。留一档案外测显示该字段是档案依赖的，"
                                "稳定性未确立"})

# ============ 4. blood_color =================================================
r = rows_in["blood_color"]
add(**{"字段": "blood_color（血色）", "locked可评估n": 44, "locked结果来源": LOCKED18,
       "最终模型/结构": DINO, "目标定义": "二分类 正常 vs 暗红/淡红；静态显微原图，病例级",
       "Acc": "0.6364", "locked众数基线": "0.5227", "Acc−基线": "+0.1136",
       "BA": "0.6356（由已存混淆矩阵推算）", "AUROC": NA_STORE,
       "每类召回": "类0=0.652 / 类1=0.619", "混淆矩阵": "[[15,8],[8,13]]",
       "95%CI": "Δ的CI=[-0.0909,+0.3182]（含0，未判定）",
       "70%覆盖准确率": NA_STORE, "development OOF": dev_str(r),
       "LOAO": loao_dev("blood_color"),
       "字段状态": "signal_present_not_clinically_validated（Tier2；CI含0）",
       "是否进入RAG": "correlation_only", "RAG身份": "image_correlation",
       "允许建议": "无；仅研究提示",
       "禁止建议": FORBID_ALL + "；不得提示缺氧、贫血或血液疾病",
       "不可替代原因/可替代信息": "无替代量；颜色受光源与白平衡影响，未做设备间校正，"
                                "档案依赖"})

# ============ 5. malformation_ratio (A2x) ====================================
r = rows_in["malformation_ratio"]
add(**{"字段": "malformation_ratio（畸形比例）", "locked可评估n": "39（47例中8例无标签）",
       "locked结果来源": "locked_consumed_20260924（A2x冻结后单次读取；"
                        "consumed internal holdout descriptive evaluation）",
       "最终模型/结构": "A2x = DINOv2-B 整图特征 + HF 外部形态检测器自动局部特征"
                       "（832维；局部区域由训练折产生的检测器自动提取）",
       "目标定义": "二分类 `<=10%` vs `>10%`",
       "Acc": "0.6154", "locked众数基线": "0.5385（locked自身众数）",
       "Acc−基线": "+0.0769",
       "BA": "0.6310", "AUROC": "0.7249",
       "每类召回": "类0(≤10%)=0.8333 / 类1(>10%)=0.4286",
       "混淆矩阵": "[[15,3],[12,9]]",
       "95%CI": "A2x−A0 的 BA 增益=+0.0476，95%CI=[-0.0870,+0.1818]（含0，未判定）",
       "70%覆盖准确率": "0.6667（locked，27/39）；development 同一门槛 0.7876",
       "development OOF": dev_str(r) + " ｜ A2x: Acc=0.7284 Δ=+0.1605 BA=0.7216 "
                                      "AUROC=0.7766",
       "LOAO": loao_a2x(),
       "字段状态": "strong_development_candidate_not_clinically_validated",
       "是否进入RAG": "correlation_only", "RAG身份": "image_correlation",
       "允许建议": "仅允许四值输出：`low_band_correspondence` / "
                 "`high_band_correspondence` / `unknown` / `rejected`",
       "禁止建议": "禁止输出百分比、条/mm、微米；禁止任何疾病判断；"
                 "**不得用于阴性排除**；不得写成已经上线",
       "不可替代原因/可替代信息": "自动检测形态统计是相关特征，**不等价于医生百分比**"
                                "（人工框仅17/44落在自己档内，约3倍尺度偏移）"})

# ============ unread-on-locked fields ========================================
UNREAD = "未经locked读取（当前授权不允许第8次读取）"


def unread(field, name, model, target, status, rag, ident, allow, forbid,
           alt, dev_extra="", loao_field=None):
    r = rows_in[field]
    n_avail = r.get("locked_n_labelled_available")
    add(**{"字段": name, "locked可评估n": NOT_AUTH,
           "locked结果来源": UNREAD + (
               "；（参考：2026-08 v1 那次已消费读取中该字段有 %s 例带标签，"
               "仅为标签可得性，不是模型成绩）" % n_avail if n_avail else ""),
           "最终模型/结构": model, "目标定义": target,
           "Acc": NA, "locked众数基线": NA, "Acc−基线": NA, "BA": NA, "AUROC": NA,
           "每类召回": NA, "混淆矩阵": NA, "95%CI": NA, "70%覆盖准确率": NA,
           "development OOF": dev_str(r) + dev_extra,
           "LOAO": loao_dev(loao_field or field),
           "字段状态": status, "是否进入RAG": rag, "RAG身份": ident,
           "允许建议": allow, "禁止建议": forbid, "不可替代原因/可替代信息": alt})


# ---- 6. papilla
unread("papilla", "papilla（乳头）", DINO + "（A0）",
       "三分类 乳头形态（唯一必须保留3类的字段）；静态显微原图",
       "no_signal_model_side_paused（A0 Δ=+0.0270 而 BA=0.4149 低于随机）",
       "correlation_only", "image_correlation", "无；仅研究提示",
       FORBID_ALL,
       "检测器局部特征无增益（A2x BA增益 −0.0152）。"
       "**不得采用已失败的 A1+A2 融合**：A1+A2 比 A1 差 −0.0224，三条件全不过",
       dev_extra=" ｜ A2 BA=0.4567，A2x BA=0.3997（四臂BA全在0.40~0.46）")

# ---- 7. crossing_ratio
unread("crossing_ratio", "crossing_ratio（交叉比例）", DINO + "（A0）",
       "二分类 `<=30%` vs `>30%`；**不得输出实际比例**",
       "no_signal_reference_standard_must_be_redone（A0 Δ=−0.0113 为负）",
       "no", "unknown", "无",
       FORBID_ALL + "；禁止输出实际比例",
       "自动 crossing 框数见附表探针1：rho=0.105，CI含0，且本地检测器给 −0.036"
       "（符号相反）→ **不是原字段的等价替代**。瓶颈在我们自己的标注："
       "cross_vessel 检测器 AP 在五折上上限 0.3156",
       dev_extra=" ｜ 最好的 A2x 也只有 Δ=+0.0169 BA=0.5812")

# ---- 8. capillary_count
unread("capillary_count", "capillary_count（管袢数，条/mm）", DINO + "（A0）",
       "三分类 条/mm 档位；**不得输出条/mm 数值**",
       "blocked_by_device_calibration（A0 Δ=+0.0110 而 BA=0.4523 低于随机）",
       "no", "unknown", "无",
       FORBID_ALL + "；禁止输出条/mm、微米等未标定数值",
       "“画面内可见管袢数”见附表探针2：rho=0.370（CI排除0）、本地检测器0.334同向，"
       "**但缺长度分母、设备UNCALIBRATED、无视野元数据 → 不等价于 capillary_count "
       "或条/mm**",
       dev_extra=" ｜ A2 的 BA 增益 +0.1104 已被 L2 对照与标定四条否掉")

# ---- 9~12. measurements
MEAS = {"afferent_diameter": "afferent_diameter（输入枝管径）",
        "efferent_diameter": "efferent_diameter（输出枝管径）",
        "apex_diameter": "apex_diameter（顶端管径）",
        "loop_length": "loop_length（管袢长度）"}
for f, name in MEAS.items():
    r = rows_in[f]
    unread(f, name, DINO + "（A0）",
           "三分类档位；**不得输出微米值**"
           "（device_calibration_status.json = UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION）",
           "no_signal_label_granularity_ceiling（BA=%.4f，标签仅3~5档且档间距 < 我们的MAE）"
           % r["dev_ba"],
           "no", "unknown", "无",
           FORBID_ALL + "；禁止输出微米绝对值",
           "无等价替代。output_input_ratio 由两个管径反算，是恒等式而非独立观测；"
           "四个测量字段全部关闭")

# ---- 13. flow_state
unread("flow_state", "flow_state（流态）", DINO + "（A0）",
       "二分类 流态；**观测单位是时间基**，静态图不可观测",
       "no_signal_requires_time_base（Δ=−0.0112 为负，BA=0.4928，AUROC=0.5090）",
       "no", "unknown", "无",
       FORBID_ALL + "；静态相关性**不得冒充流态观测**",
       "无等价静态量。视频仅覆盖 51.1% 病例，且本字段需按秒计的动态观测")

# ---- 14. microthrombus
unread("microthrombus", "microthrombus（微血栓）", DINO + "（A0）",
       "二分类 无 vs 有；**单位是时间基上的事件（个/min）**，静态图不可观测",
       "paused_requires_time_base（development Δ=+0.0769 是静态外观相关，不是事件计数）",
       "unknown", "unknown", "无",
       FORBID_ALL + "；**不得驱动任何血栓相关建议**；不得输出个/min",
       "**不能固定为“无”**：development 分布为 108 无 / 45 “1--2” / 29 “>2”，"
       "固定为“无”将在 74 例上错。这是 v2 未覆盖的唯一空档")

# ---- 15. rbc_aggregation
unread("rbc_aggregation", "rbc_aggregation（红细胞聚集）", DINO + "（A0）",
       "二分类 聚集；原始定义由**流动上下文**分级",
       "no_signal_collapsed_to_mode（Δ=−0.0110 为负，BA=0.4933；"
       "Acc=0.8177 全部来自 0.8287 的众数）",
       "no", "not_modelled", "无",
       FORBID_ALL,
       "静态图能否观察该目标：聚集是在**流动中**判读的，静态帧只能看到血柱外观，"
       "**不是同一个观测**。固定为众数，无等价替代")

# ============ 16~19. fixed-mode fields =======================================
def fixed(field, name, value, target, status, alt, extra_forbid=""):
    r = rows_in[field]
    dist = r["dev_distribution"]
    dist_s = " / ".join("%s:%d" % (k, v) for k, v in
                        sorted(dist.items(), key=lambda kv: -kv[1]))
    compat = r.get("locked_compatibility_rate")
    add(**{"字段": name, "locked可评估n": NOT_APP,
           "locked结果来源": NOT_APP + "（固定常量输出，无模型可评估；"
                          "2026-08 v1 记录的“兼容率”%s 是常量与报告的符合率，"
                          "**无模型参与，不得当作准确率**）" % compat,
           "最终模型/结构": "无 —— 固定常量输出", "目标定义": target,
           "Acc": NOT_APP, "locked众数基线": NOT_APP, "Acc−基线": NOT_APP,
           "BA": NOT_APP, "AUROC": NOT_APP, "每类召回": NOT_APP,
           "混淆矩阵": NOT_APP, "95%CI": NOT_APP, "70%覆盖准确率": NOT_APP,
           "development OOF": "无模型。development 标签分布：" + dist_s +
                              "；固定输出值 `%s`" % value,
           "LOAO": NOT_APP,
           "字段状态": status, "是否进入RAG": "no", "RAG身份": "unknown / not_modelled",
           "允许建议": "无",
           "禁止建议": FORBID_ALL + "；**不得把固定众数当作该患者的事实**（"
                     "这不是患者实测值）" + extra_forbid,
           "不可替代原因/可替代信息": alt})


fixed("vasomotion", "vasomotion（舒缩活动）", "0--1",
      "舒缩活动；固定常量输出", "not_modelled_fixed_constant",
      "无等价替代；舒缩是动态观测，静态图无对应量")
fixed("wbc_count", "wbc_count（白细胞数）", "1--30",
      "白细胞数；固定常量输出", "not_modelled_fixed_constant",
      "无等价替代；需按时间计数的动态观测")
fixed("sweat_duct", "sweat_duct（汗腺导管）", "0--2",
      "汗腺导管；固定常量输出",
      "not_modelled_insufficient_n（n=186 下算术上不可能建模）",
      "无等价替代")
fixed("hemorrhage", u"hemorrhage（出血）", u"无",
      u"出血；固定常量输出（现状）",
      "not_modelled_due_to_extreme_imbalance",
      u"⚠️ **不能写成“固定为无且准确率100%”**：development 中 168 例“无”、"
      u"**12 例“1--2”**，另有 5 条单位脏值（`管袢/一指甲襞`）。固定为“无”会在那 12 例"
      u"上直接错，这是覆盖率取舍，不是实测准确率。"
      u"“局部可疑红点”见附表探针3：**磁盘上没有任何检测器输出红点类别**，"
      u"无量可评（NOT_AVAILABLE），要做需区域标注而我们持有 0 份 → 无等价替代",
      extra_forbid=u"；不得据此提示出血、凝血或血管脆性问题")

# ============ 20. flow_speed_um_s ============================================
add(**{"字段": "flow_speed_um_s（流速）", "locked可评估n": NOT_APP,
       "locked结果来源": NOT_APP + "（原标签为空，n=0，无可评估对象）",
       "最终模型/结构": "无模型", "目标定义": "流速；**原标签为空**",
       "Acc": NOT_APP, "locked众数基线": NOT_APP, "Acc−基线": NOT_APP,
       "BA": NOT_APP, "AUROC": NOT_APP, "每类召回": NOT_APP, "混淆矩阵": NOT_APP,
       "95%CI": NOT_APP, "70%覆盖准确率": NOT_APP,
       "development OOF": "n=0，无标签，从未建模", "LOAO": NOT_APP,
       "字段状态": "deleted（删除）", "是否进入RAG": "no", "RAG身份": "not_modelled",
       "允许建议": "无", "禁止建议": FORBID_ALL + "；禁止输出 μm/s 数值",
       "不可替代原因/可替代信息": "无替代。流速需时间基观测且标签完全缺失"})

# ============ 21. output_input_ratio =========================================
add(**{"字段": "output_input_ratio（输出/输入管径比）", "locked可评估n": NOT_APP,
       "locked结果来源": NOT_APP + "（派生量，不独立预测）",
       "最终模型/结构": "无独立模型",
       "目标定义": "输出/输入管径比；**派生量**，由两个管径反算，不独立预测",
       "Acc": NOT_APP, "locked众数基线": NOT_APP, "Acc−基线": NOT_APP,
       "BA": NOT_APP, "AUROC": NOT_APP, "每类召回": NOT_APP, "混淆矩阵": NOT_APP,
       "95%CI": NOT_APP, "70%覆盖准确率": NOT_APP,
       "development OOF": "不独立建模（恒等式反算值）", "LOAO": NOT_APP,
       "字段状态": "derived_not_independently_predicted",
       "是否进入RAG": "no", "RAG身份": "not_modelled",
       "允许建议": "无", "禁止建议": FORBID_ALL,
       "不可替代原因/可替代信息": "本身就是派生量：由 afferent/efferent 反算，"
                                "而那两个字段均无信号且禁止输出微米 → 无可用来源"})

assert len(T) == 21, "expected 21 rows, got %d" % len(T)
df = pd.DataFrame(T, columns=COLS)
df.to_csv(os.path.join(OUT, "field_table_21_final.csv"),
          index=False, encoding="utf-8-sig")

# ---- probe appendix ---------------------------------------------------------
p1 = probes["new_field_1_auto_crossing_box_count"]
p2 = probes["new_field_2_visible_loop_count_in_frame"]
p3 = probes["new_field_3_locally_suspicious_red_dots"]
p2nd = probes["second_opinion_local_detector"]
APPENDIX = [
    {"探针": "探针1：自动 crossing 框数",
     "量的定义": "pooled_n_crossing（外部预训练4类检测器，病例内池化）",
     "是否原21字段": "不是 —— 这是新量，**不得伪装成 crossing_ratio 的准确率**",
     "人工金标准": "不存在（无人逐图标注交叉数；报告只给全检查档位）",
     "准确率": "不可计算，且不声称",
     "与临床档的Spearman rho": p1["vs_clinical_crossing_ratio_band"]["rho"],
     "95%CI": p1["vs_clinical_crossing_ratio_band"]["rho_ci95"],
     "CI是否排除0": p1["vs_clinical_crossing_ratio_band"]["ci_excludes_zero"],
     "本地检测器二次意见": p2nd["n_cross_vs_crossing_band"]["rho"],
     "结论": "**不等价**：CI 含 0，且两个独立检测器符号相反"},
    {"探针": "探针2：画面内可见管袢数",
     "量的定义": "pooled_n_total（同一检测器在画面内找到的管袢数）",
     "是否原21字段": "不是 —— 新量，**不等价于 capillary_count 或条/mm**",
     "人工金标准": "不存在（无人标注画面内应有几条）",
     "准确率": "不可计算，且不声称",
     "与临床档的Spearman rho": p2["vs_clinical_capillary_count_band"]["rho"],
     "95%CI": p2["vs_clinical_capillary_count_band"]["rho_ci95"],
     "CI是否排除0": p2["vs_clinical_capillary_count_band"]["ci_excludes_zero"],
     "本地检测器二次意见": p2nd["n_total_vs_count_band"]["rho"],
     "结论": "有关联（CI排除0，两检测器同向），**但缺长度分母、设备UNCALIBRATED、"
             "无视野元数据 → 不是等价替代，禁止输出条/mm**"},
    {"探针": "探针3：局部可疑红点",
     "量的定义": NA + " —— 磁盘上无任何检测器输出红点/出血类别",
     "是否原21字段": "不是",
     "人工金标准": "不存在", "准确率": NA,
     "与临床档的Spearman rho": NA, "95%CI": NA, "CI是否排除0": NA,
     "本地检测器二次意见": NA,
     "结论": "无量可评。外部检测器类别为 %s，本地为 %s，均无红点类；"
             "要做需区域标注，我们持有 0 份"
             % (p3["external_detector_classes"], p3["local_detector_classes"])},
]
pd.DataFrame(APPENDIX).to_csv(os.path.join(OUT, "probe_appendix.csv"),
                              index=False, encoding="utf-8-sig")

SUMMARY = {
    "locked上有真实历史结果的字段": 5,
    "未授权读取_不能补充locked指标的字段": 10,
    "不需要locked模型评估的固定_删除_派生字段": 6,
    "合计": 21,
    "达到_70pct覆盖_85pct准确率_交付门槛的字段数": 0,
    "门槛实测": {
        "malformation_ratio_development_70pct": 0.7876,
        "malformation_ratio_locked_70pct": 0.6667,
        "note": "提高拒答率也到不了 85%（development 50% 覆盖时 0.8148 已是全曲线最好）",
        "其余字段": "该次 locked 读取未存逐病例概率，覆盖曲线不可推算",
    },
    "locked_47的身份": "已消费的内部留出（consumed internal holdout），"
                      "**不是外部验证、不是临床验证、不是干净最终测试集**"
                      "（已被读取7次，其中3次在其上做过模型选择）",
    "纪律声明": [
        "development 数字**未**填入任何 locked 列",
        "无逐病例概率的历史 locked 结果，AUROC 与 70% 覆盖准确率填 "
        + NA_STORE + "，未用 development 数字替代",
        "本次运行 locked_cases_seen = 0：未读取、未解码、未提特征、未训练",
        "未修改任何既有模型或评估产物",
    ],
}
io.open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8").write(
    json.dumps(SUMMARY, ensure_ascii=False, indent=1))

print("rows:", len(df), "cols:", len(df.columns))
print("wrote", OUT)
for k, v in SUMMARY.items():
    if isinstance(v, int):
        print(" ", k, "=", v)

