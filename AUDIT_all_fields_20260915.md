# 全字段真实能力审计 — 2026-09-15

本文档记录**实测数字与复现方式**，不面向汇报，不做美化。凡我此前报错的数字，本文列出更正。

前置声明：本文取代 `REPORT_model_layer_20260913.md` 中的全部指标。那份文档有两处基线错误（见 §6），且只覆盖 5 个字段中的 4 个，未覆盖其余 15 个字段。

---

## 1. 结论摘要

- 标签共 102 列，其中**实质临床字段 20 个**。此前所有实验只覆盖 5 个，**15 个字段本次首测**。
- 20 个字段中**只有 5 个真正可用**：clarity、exudation、blood_color、subpapillary_venous_plexus、loop_length。加 periloop_score 勉强 6 个。
- **15 个字段不能上线。** 精确拆分（§15）：**9 个 delta 为负**，即模型不如"永远输出多数类"这条零成本规则，上模型是净损失，其中 5 个（rbc_aggregation、sweat_duct、hemorrhage、flow_state、vasomotion）CI 上界<0 统计显著；另 **6 个 delta 为正但 CI 含 0**，与众数规则统计上不可区分。「14 个跑不过众数」是我的措辞不严谨，正确说法见此。
- **LoRA 微调真实增益为 0**，r4 为统计显著负增益。该路线应关闭。
- `afferent_diameter` 在无偏协议下**掉出可用名单**（此前报 +0.170，实为 +0.067 且 CI 含 0）。
- 好成绩的主要来源是**类别压缩**：按医生原始档位，exudation 与 SVP 均无效，只有压成二分类后才有效。

---

## 2. 协议与口径

| 项 | 设定 |
|---|---|
| 数据 | 开发集 186 例；locked_test 47 例**全程未加载** |
| 折划分 | `development_fold`（0..4）为测试折，病例级，同一病例所有帧同折 |
| 基线 | 每折**训练集内部**的多数类占比（非全局众数） |
| 配置选择 | 内层验证折 = (test+1)%5，在验证折上选特征源与 PCA 维数，**不看测试集** |
| 特征源 | dino(768) / geom(394) / both / inst(A3 实例聚合) / inst_geom |
| 降维 | PCA 32 或 64 维（186 例样本下不降维必过拟合） |
| 分类器 | LogisticRegression C=0.1, class_weight=balanced, max_iter=4000 |
| 不确定性 | **病例级** bootstrap 2000 次；基线在每次重采样内重算 |
| 判定 | CI 下界>0 记 OK；CI 上界<0 记 DEAD；跨 0 记 null |

**本轮为单 seed**（折划分固定，逻辑回归对 random_state 不敏感），弱于 5-seed 嵌套协议一档，但配置选择无测试集偷看。两者不可混用比较。

**已知缺陷（必须一并记录）**：部分字段 bootstrap CI 上界几乎等于点估计（如 clarity +0.3189 CI 上界 +0.314、loop_length +0.2727 上界 +0.273、total_score +0.1730 上界 +0.173）。原因是重采样内基线被重算，delta 分布右侧被基线上移压住，导致 CI 不对称且上界失真。**下界仍可用于判定，上界不可引用。**

---

## 3. 全字段能力 — 二分类（产品形态）

切点：首档（正常）vs 其余（任何异常）。脚本 `audit_binary.py`，结果 `audit_binary.json`。

**字段计数对账（20 个实质字段 = 本表 15 + §4 的 5 个测量类字段 loop_length / afferent / efferent / apex / flow_state）。**
§4 表中的 total_score / periloop_score / flow_score / morphology_score 是四个**合成分**（`morphology+flow+periloop=total`，181 例中 180 例精确成立），不计入 20 个实质字段。

| 字段 | 特征源 | n | 阳性率 | 准确率 | 基线 | delta | CI下界 | 敏感度 | 特异度 | AUC | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clarity | dino | 185 | 50.8% | 0.7730 | 0.4541 | **+0.319** | +0.167 | 0.777 | 0.769 | 0.838 | **OK** |
| exudation | dino | 183 | 49.2% | 0.7432 | 0.4426 | **+0.301** | +0.136 | 0.756 | 0.731 | 0.809 | **OK** |
| blood_color | inst | 181 | 44.8% | 0.7624 | 0.5525 | **+0.210** | +0.116 | 0.765 | 0.760 | 0.791 | **OK** |
| subpapillary_venous_plexus | dino | 184 | 57.1% | 0.7174 | 0.5707 | **+0.147** | +0.049 | 0.724 | 0.709 | 0.794 | **OK** |
| microthrombus | inst | 182 | 40.7% | 0.6758 | 0.5934 | +0.082 | −0.017 | 0.689 | 0.667 | 0.739 | null |
| malformation_ratio | geom | 162 | 43.2% | 0.6296 | 0.5679 | +0.062 | −0.037 | 0.571 | 0.674 | 0.605 | null |
| papilla | dino | 185 | 77.3% | 0.7892 | 0.7730 | +0.016 | −0.054 | 0.853 | **0.571** | 0.760 | null |
| capillary_count | inst | 182 | 68.7% | 0.6703 | 0.6868 | −0.017 | −0.110 | 0.704 | 0.596 | 0.726 | null |
| crossing_ratio | dino | 177 | 36.7% | 0.6045 | 0.6328 | −0.028 | −0.119 | 0.477 | 0.679 | **0.577** | null |
| overall_assessment | both | 176 | 90.9% | 0.8807 | 0.9091 | −0.028 | −0.062 | 0.963 | **0.062** | 0.632 | null |
| wbc_count | dino | 184 | 2.2% | 0.9728 | 0.9783 | −0.005 | −0.016 | **0.000** | 0.994 | 0.390 | null |
| vasomotion | dino | 184 | 2.7% | 0.9565 | 0.9728 | −0.016 | −0.033 | **0.000** | 0.983 | 0.398 | null |
| rbc_aggregation | both | 181 | 82.9% | 0.7569 | 0.8287 | −0.072 | −0.138 | 0.827 | **0.419** | 0.617 | **DEAD** |
| sweat_duct | dino | 180 | 15.6% | 0.7722 | 0.8444 | −0.072 | −0.133 | 0.321 | 0.855 | 0.666 | **DEAD** |
| hemorrhage | dino | 185 | 9.2% | 0.8432 | 0.9081 | −0.065 | −0.108 | **0.118** | 0.917 | **0.496** | **DEAD** |

### 3.1 高准确率但不可用的字段（准确率指标的陷阱）

| 字段 | 准确率 | 为什么不可用 |
|---|---|---|
| overall_assessment | 0.881（全表第二高） | 特异度 **0.062**。90.9% 样本是"异常"，模型学会永远说异常。对"大致正常"者几乎全判异常。 |
| wbc_count | 0.973 | 敏感度 **0.000** —— 从未识别出任何一个阳性病例。 |
| vasomotion | 0.957 | 敏感度 **0.000**，AUC 0.398（低于抛硬币）。 |
| hemorrhage | 0.843 | 敏感度 0.118，AUC **0.496** = 抛硬币。 |
| papilla | 0.789 | 特异度 0.571，delta +0.016 CI 含 0，赢不过众数。 |

**这五个字段若按准确率排名会进前列，实际全部不可上线。** 任何汇报只给准确率都会误导。

---

## 4. 测量与评分字段 — p50 中位切分

脚本 `audit_honest.py`，结果 `audit_honest.json`。

| 字段 | 特征源 | n | 准确率 | 基线 | delta | CI下界 | 判定 |
|---|---|---|---|---|---|---|---|
| loop_length | dino | 176 | 0.7273 | 0.4545 | **+0.273** | +0.114 | **OK** |
| total_score | inst_geom | 185 | 0.6270 | 0.4541 | +0.173 | +0.011 | 勉强 |
| periloop_score | both | 182 | 0.7033 | 0.5495 | **+0.154** | +0.049 | **OK** |
| flow_score | inst | 183 | 0.6503 | 0.5464 | +0.104 | +0.005 | 勉强 |
| morphology_score | both | 185 | 0.6108 | 0.4486 | +0.162 | −0.016 | null |
| efferent_diameter | inst | 164 | 0.6220 | 0.5427 | +0.079 | −0.030 | null |
| afferent_diameter | inst | 165 | 0.5879 | 0.5212 | +0.067 | −0.042 | null |
| apex_diameter | inst | 169 | 0.5976 | 0.5740 | +0.024 | −0.077 | null |
| flow_state（8 类原样） | dino | 184 | 0.2228 | 0.2772 | −0.054 | −0.212 | **DEAD** |

三个管径字段（afferent / efferent / apex）**全部无效**。此前"afferent 可交付"的结论不成立。

---

## 5. 类别压缩效应 — 好成绩的真实来源

同一字段、同一特征、同一协议，只改类别数：

| 字段 | 医生原始档位 | 多分类 acc / delta | 二分类 acc / delta |
|---|---|---|---|
| clarity | 3 | 0.6216 / +0.130 (OK) | 0.7730 / +0.319 (OK) |
| blood_color | 4 | 0.5967 / +0.160 (OK) | 0.7624 / +0.210 (OK) |
| exudation | 4 | 0.5574 / +0.049 **(CI含0)** | 0.7432 / +0.301 (OK) |
| subpapillary_venous_plexus | 4 | 0.4783 / +0.049 **(CI含0)** | 0.7174 / +0.147 (OK) |
| papilla | 3 | 0.5135 / +0.097 (CI含0) | 0.7892 / +0.016 (CI含0) |

**按医生报告单上实际填写的档位，exudation 和 SVP 都是无效的**，只在压成"有/无"后才有效。

产品含义：只能回答"有/无渗出"，不能回答"几个加号"；只能回答"静脉丛可见/不可见"，不能回答"可见几排"。而后者是医生实际书写的内容。

---

## 6. 我此前报错的数字（更正表）

### 6.1 基线错误导致 delta 虚高

从 `frozen_cls_patch_nested_predictions.csv.gz`（4590 行）逐格重算：

| 字段 | 我曾报的基线 | 真实基线 | 我曾报的 delta | 真实 delta |
|---|---|---|---|---|
| clarity | 0.4541 | **0.5081** | +0.3449 | **+0.2908** |
| exudation | 0.4426 | **0.5082** | +0.3038 | **+0.2383** |

准确率本身无误（0.7989 / 0.7464 逐位一致），错的是基线。4 字段均值 delta 从 +0.269 降到 **+0.239**。

### 6.2 afferent_diameter 此前结论不成立

| 版本 | delta | 判定 | 问题 |
|---|---|---|---|
| 我曾报 | +0.170（5/5 seed） | 可交付 | 在 A3/B × 3 个 C 值中选择，选择依据部分基于测试表现 |
| 本次无偏 | **+0.067**，CI[−0.042,+0.133] | **无效** | 配置在验证折上选 |

### 6.3 不可引用的 json 字段

`frozen_cls_patch_results.json` 头部 `mean_ba_0_5 = 0.7672`，但从原始预测重算 5 字段 BA 均值为 **0.7490**（剔除 papilla 为 0.7737）。三个数互不一致，该字段口径不明。**禁止引用 0.7672。**

### 6.4 无效的 CI

服务器版 `all_fields_audit.py` 输出形如 `ci[+0.151,+0.151]` 的 CI —— 对 5 个 seed 的 delta 做 bootstrap，而 5 个 seed 的 delta 完全相同（折固定 + 逻辑回归对 seed 不敏感），bootstrap 退化为一个点。**该脚本产出的所有 CI 无信息量，已弃用。** 本文 CI 均改为病例级 bootstrap。

### 6.5 进度误报

我曾报"LoRA 实验正常推进，预计 2026-09-14 出结果"。实际两个进程于 **2026-09-13 19:06 / 19:14 崩溃**（见 §8），而我做健康检查的时间是 18:55，仅 11 分钟之后。我未核对日期就沿用了旧结论。

---

## 7. LoRA 真实增益

分类字段，5 seed × 5 fold，**按 seed+字段配对**比较（25 对），基准 = frozen_cls_patch：

| 配置 | 配对均值 | 95% CI | 赢的对数 |
|---|---|---|---|
| lora_r16 | −0.0017 | [−0.0133, +0.0105] | 9/25 |
| lora_r8 | −0.0077 | [−0.0196, +0.0054] | 8/25 |
| lora_r4 | **−0.0195** | **[−0.0361, −0.0018]** | 6/25 |
| frozen_cls（无 patch） | −0.0199 | [−0.0338, −0.0051] | 4/25 |

逐 seed 5 字段均值准确率：

| 配置 | s17 | s42 | s123 | s456 | s789 | 均值 |
|---|---|---|---|---|---|---|
| **frozen_cls_patch** | 0.7547 | 0.7755 | 0.7756 | 0.7603 | 0.7430 | **0.7618** |
| lora_r16 | 0.7700 | 0.7484 | 0.7678 | 0.7581 | 0.7559 | 0.7601 |
| lora_r8 | 0.7624 | 0.7526 | 0.7537 | 0.7526 | 0.7493 | 0.7541 |
| lora_r4 | 0.7558 | 0.7341 | 0.7329 | 0.7450 | 0.7437 | 0.7423 |

**增益为 0；r4 统计显著为负（CI 上界 −0.0018 不含 0）。** 冻结 DINOv2 patch 特征 + 线性头优于全部 LoRA 配置。

测量字段（LoRA 三档序数，仅 seed 17，实验崩溃）：afferent **+0.0606** vs 冻结几何 5-seed **+0.1515** —— LoRA 差 9 个点。

逐 seed 波动幅度 0.027~0.081（clarity 0.081、blood_color 0.072）。**任何小于 0.08 的字段级改进在噪声内**（此前我说 0.05，偏乐观）。

结论：**LoRA 路线关闭。** 三个 rank × 5 seed 已跑死，不是超参问题。

---

## 8. LoRA 测量实验的崩溃记录

| 项 | r8 | r4 |
|---|---|---|
| 崩溃时间 | 2026-09-13 19:06 | 2026-09-13 19:14 |
| 位置 | `m_lora_ordinal.py:192` `cum_targets(...).cuda()` | `timm/layers/attention.py:119` qkv linear |
| 报错 | `CUDA error: unspecified launch failure` | `CUDA error: unknown error` |
| 完成进度 | 8/25 seed-fold | 9/25 seed-fold |
| 落盘 | 仅 seed 17 | 仅 seed 17 |

均为运行中 GPU 被抽走的表现，非代码缺陷（无 OOM、无 NaN、无断言失败）。`locked_cases_seen = 0` 已确认。

seed 17 单种子三档序数结果（**单 seed，不可作结论**）：

| 字段 | r4 delta | r4 CI | r8 delta | r8 CI |
|---|---|---|---|---|
| afferent | +0.0606 | [−0.042,+0.158] | +0.0606 | [−0.030,+0.152] |
| efferent | +0.0000 | [−0.104,+0.104] | +0.1280 | [+0.043,+0.220] |
| apex | +0.0592 | [−0.012,+0.130] | +0.0118 | [−0.089,+0.107] |
| loop_length | +0.1420 | [+0.040,+0.244] | +0.1023 | [−0.011,+0.210] |

r4/r8 在 efferent 上一个 0.0000 一个 +0.1280 —— 同 seed 下两配置差异如此之大，说明单 seed 数字不可靠。

预先声明的失败条件（两配置增益均<0.02 且 CI 含 0）未触发，成功条件亦未触发。**该实验状态：未完成、无结论。**

---

## 9. 低患病率下的 PPV — 真正的落地门槛

开发集患病率被人为富集到 44.8%~57.1%，真实平台人群不是这个分布。按各字段实测敏感度/特异度换算：

| 字段 | sens | spec | PPV@5% | PPV@10% | PPV@30% | NPV@5% |
|---|---|---|---|---|---|---|
| clarity | 0.777 | 0.769 | **0.150** | 0.271 | 0.590 | 0.985 |
| blood_color | 0.765 | 0.760 | **0.144** | 0.262 | 0.577 | 0.984 |
| exudation | 0.756 | 0.731 | **0.129** | 0.238 | 0.546 | 0.982 |
| subpapillary_venous_plexus | 0.724 | 0.709 | **0.116** | 0.216 | 0.516 | 0.979 |

**5% 患病率下报 100 个阳性，约 85 个是假阳性。** 准确率数字不变，产品可用性归零。

监管现实：**FDA、NMPA、EU MDR 均未规定 sensitivity/specificity/AUC 数值下限。** PMDA 已批准 151 个 SaMD（40 个用 AI），standalone 敏感度跨度 **67.7%~100%**，不存在统一门槛。门槛来自 PPV、外部验证与定位。

对照已通过 FDA 的 ECG-AI（低射血分数）：4 个地理分散中心、13,960 例、AUROC 0.92、敏感度 84.5%、特异度 83.6% —— 各项均强于我们；在 7.9% 患病率下 PPV 仍仅 **30.5%**，NPV 98.4%，故作者定位为 **rule-out 工具而非诊断工具**。

**我们唯一站得住的定位与之相同：** NPV 0.98~0.99 可用 → 阴性可说"本次未见异常"；阳性只能说"建议进一步检查"，不得出结论。

---

## 10. 外部验证：零

全部数字来自同一批 186 例、同一台设备、同一套标注流程的内部交叉验证。**locked-47 是同批数据留出，不构成外部验证。** 这是当前最大缺口，重于任何模型改进。

## 11. 文献对标的正确读法

| 来源 | 数字 | 为什么不可直接对标 |
|---|---|---|
| Yayla 2025, *Diagnostics* (PMID 41300936) | InceptionV3 准确率 **98.95%**，AUC 99~100% | 1280 图 / 50 SSc+30 健康人，按**图像级** 70/20/10 划分，测试集 70~100 张图来自与训练集**同一批 80 人** → 被试泄漏。我们是**病例级**划分。 |
| CAPoxy 2026, *PLoS One* (PMID 42585238) | 84% 准确率 | 22 人（10 SSc / 12 对照），作者自述 proof-of-concept，无 AUC/敏感度/特异度。 |
| Sieiro Santos 等综述 (PMID 42007790) | "达到专家级一致性" | 同时承认多为小样本单中心、缺外部验证。 |

**该领域深度学习文献中没有一项做过外部验证。**

## 12. 标签噪声天花板

| 来源 | 指标 | 数值 |
|---|---|---|
| Dinsdale 2017 (PMID 28163035)<br>10 位专家 / 7 个欧洲中心 / 173 人 | 图像"是否可判读"专家间 ICC | **0.14** |
| 同上 | 毛细血管密度专家间 ICC | **0.64** |
| 同上 | 平均顶端管径专家间 ICC | 0.85 |
| 同上 | 整体分级专家间 ICC | 0.78 |
| SCLEROCAP / Boulon 2017 (PMID 28957554)<br>385 例，双盲双评 | Maricq 分类专家间 κ | **0.47** (0.28,0.66)，共识后 0.64 |
| 同上 | Cutolo 分类专家间 κ | **0.49** (0.33,0.65)，共识后 0.69 |
| 同上 | 毛细血管密度可靠性评级 | 仅 "moderate" |

标签由 κ≈0.5 的人工判读产生 → **模型准确率 0.77 可能已接近标签噪声上限，继续优化模型不会有回报。** 这与"标签粒度天花板"诊断一致。

---

## 13. 脚本与产出清单

### 本地（RTX 5070，服务器无卡期间使用）

| 文件 | 作用 |
|---|---|
| `.tmp_probe/audit_local.py` | 特征加载 + `evaluate()` + 全字段跑批（**含测试集选配置偏差，仅作探索**） |
| `.tmp_probe/audit_honest.py` | 无偏版：配置在验证折选。产出 `audit_honest.json` |
| `.tmp_probe/audit_binary.py` | 二分类版（产品形态）。产出 `audit_binary.json` |
| `.tmp_probe/feat/` | 从服务器下载的特征（dinov2 2.6M / geometry 2.5M / A3 5.2M / B 2.3M / manifest 258K） |

复现：

```bash
cd .tmp_probe && python -u audit_honest.py && python -u audit_binary.py
```

注意 `audit_honest.py` 通过 `exec(open("audit_local.py",encoding="utf-8").read()...)` 复用前者定义，**必须显式指定 encoding**（Windows 默认 gbk 会在 CJK 处报 UnicodeDecodeError）。

### 服务器（`/root/nailfold/scripts/`）

| 文件 | 状态 |
|---|---|
| `all_fields_audit.py` | **已弃用**，CI 计算有缺陷（§6.4），且未降维在 1300 维上过拟合 |
| `m_lora_ordinal.py` | 端到端 LoRA 三档序数，需 GPU，已崩溃（§8）。缺陷：仅每 seed 结束落盘，中断丢当前 seed |
| `m_frozen_geom.py` / `m_frozen_ordinal.py` / `m_binary.py` / `m_report_detail.py` | CPU 可跑，无需 GPU |

### 已被本文取代的文档（已就地加更正标注，勿单独引用）

| 文件 | 失实之处 |
|---|---|
| `REPORT_model_layer_20260913.md` | §一「6 个字段可用」只覆盖 9 字段；§8「实验在跑今晚出结果」为假（已崩溃）；A1 表 clarity/exudation 基线低报；A2 表 afferent +0.170 含 test-argmax |
| memory `nailfold-deliverable-table` | delta 列基于错误基线 |
| memory `nailfold-measurement-ordinal-run` | afferent「+0.15 可交付」已被推翻 |
| memory `nailfold-full-label-audit` | 「overall_assessment 是最强字段 AUC 0.773」在产品口径下不成立（特异度 0.062） |

### 数据来源路径

| 资源 | 路径 |
|---|---|
| 标签清单 | `artifacts/manifest/locked_evaluation_v1_reviewed.csv`（102 列，20 个实质字段） |
| 分类 5-seed 预测 | `artifacts/experiments/unified_protocol_20260908/*_nested_predictions.csv.gz` |
| 测量二分类/序数 | `artifacts/experiments/measurement_ordinal_20260913/*.json`（17 个） |
| DINOv2 特征 | `artifacts/features/dinov2/{features.npy,index.csv}`（1708 帧 × 768） |
| 几何特征 | `artifacts/features/geometry_dev/features.npy`（1708 × 394） |
| 实例特征 A3 / B | `artifacts/experiments/v10_A3_instance_mil/instance_features.parquet` / `v10_B_measurement/instance_features_native.parquet` |

### 服务器环境

- 主机 `ssh -p 12956 root@connect.westc.seetacloud.com`（memory 中 westd/bjb1 两条均已不可达）
- Python 必须全路径：`/root/miniconda3/envs/nfc/bin/python`（含 pyarrow）；`lmed` 环境无 pyarrow
- 当前**无卡开机**：`/dev/nvidia*` 不存在，`torch.cuda.is_available()` False，`/usr/bin/nvidia-smi` 为 0 字节占位符
- `/` overlay 86% 满（剩 4.4G，miniconda 自占 26G）；`/root/autodl-tmp` 剩 89G

---

## 14. 安全约束（保持有效）

- **locked-47 不得训练、不得模型选择、不得评估、不得外发。** 每个脚本断言 `not (set(X.index) & locked)` 并记录 `locked_cases_seen: 0`。本轮全部脚本已确认。
- Qwen key 在 `/root/.qwen_key`（600），不得打印、记录或写入任何产出文件。
- 禁止声称绝对微米精度；禁止把 `calibration_factors.json` 的 `um_per_pixel` 当标定证据（label 反拟合，存在四个互不一致取值）。
- 不修改既有 artifacts，每个任务只写自己的输出目录。

---

## 15. 交付名单

**可上线（5 个）**

| 字段 | 形态 | 准确率 | delta |
|---|---|---|---|
| clarity | 二分类 清晰/不清 | 0.773 | +0.319 |
| exudation | 二分类 有/无 | 0.743 | +0.301 |
| blood_color | 二分类 | 0.762 | +0.210 |
| subpapillary_venous_plexus | 二分类 可见/不可见 | 0.717 | +0.147 |
| loop_length | 相对同群 偏长/偏短 | 0.727 | +0.273 |

边缘可选：periloop_score（+0.154）、total_score（+0.173，但 total_score 是合成分，产品不宜输出）。

**必须下线（15 个，= 20 − 5）**，按下线理由分两类：

| 类别 | 字段 | 含义 |
|---|---|---|
| **显著为负（9 个，delta<0）** | overall_assessment、hemorrhage、wbc_count、vasomotion、rbc_aggregation、sweat_duct、flow_state、capillary_count、crossing_ratio | 模型**不如**"永远输出多数类"这条零成本规则；上模型是净损失 |
| **不优于众数（6 个，delta>0 但 CI 含 0）** | microthrombus +0.082、malformation_ratio +0.062、papilla +0.016、efferent_diameter +0.079、afferent_diameter +0.067、apex_diameter +0.024 | 与众数规则**统计上不可区分**；不能声称有能力，但也不是"更差" |

（原文此处写"13 个"并漏了 microthrombus，是我的计数错误；且把两类混称为"跑不过众数"不够准确——真正**跑不过**的是 9 个。§1 摘要同步更正。）

**不建模**：`output_input_ratio`（= efferent/afferent 反算恒等式）。`flow_speed_um_s` 全空（186/186 缺失）、`patient_id` 全空。

## 16. 后续投入优先级

1. **第二中心 / 第二台设备数据** —— 唯一绕不过的门槛，当前为零。
2. **标签一致性测量** —— 不知道 0.77 距天花板多远即为盲目优化。（用户已否决双人全量重标；至少需部分样本一致性抽测。）
3. **低患病率阈值工程** —— 工作点从"最大化准确率"移到"控制假阳性"。
4. **弃答机制** —— 实测 afferent 60% 覆盖率 delta +0.170→+0.212；loop 40% 覆盖率 +0.205→+0.254（注：基于旧协议数字，需按本文口径重测）。

**不再投入**：LoRA 超参、更大模型、集成。三条路实测增益分别为 −0.020~+0.003、无、+0.003。
