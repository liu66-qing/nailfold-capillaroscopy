# 实验计划 v10：病例级标签下沉 + 测量值重构

日期：2026-09-13
授权：用户批准执行，服务器就绪（2× RTX 4090，0 占用）
标签版本：`/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv`（审标后，权威）

---

## 0. 铁律（所有子任务必须遵守，违反即作废）

1. **locked-47 绝对不碰**。不训练、不选模型、不评测、不看。每份 metrics.json 必须写
   `"locked_cases_seen": 0` 并由代码断言保证（参考 `scripts/train_binary_mil_features.py:133-141` 的校验模式）。
2. **折纪律**：`development_fold` 为 test 折，`val = (test+1) % 5`，其余 3 折训练。
   选 epoch / 选超参 / 选 pooling 算子**只能用 val 折**，report 只报 test 折 OOF。
3. **禁止跨字段挑选后汇总**。每个字段独立报告，不出 mean BA 这种会掩盖塌缩的数。
4. **每个实验必须同时报 4 个数**，缺一即视为未完成：
   - `accuracy` 与**折内众数基线**的差值（基线在训练折上算众数，不是全局众数）
   - 每个 n≥10 类别的 recall（n<10 的类别单独列，**不进任何汇总指标**）
   - `adj1`（|pred−true| ≤ 1 级的比例，序数字段必报）
   - `QWK`（quadratic weighted kappa）
   连续字段报 `MAE` + **折内中位数基线 MAE** + `MAE_CI95`。
5. **失败要写失败报告**，格式照 `artifacts/experiments/E*_failure_report.md`。
   不许把没过门槛的结果包装成"有提升"。不许调整门槛去迁就结果。
6. 不修改任何既有 baseline / v1 / v2 产物。新产物写到 `artifacts/experiments/v10_*/`。

---

## 1. 类别方案（已定，不再讨论）

依据：审标标签上的 5 折 OOF 实测（accuracy−众数基线 / QWK 双口径）。
用户已确认「浅红 / 淡红」在实际诊断中结论相同，无需双人重标验证。

| 字段 | 方案 | 类别 | dev 分布 | 理由 |
|---|---|---|---|---|
| exudation | **3 类** | 无 / 轻(+,++) / 重(+++) | 93 / 86 / 4 | acc +0.158、QWK 0.267→0.367 双升。`+++` n=4，**单列不进汇总，明示"样本不足不做保证"** |
| clarity | **2 类** | 清晰 / 不清+模糊 | 91 / 94 | acc +0.270、QWK 0.356→0.451 |
| blood_color | **2 类** | 暗红+暗紫 / 浅红+淡红 | 81 / 100 | acc +0.083、QWK 0.141→0.268。临床结论相同（用户确认） |
| subpapillary | **3 类** | 不见 / 可见1-2排 / >2排扩张 | 79 / 78 / 27 | acc +0.060。数排数是主观边界 |
| microthrombus | **3 类（保留）** | 无 / 1-2 / >2 | 108 / 45 / 29 | 合并后 QWK 无增益（0.265→0.267），且是方差贡献第一的字段（41%），塌缩成 2 类等于放弃它 |
| papilla | **3 类（保留）** | 平坦 / 浅波纹 / 波纹 | 42 / 77 / 66 | 合并后 acc −0.124、QWK −0.056。真三类 |
| capillary_count | **3 类** | >=7 / 5-6 / <=4 | 125 / 42 / 15 | 原始有 n=1 类别不可学 |
| **rbc_aggregation** | **固定众数** | — | — | 原始 −0.121、合并 −0.231，两种切法都跑不过基线 |
| **crossing_ratio** | **固定众数** | — | — | acc−基线 −0.24/−0.17，QWK 0.015。特征里没有这个信息 |
| **malformation_ratio** | **固定众数** | — | — | acc−基线 −0.22/−0.12，QWK 0.157。同上 |
| **hemorrhage / sweat_duct / vasomotion / wbc_count / flow_speed** | **固定众数** | — | — | 方差贡献 <0.7%，众数占比 0.93~1.00 |
| **flow_state / rbc_aggregation（流动类）** | **不建模** | — | — | EXP-B 已证否（视频 BA 0.204 < 随机 0.250） |

**建模字段共 7 个**：exudation, clarity, blood_color, subpapillary, microthrombus, papilla, capillary_count。

---

## 2. Track A：病例级标签下沉

### 背景：为什么还值得做（以及哪些已被封闭）

已封闭，**不要重跑**：
- E2（quality/topk pooling 对比，5 字段全部未过门槛）
- E5（frozen DINOv2/HuluMed/dual + attention MIL，最佳 BA 0.679 < v1 0.684）
- E7（渐进解冻，MAE 3.993/4.128，灾难性过拟合）

尚未被封闭的三个机制差异：
1. **目标从二分类改为多分类/序数**。E2/E5 全在二分类上做的，而二分类会把"轻 vs 重"折叠掉——
   这正是 pooling 算子唯一能起作用的维度。
2. **端到端 LoRA + case-level attention pooling**（`goal_mode_v9_architecture.md` 写了但从未执行）。
   现有 LoRA 脚本是**帧级训练 + 推理时 logit 均值**，梯度从未见过"病例"这个单位。
3. **instance 级袋而非 frame 级袋**。E5 的袋元素是整帧（9 帧/例），
   instance 级是单根管袢（~65/帧，需先修过检），袋大小差 2 个数量级。
   稀疏病灶（血栓/渗出/出血）在帧级袋里被 9 个元素平均，在 instance 级袋里不会。

### 支持性证据（本轮实测）
- 394 维几何特征中 **298 维的组内（同病例帧间）方差 > 总方差 50%**，pseudo-ICC = 0.386。
  帧间差异大于病例间差异，但标签只有病例级一个值。
- max/q90 pooling 对测量值的最大 |ρ| 优于 mean：apex 0.400→0.462、efferent 0.413→0.440。

### A1：多分类 + 逐字段 pooling 算子（CPU，快，先跑）
**目的**：在正确的目标（多分类/序数）下，重测 pooling 算子。这是 E2 的多分类版本。
- 特征：`artifacts/features/dinov2/features.npy` (1708,768) +
  `geometry_dev/features.npy` (1708,394) + `roi_quality_dev_v1/features.npy` (1708,372)
- pooling 候选：`mean` / `median` / `max` / `q90` / `topk(k=3) mean` / `logsumexp`
- 头：多项 logistic（序数字段另跑一套 ordinal cumulative-link）
- **算子在 val 折上选，test 折只评一次**
- 门槛：某字段 acc−基线 提升 ≥ **0.05** 且 **≥3/5 折同向** 且主类 recall 不降 >0.02
- 预期：`max`/`topk` 应在 microthrombus / exudation 上胜 `mean`（存在性事件）；
  在 clarity / blood_color 上不应胜（全局属性）。**如果 max 在 clarity 上也"胜"，说明是噪声，整个 A1 作废。**
  这条是 A1 的内建 sanity check，必须报告。

### A2：端到端 LoRA + case-level attention pooling，多分类（GPU 0）
**目的**：让梯度看见"病例"这个单位。v9 架构，从未执行。
- backbone：`vit_base_patch14_dinov2.lvd142m`，权重
  `/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth`
- **backbone 全程冻结，只训 LoRA**（rank 8 / lr 1e-4，取 baseline v2 最优配置，不做超参搜索——
  已饱和，见 `nailfold-method-ceilings`）
- 改动核心：一个病例的全部帧作为一个 batch 前向 → **gated attention pooling**（Ilse et al. 风格）
  → case 表示 → 7 个多分类头。loss 在 case 级算，不在 frame 级算。
- 逐字段 pooling 温度可学；attention 权重必须存盘（用于 A1 的 sanity 对照）
- 对照臂（必须跑，否则无法归因）：**同一代码、同一折、pooling 换成不动的 mean**。
  这是 A2 唯一合法的比较对象，**不是 baseline v2 的 0.799**（任务不同，不可比）
- 门槛：≥4/7 字段 acc−基线 提升 ≥0.05 且 ≥3/5 折同向

### A3：instance 级 MIL，只打 3 个稀疏病灶字段（CPU + GPU 1 空隙）
**目的**：袋元素从"帧"换成"单根管袢"。
- 输入：`artifacts/experiments/model-upgrade-20260831/instance_npz_dev2/`（1687 帧，186 例全覆盖）
- **第一步必须先修过检**：当前中位数 65 实例/帧，而公开 ANFC-THU 标注中位数是 10 环/图
  （`data/ANFC-THU.v1i.coco`，2418 normal + 57 abnormal，257 图）。
  score 范围 0.051~0.745。用 ANFC-THU 的每图实例数分布（med 10, p10 3, p90 17）
  反推 score 阈值 + NMS，**把每帧实例数校到中位数 8~15**。
  这一步的产出必须单独报告：修完之后每帧实例数分布。**修不到这个区间就停，不要继续往下跑。**
- 每实例特征：掩膜面积 / 骨架长度 / 局部宽度分位数 / bbox 长宽比 / score / quality / stability
  + 该实例 ROI 的颜色统计（血栓和渗出是颜色/纹理事件）
- 袋 → gated attention → 3 个头（microthrombus 3 类 / exudation 3 类 / hemorrhage 2 类）
- 门槛：microthrombus acc−基线 ≥ **+0.05**（当前 −0.110，这是全项目方差贡献第一的字段）

---

## 3. Track B：测量值

### 现状（本轮实测，dev 186，5 折 OOF）

| 字段 | 现模型 MAE | 折内中位数基线 | 有效? |
|---|---|---|---|
| afferent_diameter | 3.30 | 3.36 | 否（−1.6%） |
| efferent_diameter | 3.67 | 3.79 | 否（−3.3%） |
| apex_diameter | 6.79 | 6.86 | 否（−1.0%） |
| output_input_ratio | 0.310 | 0.290 | **反而更差** |
| loop_length | 80.9（换 QuantileTransformer+Ridge → **71.0**） | 86.5 | **是（−18%）** |

三个硬事实：
1. **`output_input_ratio` 100% 等于 `efferent/afferent`**（164 对全部误差 ≤0.05，纯四舍五入）。
   它没有独立信息。**必须改成计算字段，禁止单独建模。** 现在给它单独建模是纯粹的错误。
2. **评分阈值间距比数据精度小一个量级**。afferent 阈值 6.5/8.5/13.5/15.5/17.5/19.5/25.0（间距 1~2μm），
   而 MAE 3.3μm → **98% 的病例落在某阈值的一个 MAE 之内**。
   蒙特卡洛（当前 MAE 作噪声）落对分档概率：afferent 0.398 / efferent 0.459 / apex 0.432 / loop 0.441，
   **全部低于"永远猜众数档"的 0.455 / 0.604 / 0.544 / 0.466。**
3. **没有 μm/pixel 标定**。`artifacts/audits/vascular_dataset_governance_20260830/device_calibration_status.json`
   状态 `UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION`，明确 forbid `absolute micron measurements`。
   E8 已因此 HOLD。**这不是算法能解决的。**

特征侧天花板：394 维几何特征与测量值的最大 Spearman |ρ| 只有
进径 0.29 / 出径 0.41 / 顶径 0.40 / 管袢长 0.51 / 出入比 0.27。
根因在 `scripts/extract_geometry_features.py:27`：**1024×768 统一 resize 到 512×384**，
所有宽度统计都在缩放后的像素域做——用砍半分辨率的像素去测微米。

### B1：原生分辨率 instance 几何 → 测量回归（GPU 1）
**目的**：把测量从"全图纹理统计"换成"实例掩膜上量骨架长度和局部宽度"。这是测量该有的做法。
- 复用 A3 修好过检的实例掩膜，**在 1024×768 原生分辨率上**重跑（不 resize）
- 每实例：骨架总长（`skimage.morphology.skeletonize` + 路径长）、
  距离变换局部宽度（顶点区 / 上升支 / 下降支分段取分位数）、曲率、bbox
- case 级聚合：**mean / median / q75 / q90 / max 全跑，在 val 折上选**（本轮实测 q90/max 优于 mean）
- 目标：`afferent_diameter` / `efferent_diameter` / `apex_diameter` / `loop_length`
  （**`output_input_ratio` 不建模，由前两者相除得出**）
- 训练时对 y 做 1/99 分位 winsorize（afferent 有 325.0、apex 有 113.0 等明显 OCR 错值），
  评测时**用原始 y**，不许在评测端也 winsorize
- 门槛：**MAE 相对折内中位数基线降低 ≥10%**，且 5 折中 ≥4 折同向。
  达不到就写失败报告，不要报"接近基线"。

### B2：测量值改为分级交付 + 不确定度（CPU）
**目的**：在标定缺失、MAE 3μm、阈值间距 1.5μm 的现实下，输出"12μm"是不诚实的。
给出可交付的替代形态。
- 形态 1：**二分类"是否扩张"**。本轮已测（几何特征 + logistic）：
  afferent >15.5μm **AUC 0.789**（sens 0.53 / spec 0.83）、efferent >19.5μm **AUC 0.733**、
  apex >24.5μm AUC 0.608、loop >359μm AUC 0.697、ratio >2.0 **AUC 0.459（比随机差）**
  → 用 B1 的新特征重测，**逐字段给 AUC + CI + 最优阈值下的 sens/spec**
- 形态 2：**3 档序数**（正常 / 边界 / 异常），报 adj1 + QWK
- 形态 3：**conformal prediction 区间**。给每个测量值输出 90% 覆盖区间而非点估计。
  这是唯一在无标定条件下还诚实的连续值交付方式。报**实际覆盖率**和**平均区间宽度**。
- 交付判定：某字段若 AUC < 0.70 且 3 档 QWK < 0.30，**明确标注"当前不可靠，不出结论"**，
  不要给一个假的数字。按现有结果，apex_diameter 和 output_input_ratio 大概率落在这一档。

### B3：标定缺口的工程解（不需要 GPU，写成待办）
无 μm/pixel 标定是根本阻塞。可行的三条路，按成本排序：
1. 找出设备型号 + 倍率（用户侧信息，不是算法）
2. 报告图 `report_image`（978 张）里若有标尺或数值刻度，OCR 出来做反推标定
3. 用甲襞解剖学先验做相对标定（如以指甲襞宽度或表皮厚度为内参）
本轮不执行，但必须在最终报告里写明"测量值的绝对单位不可声明"。

---

## 4. 资源分配与并行

| 任务 | 设备 | 预估 | 依赖 |
|---|---|---|---|
| A1 多分类 pooling 算子 | CPU | 1~2h | 无 |
| A2 端到端 LoRA + case attention | **GPU 0** | 6~10h | 无 |
| A3 instance MIL（含修过检） | CPU | 3~5h | 修过检必须先过 |
| B1 原生分辨率 instance 几何回归 | **GPU 1** | 4~6h | 与 A3 共享修过检结果 |
| B2 分级交付 + conformal | CPU | 1~2h | 可先用旧特征跑，B1 出来后重跑 |

A3 与 B1 共享"修过检"这一步，由 A3 先产出阈值配置，B1 复用。

---

## 5. 全局验收（这轮算不算成功）

**算成功**，满足任一：
- Track A：≥2 个建模字段 acc−折内众数基线 提升 ≥0.05 且 ≥3/5 折同向，
  其中**至少 1 个是 microthrombus 或 exudation**（方差贡献前二，合计 54%）
- Track B：`loop_length` MAE 相对基线降 ≥10%，**且**至少 2 个测量字段拿到 AUC ≥0.75 的可交付二分类

**算失败**（同样是有价值的结论，必须照实写）：
- 全部字段仍打不过折内众数基线 → 说明瓶颈不在聚合方式，而在**病例级弱标签本身**，
  下一步只能是获取帧级/实例级标注，或者接受"7 字段固定众数 + 7 字段建模"的产品形态

**不接受的结论**：任何形式的"接近基线"、"有一定提升趋势"、"若调整门槛则通过"。
