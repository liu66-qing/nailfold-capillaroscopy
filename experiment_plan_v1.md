# 甲襞微循环全字段自动分析 — 实验规划 v1

> 日期: 2026-09-09
> 目标: 从 5 字段二分类 → 全部可训练字段覆盖，评估维度从 BA 升级到积分体系

---

## 1. 评判尺子（Evaluation Protocol）

### 1.1 为什么不能只看 BA

当前 5 字段模型用二分类 BA 评估。问题：
- **二分类丢失积分精度**：exudation 的 无/+/++/+++ 对应积分 0/1.5/2.4/3.6，二分类只区分"无 vs 有"，即使 BA=1.0 也无法给出正确积分
- **BA 不反映积分权重**：papilla BA=0.64 看起来很差，但 papilla 满分 1.6；microthrombus 没模型（等于 BA~0.33），满分 4.8，对总积分影响大 3 倍
- **最终交付物是临床报告**，报告质量 = 总积分准确度 + 综合判断准确度

### 1.2 指标定义

**主指标（决定实验是否有效的唯一标准）**：

| 指标 | 定义 | 含义 |
|---|---|---|
| **Total Score MAE** | mean(\|predicted_total_score - true_total_score\|) | 报告总积分的平均绝对误差 |
| **Overall Assessment Acc** | accuracy of 5-class overall_assessment | 综合判断分对了多少 |

**辅指标（用于诊断问题，不用于判断实验价值）**：

| 指标 | 定义 | 适用范围 |
|---|---|---|
| Per-field Score MAE (sMAE) | 单字段积分误差 | 所有字段 |
| Per-field BA | Balanced Accuracy | 分类字段 |
| Per-field Value MAE | 原始值误差（μm 等） | 测量字段 |

### 1.3 积分计算规则

积分映射已从 OCR 报告逆向工程得到，存储在：
- 规则文件: `/root/nailfold/artifacts/labels/score_rules_v3.json`
- 分类字段: categorical_lookup（查表）
- 测量字段: numeric_tree（决策树分 bin）
- 综合判断阈值: total_score < 1 正常, 1-2 大致正常, 2-4 轻度, 4-8 中度, ≥8 重度

积分结构：
```
morphology_score = clarity + capillary_count + afferent_diameter + efferent_diameter
                   + output_input_ratio + apex_diameter + loop_length
                   + crossing_ratio + malformation_ratio

flow_score = flow_state + flow_speed + vasomotion + rbc_aggregation
             + wbc_count + microthrombus + blood_color

periloop_score = exudation + hemorrhage + SVP + papilla + sweat_duct

total_score = morphology_score + flow_score + periloop_score
overall_assessment = threshold(total_score)
```

### 1.4 评估协议

- **数据划分**: 186 dev cases (5-fold CV, case-level isolation) + 47 locked test (仅最终使用一次)
- **模型选择**: 5 seeds × 5 folds, 嵌套 CV 选 epoch
- **基线对比指标**: ensemble ba_0_5（5-seed majority vote, threshold=0.5），最朴素无额外搜索
- **每次实验必须在相同 186 dev cases 上 OOF 评估**

---

## 2. 冻结基线 Baseline v0

### 2.1 基线定义

对每个字段使用当前最好可用方法：
- 5 个已有模型的字段: DINOv2 ViT-B/14 frozen CLS+patch（GatedPool），二分类，5 seeds × 5 folds
- 10 个无模型的分类字段: 预测训练集众数
- 4 个无模型的测量字段: 预测训练集中位数
- 2 个无标签字段: 跳过（output_input_ratio, flow_speed_um_s）
- 4 个派生字段 + 综合判断: 由上述预测积分求和 + 查阈值

基线模型位置: `/root/nailfold/artifacts/experiments/unified_protocol_20260908/`
基线 OOF 预测: `frozen_cls_patch_nested_predictions.csv.gz`

### 2.2 基线数字（已锁定，不可更改）

**整体指标：**

| 指标 | Baseline v0 |
|---|---|
| **Total Score MAE** | **3.48** |
| **Overall Assessment Acc** | **29.0%** (53/183) |
| Pred total mean±std | 3.9 ± 1.2 |
| True total mean±std | 6.5 ± 3.6 |
| 系统性偏差 | **低估 2.6 分**（无模型字段全预测正常 → 积分=0）|

**Per-field 指标：**

模型字段（DINOv2 frozen CLS+patch, 二分类, ensemble ba_0_5）：

| 字段 | BA | sMAE | 众数 sMAE | 积分增益 | max score |
|---|---|---|---|---|---|
| clarity | 0.827 | 0.065 | 0.111 | +0.046 | 0.6 |
| blood_color | 0.750 | 0.084 | 0.127 | +0.043 | 0.8 |
| exudation | 0.798 | 0.571 | 0.959 | +0.388 | 3.6 |
| SVP | 0.821 | 0.194 | 0.251 | +0.057 | 1.0 |
| papilla | 0.640 | 0.469 | 0.508 | +0.040 | 1.6 |

无模型分类字段（众数基线）：

| 字段 | 众数 | 众数占比 | sMAE | score std | max score |
|---|---|---|---|---|---|
| **microthrombus** | 无 | 61% | **1.202** | 1.723 | 4.8 |
| **rbc_aggregation** | 中度 | 42% | **0.581** | 0.772 | 3.0 |
| **flow_state** | 粒流 | 35% | **0.427** | 0.654 | 6.0 |
| **capillary_count** | >=7 | 69% | 0.372 | 0.722 | 6.0 |
| **malformation_ratio** | <=10% | 54% | 0.273 | 0.417 | 1.2 |
| crossing_ratio | <=30% | 65% | 0.104 | 0.206 | 1.2 |
| hemorrhage | 无 | 94% | 0.046 | 0.186 | 0.8 |
| vasomotion | 0--1 | 97% | 0.003 | 0.017 | 0.1 |
| wbc_count | 1--30 | 98% | 0.003 | 0.026 | 0.2 |
| sweat_duct | 0--2 | 99% | 0.001 | 0.010 | 0.1 |

无模型测量字段（中位数基线）：

| 字段 | 中位数 | sMAE | score std | max score |
|---|---|---|---|---|
| loop_length | 254 μm | 0.398 | 0.679 | 2.0 |
| afferent_diameter | 10 μm | 0.250 | 0.355 | 1.2 |
| apex_diameter | 17 μm | 0.228 | 0.369 | 1.0 |
| efferent_diameter | 14 μm | 0.132 | 0.237 | 0.8 |

### 2.3 标签质量说明

- 标签来源: OCR 识别临床报告（rapidocr_consensus_clean_v2.csv），241 cases
- 标签置信度: 多数字段 mean_confidence > 0.85（afferent/efferent_diameter 低一些 ~0.84）
- **已证明**: 训练标签质量对模型性能贡献 ~0（label ablation: C-B = -0.003 mean BA）
- **已证明**: 评估标签一致性才是关键（A→B = +0.045 mean BA）
- 当前 5 字段有人工审核的评估标签；其余字段只有 OCR 标签，评估可能偏噪

---

## 3. 实验计划

### 3.1 按积分影响力排序的优先级

字段对 Total Score MAE 的贡献 = 当前 sMAE × (是否有改善空间)。
按"潜在积分收益"排序：

| 排名 | 字段 | 当前 sMAE | score std | 有模型？ | 潜在收益 |
|---|---|---|---|---|---|
| 1 | microthrombus | 1.202 | 1.723 | 无 | 极高 |
| 2 | exudation | 0.571 | 1.018 | 有(二分类) | 高(升级多类) |
| 3 | rbc_aggregation | 0.581 | 0.772 | 无 | 高 |
| 4 | papilla | 0.469 | 0.662 | 有(二分类) | 中(升级多类) |
| 5 | flow_state | 0.427 | 0.654 | 无 | 中-高 |
| 6 | loop_length | 0.398 | 0.679 | 无 | 中(需特征工程) |
| 7 | capillary_count | 0.372 | 0.722 | 无 | 中 |
| 8 | malformation_ratio | 0.273 | 0.417 | 无 | 中 |
| 9 | afferent_diameter | 0.250 | 0.355 | 无 | 低-中(需特征工程) |
| 10 | apex_diameter | 0.228 | 0.369 | 无 | 低-中(需特征工程) |

### EXP-1: 全分类字段多类 DINOv2（最高优先级）

**目标**: 一次性解决两个问题：
1. 覆盖全部 12 个非固定分类字段
2. 将现有 5 个字段从二分类升级为多类（消除积分精度损失）

**改动**:
- FIELDS 扩展到 12 个字段（排除 vasomotion/wbc_count/sweat_duct/hemorrhage 这 4 个近固定字段）
- MAP 改为多类映射（每个字段 n_classes 个类别）
- Model head 改为 n_classes 输出（不再是全部 2 类）
- 损失函数: 多类 CE，对不平衡字段使用 inverse-frequency class weights
- 其余不变: frozen CLS+patch, GatedPool, 5 seeds × 5 folds, 0.75 case CE + 0.25 frame CE

**新增字段 MAP 定义**:

```python
FIELDS_V2 = {
    # 升级为多类的原有字段
    "clarity": {"清晰": 0, "不清": 1, "模糊": 2},                    # 3 类
    "blood_color": {"淡红": 0, "浅红": 1, "暗红": 2, "暗紫": 3},      # 4 类
    "exudation": {"无": 0, "+": 1, "++": 2, "+++": 3},               # 4 类
    "subpapillary_venous_plexus": {"不见": 0, "可见1排": 1, "可见2排": 2, ">2排,扩张": 3},  # 4 类
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 2},               # 3 类

    # 新增字段
    "capillary_count": {"<1": 0, "1--2": 1, "3--4": 2, "5--6": 3, ">=7": 4},  # 5 类
    "crossing_ratio": {"<=30%": 0, "30--60%": 1, "60--80%": 2, ">80%": 3},     # 4 类
    "malformation_ratio": {"<=10%": 0, "10--30%": 1, "30--60%": 2, ">60%": 3}, # 4 类
    "flow_state": {"线流": 0, "线粒流": 1, "粒线流": 2, "粒流": 3, "粒缓流": 4, "粒摆流": 5, "全停": 6},  # 7 类
    "rbc_aggregation": {"无": 0, "轻度": 1, "中度": 2, "重度": 3},     # 4 类
    "microthrombus": {"无": 0, "1--2": 1, ">2": 2},                   # 3 类
    "hemorrhage": {"无": 0, "1--2": 1},                                # 2 类（保留二分类）
}
```

**dev 标签覆盖（186 cases）**:

| 字段 | dev 有标签 | 类数 | 最小类样本数 |
|---|---|---|---|
| clarity | 185 | 3 | 模糊: ~8 |
| blood_color | 172 | 4 | 暗紫: ~1 (需合并或排除) |
| exudation | 178 | 4 | +++: ~6 |
| SVP | 181 | 4 | >2排,扩张: ~25 |
| papilla | 181 | 3 | ~56 each |
| capillary_count | 182 | 5 | <1: ~1, 1--2: ~1 (底部两类需合并) |
| crossing_ratio | 171 | 4 | >80%: ~4 |
| malformation_ratio | 159 | 4 | ~25 each |
| flow_state | 179 | 7 | 全停:~1, 粒摆流:~2 (需合并或排除) |
| rbc_aggregation | 181 | 4 | 重度: ~13 |
| microthrombus | 182 | 3 | >2: ~27 |
| hemorrhage | 180 | 2 | 1--2: ~10 |

**注意事项**:
- blood_color 的"暗紫"只有 1 例 → 合并到"暗红"（积分相同=0.4/0.8，实际暗紫0.8暗红0.4，需确认）
- capillary_count 的"<1"和"1--2"各只有 ~1 例 → 合并为"<=2"
- flow_state 的"全停"和"粒摆流"各 1-2 例 → 合并为"粒摆流/全停"
- 对极小类做合并后重新定义积分映射

**训练超参**: 与 Baseline v0 一致
- backbone: DINOv2 ViT-B/14, frozen
- pooling: CLS + GatedPool(patch tokens), concat → 1536-dim
- head: per-field linear (1536 → n_classes)
- lr: 1e-3, AdamW
- epochs: 20, 选 validation mean BA 最佳 epoch
- dropout: 0.1
- loss: 0.75 × case CE + 0.25 × frame CE, class-weighted

**预期结果**:
- 多类化后 exudation sMAE 应从 0.571 显著下降（当前二分类无法区分+/++/+++）
- papilla sMAE 应从 0.469 下降（当前无法区分浅波纹/波纹）
- microthrombus 从众数基线 sMAE=1.202 应大幅下降（DINOv2 对视觉特征有区分力）
- rbc_aggregation、flow_state 有望 sMAE 减半
- **保守预期 Total Score MAE: 2.2-2.8**（对比 Baseline v0: 3.48）

**耗时**: ~3-4 小时（12 字段 × 5 seeds × 5 folds × 20 epochs）
**GPU**: 2× RTX 4090, 可双卡并行

**脚本基础**: `/tmp/unified_protocol_20260908.py`，需修改 FIELDS/MAP/Model.heads

---

### EXP-2: 三类检测器 → 计数/比例字段（第二优先级）

**目标**: 用检测框直接算 crossing_ratio、malformation_ratio、capillary_count

**依据**: 
- 11,600 张图、76,380 个框（vessel: 35540, malformed_vessel: 29960, cross_vessel: 10860）
- 覆盖全部 233 cases + 347 额外 cases
- 标注量远超分类标签（76k 框 vs 241 case），且是 instance-level 标注

**做法**:
1. 标注格式转换: VOC XML → YOLO txt
2. 训练 YOLO11m-det（3类: vessel, malformed_vessel, cross_vessel）
3. 5-fold CV（复用 seg_vessel_v2 的 fold 划分，保证 case-level 隔离）
4. OOF 推理后:
   - `crossing_ratio = cross_vessel_count / total_vessel_count`，映射到 <=30%/30-60%/60-80%/>80%
   - `malformation_ratio = malformed_vessel_count / total_vessel_count`，映射到 <=10%/10-30%/30-60%/>60%
   - `capillary_count = total_vessel_count / 视野数` → 映射到 <1/1-2/3-4/5-6/>=7（需确认 mm 单位换算）
5. 对比 EXP-1 同字段的 DINOv2 结果，取更优者

**标注文件位置**: `E:\甲劈微循环\data\血管数据集\分类数据集\annotations\` (11,600 XMLs)

**前置条件**: 
- 需要确认 XML 中的图片分辨率和视野对应关系（capillary_count 是 "条/mm"，需要 pixel→mm）
- 需要将 "corss_vessel"(20 个) typo 标注修正为 "cross_vessel"

**预期**: 
- 这三个字段的标注信息量远超 case-level 标签，检测器应该给出更准的比例/计数
- 如果 EXP-1 的 DINOv2 已经在这三个字段上 sMAE < 众数基线的 50%，则本实验可降优先级

**耗时**: ~4-6 小时  
**依赖**: 无，可与 EXP-1 并行（如果用另一张 GPU）

---

### EXP-3: 测量字段特征工程（第三优先级）

**目标**: 从 instance mask 提取物理测量值 → afferent_diameter, efferent_diameter, apex_diameter, loop_length

**前置条件**:
1. **像素标定系数**: 同一设备（除 ANFC 外），需确定 μm/pixel 换算。方案:
   - 从标签值和像素测量值做线性回归反推（已知 ~200 case 的 μm 标签 + 对应 mask）
   - 或从设备参数手册获取
2. **Instance mask 数据**: 已有 `/root/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev/`，1687 帧，每帧 masks/boxes/quality

**做法**:
1. 标定: 对已有 geometry features（394-dim），找出哪些列对应 width/length，与标签做回归得到 μm/pixel
2. 特征提取（如果 394-dim 不含所需特征，需从 mask 重新提取）:
   - `skeletonize(mask)` → 中轴线
   - `loop_length` = 中轴线像素长度 × 标定系数
   - `apex_diameter` = skeleton 最高点处的 mask 宽度 × 标定系数
   - `afferent_diameter` = skeleton 下端左侧宽度 × 标定系数（需判断输入/输出端）
   - `efferent_diameter` = skeleton 下端右侧宽度 × 标定系数
3. 从帧级测量聚合到 case 级: 取中位数或按 quality score 加权平均
4. 直接将预测 μm 值通过 score_rules numeric_tree 映射为积分
5. 用 5-fold CV OOF 评估 sMAE

**风险**:
- afferent/efferent 区分需要知道血管方向（哪端是输入、哪端是输出），可能需要额外的方向标注或启发式规则
- 先做 loop_length 和 apex_diameter（不需要分方向），验证 pipeline 可行后再做 diameter

**预期**:
- loop_length: sMAE 0.398 → ~0.15-0.25（如果标定准确）
- apex_diameter: sMAE 0.228 → ~0.10-0.15
- afferent/efferent: 取决于方向判别的准确性

**耗时**: ~2 天（含 pipeline 调试）
**依赖**: 需要先确认标定系数

---

### 固定字段处理（无需实验）

| 字段 | 处理 | sMAE | 理由 |
|---|---|---|---|
| vasomotion | 永远输出 "0--1" | 0.003 | 97% 同值，积分贡献 max 0.1 |
| wbc_count | 永远输出 "1--30" | 0.003 | 98% 同值，积分贡献 max 0.2 |
| sweat_duct | 永远输出 "0--2" | 0.001 | 99% 同值，积分贡献 max 0.1 |
| hemorrhage | 视 EXP-1 结果决定 | 0.046 | 94% 无，模型可能不如众数 |

### 不可训练字段

| 字段 | 原因 | 处理 |
|---|---|---|
| output_input_ratio | 标签全缺 (n=0 in score_rules) | EXP-3 做出 afferent/efferent 后可计算 |
| flow_speed_um_s | 标签全缺 (n=0), 需视频时序 | 跳过，积分置 0 |

---

## 4. 结果记录模板

每次实验完成后，填写以下表格并追加到本文档末尾：

```
### EXP-X 结果 (日期: YYYY-MM-DD)

**改动摘要**: ...
**训练配置**: ...

**主指标对比**:

| 指标 | Baseline v0 | 本次 | 变化 |
|---|---|---|---|
| Total Score MAE | 3.48 | | |
| Overall Assessment Acc | 29.0% | | |
| Pred total mean±std | 3.9±1.2 | | |

**Per-field sMAE 对比**:

| 字段 | Baseline v0 sMAE | 本次 sMAE | 变化 |
|---|---|---|---|
| clarity | 0.065 | | |
| blood_color | 0.084 | | |
| exudation | 0.571 | | |
| SVP | 0.194 | | |
| papilla | 0.469 | | |
| capillary_count | 0.372 (mode) | | |
| crossing_ratio | 0.104 (mode) | | |
| malformation_ratio | 0.273 (mode) | | |
| flow_state | 0.427 (mode) | | |
| rbc_aggregation | 0.581 (mode) | | |
| microthrombus | 1.202 (mode) | | |
| hemorrhage | 0.046 (mode) | | |
| afferent_diameter | 0.250 (med) | | |
| efferent_diameter | 0.132 (med) | | |
| apex_diameter | 0.228 (med) | | |
| loop_length | 0.398 (med) | | |

**Per-field BA** (分类字段):
| 字段 | Baseline v0 BA | 本次 BA |
|---|---|---|
| ... | ... | ... |

**结论**: ...
**下一步**: ...
```

---

## 5. 执行顺序与时间线

```
Day 1 (今天):
  ├── EXP-1: 写多类训练脚本，开始训练 (~3-4h)
  │   └── 训练结束后立即算 Total Score MAE 和 Assessment Acc
  └── (并行) EXP-2 准备: XML→YOLO 格式转换，处理 typo

Day 2:
  ├── EXP-1 结果分析，填写结果记录
  ├── 根据 EXP-1 结果决定 EXP-2 是否执行
  │   └── 如果 DINOv2 在 crossing/malformation/capillary 上 sMAE > 众数的 60%，则执行
  └── EXP-2 训练 (~4-6h)

Day 3-4:
  ├── EXP-2 结果分析
  ├── EXP-3 标定系数确定
  └── EXP-3 特征工程 pipeline 开发

Day 5:
  ├── EXP-3 评估
  ├── 全字段集成: 每个字段取最优方法
  └── 计算最终 Total Score MAE 和 Assessment Acc
```

---

## 6. 关键资源清单

**服务器**: `ssh -p 12956 root@connect.westc.seetacloud.com`
**GPU**: 2× RTX 4090

**代码**:
- 训练框架: `/tmp/unified_protocol_20260908.py`
- Papilla CE 实验(已完成): `/tmp/papilla_ce_v3.py`
- Baseline v0 计算脚本: `/tmp/baseline_v0.py`

**数据**:
- 标签: `/root/nailfold/artifacts/labels/rapidocr_consensus_clean_v2.csv` (241 cases)
- 积分规则: `/root/nailfold/artifacts/labels/score_rules_v3.json`
- DINOv2 特征: `/root/nailfold/artifacts/features/dinov2/`
- Instance masks: `/root/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev/` (1687 frames)
- 检测标注: `E:\甲劈微循环\data\血管数据集\分类数据集\annotations\` (11,600 XMLs, 76,380 boxes)
- Seg 模型权重: `/root/nailfold/artifacts/models/seg_vessel_fold{0-4}/weights/best.pt`
- Geometry 特征: `/root/nailfold/artifacts/features/geometry_dev/` (394-dim, 1708 frames, 无列名)

**已完成实验结论（不再重复）**:
- 训练标签质量不是瓶颈（label ablation: C-B = -0.003）
- Papilla 加权 CE 无效（ordinary vs weighted BA 均为 0.597）
- Seg 聚合特征做分类全部失败（BA 0.25-0.55）
- 表格学习、LoRA 超参搜索、冻结特征 MIL 均已到顶

---

## 附录 A: 积分影响力分析

各字段按 score_std（积分标准差）排序，反映对总积分变异的实际贡献：

| 字段 | score mean | score std | max score | 当前 sMAE | 理论最优 sMAE |
|---|---|---|---|---|---|
| microthrombus | 1.202 | 1.723 | 4.8 | 1.202 (mode) | ~0.3-0.5 |
| exudation | 0.972 | 1.018 | 3.6 | 0.571 (binary) | ~0.2-0.4 |
| rbc_aggregation | 0.716 | 0.772 | 3.0 | 0.581 (mode) | ~0.2-0.4 |
| capillary_count | 0.372 | 0.722 | 6.0 | 0.372 (mode) | ~0.1-0.3 |
| loop_length | 0.491 | 0.679 | 2.0 | 0.398 (med) | ~0.1-0.2 |
| papilla | 0.649 | 0.662 | 1.6 | 0.469 (binary) | ~0.2-0.3 |
| flow_state | 0.625 | 0.654 | 6.0 | 0.427 (mode) | ~0.2-0.3 |
| malformation_ratio | 0.273 | 0.417 | 1.2 | 0.273 (mode) | ~0.1-0.2 |
| apex_diameter | 0.250 | 0.369 | 1.0 | 0.228 (med) | ~0.05-0.1 |
| afferent_diameter | 0.250 | 0.355 | 1.2 | 0.250 (med) | ~0.08-0.15 |
| SVP | 0.251 | 0.340 | 1.0 | 0.194 (binary) | ~0.1-0.15 |
| efferent_diameter | 0.132 | 0.237 | 0.8 | 0.132 (med) | ~0.05-0.1 |
| crossing_ratio | 0.104 | 0.206 | 1.2 | 0.104 (mode) | ~0.05-0.08 |
| hemorrhage | 0.046 | 0.186 | 0.8 | 0.046 (mode) | ~0.02-0.04 |
| blood_color | 0.277 | 0.134 | 0.8 | 0.084 (binary) | ~0.05-0.08 |
| clarity | 0.106 | 0.143 | 0.6 | 0.065 (binary) | ~0.03-0.05 |
| wbc_count | 0.003 | 0.026 | 0.2 | 0.003 (mode) | 0.003 |
| vasomotion | 0.003 | 0.017 | 0.1 | 0.003 (mode) | 0.003 |
| sweat_duct | 0.001 | 0.010 | 0.1 | 0.001 (mode) | 0.001 |
