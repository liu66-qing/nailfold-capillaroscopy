# 甲襞微循环全字段自动分析 — 实验规划 v2

> 日期: 2026-09-10
> 基于 v1 + EXP-1/1b/2 实验结果修订
> 核心发现: frozen DINOv2 对 microthrombus/flow_state/rbc_aggregation 无信号，瓶颈在信号源

---

## 1. 已完成实验结果（EXP-1 / EXP-1b / EXP-2）

### 1.1 EXP-1: 全分类字段多类 DINOv2 ❌ 整体失败

| 指标 | Baseline v0 | EXP-1 | 判定 |
|---|---|---|---|
| **Total Score MAE** | 3.48 | 3.41 | -0.07, 无实质改善 |
| **Overall Assessment Acc** | 29.0% | 27.6% | 退步 |
| Pred total mean±std | 3.9±1.2 | 4.79±2.54 | |

Per-field sMAE 对比:

| 字段 | Baseline v0 | EXP-1 | 判定 |
|---|---|---|---|
| clarity | **0.065** | 0.103 | ❌ 退步 |
| blood_color | **0.084** | 0.102 | ❌ 退步 |
| exudation | 0.571 | **0.549** | ✓ 微进步 |
| SVP | **0.194** | 0.199 | ❌ 退步 |
| papilla | 0.469 | **0.448** | ✓ 微进步 |
| capillary_count | 0.372 | **0.318** | ✓ 改善 |
| crossing_ratio | **0.104** | 0.144 | ❌ 退步 |
| malformation_ratio | 0.273 | **0.234** | ✓ 改善 |
| flow_state | **0.427** | 0.492 | ❌ 退步 (BA=0.186≈随机) |
| rbc_aggregation | **0.581** | 0.579 | → 持平 (BA=0.269≈随机) |
| microthrombus | **1.202** | 1.405 | ❌ 大幅退步 (BA=0.418) |
| hemorrhage | **0.046** | 0.067 | ❌ 退步 |

**结论**: 多类化只对 4 个字段有效（exudation, papilla, capillary_count, malformation_ratio），5 个原有字段中 3 个退步。microthrombus/flow_state/rbc_aggregation 的 BA ≈ 随机，DINOv2 frozen 特征对这些字段无区分力。

### 1.2 EXP-1b: Ordinal loss 变体 ❌ 更差

| 指标 | EXP-1 | EXP-1b |
|---|---|---|
| Total Score MAE | 3.41 | 3.29 |
| Overall Acc | 27.6% | 41.6% (假象) |
| Pred total mean | 4.79 | **8.18** (严重高估) |

EXP-1b 的 ordinal soft label 让模型倾向预测高积分类，系统性高估 1.65 分。Overall Acc=41.6% 是假象（高估恰好碰到真实分布的中度/重度区间）。每个字段 sMAE 几乎全部比 EXP-1 更差。

**结论**: ordinal loss 在这个场景下无效，放弃。

### 1.3 EXP-2: YOLO 3 类检测 → 部分有效

5-fold YOLO11m, best epoch=5-6 (early stopping), 580 cases OOF:

| 字段 | Baseline sMAE | EXP-2 sMAE | 判定 |
|---|---|---|---|
| crossing_ratio | **0.104** | 0.125 | ❌ 未赢 baseline |
| malformation_ratio | **0.273** | 0.510 | ❌ 严重偏高 |

malformation_ratio 失败原因: 检测标注的 "malformed_vessel" 判定标准比临床报告宽松（标注 39% 畸形，临床报告 54% <=10%），标注定义不一致。

crossing_ratio 0.125 接近但没赢 baseline 0.104。

**结论**: EXP-2 检测器本身 mAP 还行，但临床标签和检测标注的语义不对齐，直接用比例映射不可靠。

---

## 2. 当前最优组合（Cherry-pick Best-per-field）

| 字段 | 最优方法 | sMAE | 来源 |
|---|---|---|---|
| clarity | DINOv2 二分类 | 0.065 | Baseline v0 |
| blood_color | DINOv2 二分类 | 0.084 | Baseline v0 |
| exudation | DINOv2 多类 | 0.549 | EXP-1 |
| SVP | DINOv2 二分类 | 0.194 | Baseline v0 |
| papilla | DINOv2 多类 | 0.448 | EXP-1 |
| capillary_count | DINOv2 多类 | 0.318 | EXP-1 |
| malformation_ratio | DINOv2 多类 | 0.234 | EXP-1 |
| crossing_ratio | 众数 | 0.104 | Baseline |
| flow_state | 众数 | 0.427 | Baseline (学不动) |
| rbc_aggregation | 众数 | 0.581 | Baseline (学不动) |
| microthrombus | 众数 | 1.202 | Baseline (学不动) |
| hemorrhage | 众数 | 0.046 | Baseline |
| 4 测量字段 | 中位数 | 1.008 合计 | Baseline |
| 3 固定字段 | 众数 | 0.007 合计 | 固定 |

**预估 cherry-pick Total Score MAE ≈ 3.26**

---

## 3. 瓶颈分析

### 3.1 Total Score MAE 的分解

当前 Total Score MAE ≈ 3.26 = 各字段 sMAE 之和的某种上界。

**不可改善的地板** (众数字段):
- microthrombus: 1.202 (score_std=1.723, max=4.8)
- rbc_aggregation: 0.581 (score_std=0.772, max=3.0)
- flow_state: 0.427 (score_std=0.654, max=6.0)
- 小计: **2.21**

**可改善的天花板** (已有模型 + 测量):
- 其余字段 sMAE 合计: ~3.05
- 如果全部做到理论最优: ~0.8-1.2
- 理论改善空间: ~1.8-2.2

**结论**: 即使所有其他字段做到极致，地板 2.21 限制了 Total Score MAE 不可能低于 ~2.5。要突破必须攻克 microthrombus/flow_state/rbc_aggregation。

### 3.2 为什么学不动

| 字段 | 失败原因 | 所需信号 |
|---|---|---|
| flow_state | **时序信息**：粒流/线流的区别在于红细胞运动模式 | 视频光流 |
| rbc_aggregation | **微观动态**：红细胞是否聚集成团 | 高分辨率 + 可能需视频 |
| microthrombus | **局部微观**：血管内白色栓子 | 高分辨率血管级 patch |

---

## 4. 修订实验计划

### EXP-4: Flow State 视频分类（最高优先级）🔥

**目标**: 用视频时序特征预测 flow_state

**数据**:
- 247 个 AVI, 1024×768, 20fps, ~12s
- 覆盖 128 unique cases，其中 95 个 dev case 有 flow_state 标签
- 合并后 4 类: 线流+线粒流(12), 粒线流(31), 粒流(39), 粒缓流+(13)
- **问题: 95 case 做 5-fold CV，每 fold 只有 ~19 test cases，统计波动大**

**方案 A: 光流特征 + 简单分类器（推荐先做）**
1. 对每个视频提取密集光流 (Farneback / RAFT)
2. 统计特征: 光流 magnitude 的 mean/std/percentiles, 方向一致性, 时间变异度
3. 每帧 → 一个特征向量，case 级聚合 (median/mean)
4. 用 SVM/RF/LightGBM 做 4 类分类，5-fold CV
5. **优势**: 不需要深度学习训练，特征工程可解释，95 case 对传统 ML 够用

**方案 B: 视频 backbone (如果 A 效果不够)**
1. 提取连续 16 帧 clip
2. 用预训练的 VideoMAE/TimeSformer frozen 特征
3. Linear probe 分类
4. **风险**: 95 case 训练 video model 容易过拟合

**前置条件**: 确认 OpenCV 能正常读取 AVI; 确认 case 到视频的映射

**预期**: flow_state sMAE 从 0.427 降到 ~0.15-0.25（如果光流信号够强）
**单字段对 Total Score 的影响**: 最多回收 ~0.2-0.3 分
**耗时**: ~4-6 小时 (特征提取 + 实验)

### EXP-5: Microthrombus 高分辨率 Patch 分类（第二优先级）

**目标**: 从血管级 patch 判断有无微血栓

**方案**:
1. 用已有 seg 模型 crop 每根血管的 bounding box（从 instance mask）
2. Resize 到统一大小（如 224×64）
3. DINOv2 提取 patch 特征
4. Patch-level 二分类: 有栓/无栓（case 标签弱监督: "无"→全 patch 无, ">2"→至少 3 个有）
5. Case 级聚合: 有栓 patch 计数 → 映射到 无/1-2/>2

**风险**:
- 弱标签: case-level 标签不知道哪根血管有栓子
- 可能需要 MIL (multiple instance learning) 架构
- 如果栓子在静态图上就看不清，那也学不动

**前置条件**: 确认 instance mask 的质量（已有 1687 帧 masks）
**预期**: sMAE 从 1.202 降到 ~0.6-0.8（保守）
**耗时**: ~1-2 天

### EXP-3: 测量字段特征工程（第三优先级，保持）

与 v1 方案相同。
- 从 instance mask 骨架化提取 loop_length, apex_diameter
- 从标签值反推 μm/pixel 标定系数
- EXP-3 探索已完成: geometry features 是 394-dim numpy, instance masks 是 (N, 768, 128) uint8
- **预期 sMAE 改善**: 4 字段合计从 1.008 降到 ~0.4-0.6

### 不再做的实验

| 实验 | 原因 |
|---|---|
| EXP-1b ordinal loss | 已证明无效（系统性高估） |
| 更多 DINOv2 变体 (LoRA, unfreeze) | 信号源问题不是容量问题 |
| EXP-2 malformation_ratio | 标注定义不一致，修不了 |
| 表格学习/MIL on seg 特征 | v1 已证明到顶 |

---

## 5. 修订执行顺序

```
Day 1 (今天):
  ├── EXP-4A: 光流特征提取 (~2h)
  ├── EXP-4A: flow_state 分类实验 (~1h)
  └── Cherry-pick 混合模型: 计算真实 Total Score MAE (~30min)

Day 2:
  ├── 根据 EXP-4A 结果决定是否做 EXP-4B (视频 backbone)
  ├── EXP-5: microthrombus patch 分类 (~4-6h)
  └── EXP-3: 标定系数 + loop_length/apex_diameter pipeline

Day 3:
  ├── 全字段最优组合集成
  ├── 计算最终 Total Score MAE
  └── 在 47 locked test set 上做一次最终评估
```

---

## 6. 资源更新

**已完成实验结果**:
- EXP-1: `/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json`
- EXP-1b: `/root/nailfold/artifacts/experiments/exp1b_ordinal/exp1b_results.json`
- EXP-2: `/root/nailfold/artifacts/experiments/exp2_yolo_det/oof_case_predictions.json`
- 对比: `/root/nailfold/artifacts/experiments/comparison_results.json`

**视频数据**:
- 位置: `/root/nailfold/data/recovered_archive*/*/wmv*.avi`
- 规格: 1024×768, 20fps, ~12s, 共 247 个 AVI (72GB)
- 覆盖: 128 cases (dev 95, locked_test ~33)

**EXP-3 探索结果**:
- Geometry features: `/root/nailfold/artifacts/features/geometry_dev/` (features.npy, 394-dim)
- Instance masks: 每帧 ~147 个 mask, (768, 128) 尺寸, 含 quality/stability/score
