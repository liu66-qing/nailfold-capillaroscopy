# 甲襞微循环全字段自动分析 — 实验规划 v3

> 日期: 2026-09-10
> 基于 v2 + Locked Test Set 评估 + 模型结构性诊断修订
> 核心转向: 从"字段级修补"转向"模型结构性缺陷修复"

---

## 0. 与 v2 的关键差异

v2 的思路是"哪个字段差就针对性修"（光流修 flow_state, patch 修 microthrombus）。
v3 的核心洞察是：**12 个字段共享同一套模型骨架，骨架本身有 6 个结构性缺陷，修骨架比修字段更高效。**

依据: Locked Test Set (47 case) 评估结果:
- **Spearman ρ = 0.264**（模型几乎不能排序病人严重程度）
- **error 与 true_ts 相关 -0.736**（经典回归到均值）
- 线性校准后 MAE 只从 3.14→2.80（89% 误差不可被校准压缩 = 纯噪声）
- 每个分类头 samples/param = 0.04（低于合理值 250 倍）

---

## 1. 已完成实验全览

### 1.1 EXP-1: 全字段多类 DINOv2（OOF + Test）

**OOF (186 dev, 5-fold × 5-seed):**

| 指标 | Baseline v0 | EXP-1 |
|---|---|---|
| Total Score MAE | 3.48 | 3.41 |
| Overall Assessment Acc | 29.0% | 27.6% |

**Locked Test Set (47 case, full-dev 训练 → test 推理):**

| 指标 | 模型 | 众数基线 |
|---|---|---|
| Total Score MAE | **3.14** | 4.46 |
| Overall Assessment Acc | **31.9%** | 19.1% |
| ±1 级准确率 | **70.2%** | — |
| 严重低估率 (≥2级) | **21.3%** | — |
| 灵敏度 (≥轻度) | **83.7%** | — |
| 特异度 | **25.0%** (n=4) | — |
| Spearman ρ | **0.264** | — |

Per-field OOF 信号强度 (lift = model_BA − random_BA):

| 档位 | 字段 | Lift | 含义 |
|---|---|---|---|
| 有信号 (>0.15) | clarity, blood_color, exudation | +0.19~0.21 | 全局视觉属性，DINOv2 可编码 |
| 弱信号 (0.05-0.15) | SVP, papilla, capillary_count, microthrombus | +0.08~0.17 | 有信息但分类头提取不充分 |
| 无信号 (<0.05) | crossing_ratio, malformation_ratio, flow_state, rbc_aggregation, hemorrhage | -0.01~+0.05 | 需要计数/比例/时序等冻结特征不编码的信号 |

### 1.2 EXP-1b: Ordinal Soft Label ❌

系统性高估 (pred_mean=8.18 vs true=6.53)。exponential distance matrix 导致概率过度扩散。方向不是错的，但实现有缺陷。

### 1.3 EXP-2: YOLO 3 类检测 ❌

检测器 mAP 合格，但标注语义与临床标签不对齐 (检测"malformed" >> 临床 malformation_ratio)。

### 1.4 EXP-4A: 光流 + 传统 ML ❌

169/247 视频 codec 不支持 → 已 ffmpeg 转码。RF/GBM/SVM 全部 sMAE > baseline 0.427。Farneback 光流特征对 flow_state 无信号。

---

## 2. 模型结构性缺陷诊断

### 缺陷 D1: 特征维度 vs 样本量严重失配

**现象**: 每个分类头 1536→N_classes，参数量 4600-6100，但独立样本仅 ~186 case。samples/param = 0.04（合理值 ≥ 10）。

**证据**: 混淆矩阵显示 microthrombus 预测几乎均匀分布在 3 个类中 (120/30/33)，与类先验 (109/45/29) 高度相关——模型在做"加噪的类先验回归"而不是基于视觉特征分类。

**后果**: 5-seed ensemble 本质是 5 个过拟合模型的投票，不能修正系统性容量不足。

### 缺陷 D2: Loss 不感知最终评估目标

**现象**: 12 个字段 loss 等权平均 (`loss / nt`)。但 flow_state 的 score_spread=6.0 是 clarity 的 10 倍。

**证据**: clarity BA=0.541 (已不错)，模型仍花等量梯度优化它；flow_state BA=0.186 (随机水平)，却没得到更多梯度。

**后果**: 高价值字段 (flow_state, capillary_count, microthrombus) 的改善被低价值字段的梯度稀释。

### 缺陷 D3: 冻结 Backbone 的领域差距

**现象**: DINOv2 在自然图像上预训练。甲襞图像的关键视觉信号（管腔颗粒感、微小暗色栓子、管袢密度）与自然图像语义无关。

**证据**: 有信号的字段 (clarity, blood_color) 恰好是可从通用纹理/色彩推断的；无信号字段 (flow_state, rbc_aggregation) 需要领域特有的微观模式识别。

**后果**: 分类头被迫从错误的特征空间里解码，天花板很低。

### 缺陷 D4: 帧间聚合丢失集合统计信息

**现象**: 每帧独立预测 → 帧间平均 logits → argmax。

**证据**: capillary_count (BA=0.329) 本质是"整只手有多少管袢"的计数任务，每帧只看一个手指 → 帧级 logits 无法编码跨帧计数信息。crossing_ratio 和 malformation_ratio 同理。

**后果**: 计数/比例类字段的 Lift 仅 0.05-0.08，接近随机。

### 缺陷 D5: 多任务梯度冲突

**现象**: 12 个字段共享 1536 维特征 → 各自独立线性头。

**证据**: clarity (全局模糊度) 和 microthrombus (局部微小结构) 需要的特征方向正交甚至冲突。当 microthrombus 的梯度调整某维度时，可能破坏 clarity 已学到的模式。

**后果**: 多任务联合训练反而不如单字段训练——clarity 多类 BA=0.541 低于 Baseline v0 的二分类。

### 缺陷 D6: 序数结构被忽略

**现象**: 标准 CE loss 将"预测 class 0 实际 class 4"和"预测 class 3 实际 class 4"视为同等错误。

**证据**: EXP-1b 尝试 ordinal soft label 但实现有缺陷 (exponential distance → 过度扩散)。方向不是错的。CORAL/CORN 是经过验证的 ordinal 方法。

**后果**: 模型没有动力避免"差很远"的预测，导致 21.3% 的严重低估。

---

## 3. Score Spread 与误差贡献分析

各字段对 total_score 的理论最大影响和当前误差占比:

| 字段 | Score Spread | 当前 sMAE | 占总 sMAE% | sMAE/Spread | 解读 |
|---|---|---|---|---|---|
| microthrombus | 4.8 | 1.405 | 30.3% | 0.293 | 高价值高误差，最需优化 |
| rbc_aggregation | 3.0 | 0.579 | 12.5% | 0.193 | 高价值高误差 |
| exudation | 3.6 | 0.549 | 11.8% | 0.152 | 中等 |
| flow_state | 6.0 | 0.492 | 10.6% | 0.082 | 最高理论价值，当前误差相对低 (因为众数偏保守) |
| papilla | 1.6 | 0.448 | 9.7% | 0.280 | sMAE/Spread 高 = 模型预测质量差 |
| capillary_count | 6.0 | 0.318 | 6.9% | 0.053 | 高价值但 sMAE 相对可控 |
| malformation_ratio | 1.2 | 0.234 | 5.0% | 0.195 | — |
| SVP | 1.0 | 0.199 | 4.3% | 0.199 | — |
| crossing_ratio | 1.2 | 0.144 | 3.1% | 0.120 | — |
| clarity | 0.6 | 0.103 | 2.2% | 0.172 | 低价值，可忽略 |
| blood_color | 0.8 | 0.102 | 2.2% | 0.127 | 低价值，可忽略 |
| hemorrhage | 0.8 | 0.067 | 1.4% | 0.084 | — |
| **合计** | — | **4.640** | — | — | — |
| 4 测量字段 | ~5.0 合计 | ~1.008 | — | — | 中位数基线 |

---

## 4. 实验方案（按缺陷驱动设计）

### 4.1 EXP-6: 骨架修复（P0, 修 D1+D2+D6）

**目标**: 不改 backbone、不加新数据源，仅修分类头和 loss，验证骨架修复能带来多大改善。

**改动 A — Bottleneck 降维 (修 D1)**:
```
当前:  1536 → Linear(N_classes)   params=4600~6100, samples/param=0.04
改为:  1536 → Linear(64) → ReLU → Dropout(0.3) → Linear(N_classes)   params=~350, samples/param=0.53
```
降维比 13×。同时 dropout 从 0.1 提到 0.3。

**改动 B — Score-Weighted Loss (修 D2)**:
```python
# 当前: one += 0.75 * case_ce + 0.25 * frame_ce; nt += 1  (等权)
# 改为:
field_weight = SCORE_SPREAD[f] / mean(SCORE_SPREAD.values())
one += field_weight * (0.75 * case_ce + 0.25 * frame_ce)
nt += field_weight
```

**改动 C — Ordinal-Aware Loss (修 D6)**:
选项 C1: **CORAL (Consistent Rank Logits)**
```python
# 将 N_classes 分类转为 N_classes-1 个二分类: P(Y > k)
# loss = sum of BCE for each rank threshold
```
选项 C2: **MSE 回归 class index**
```python
# 输出 1 维 → MSE(pred, true_class_idx / (N_classes-1))
# 推理时 round(pred * (N_classes-1))
```
选项 C3: **Soft ordinal CE (修正版)**
```python
# 与 EXP-1b 类似但修正: 用三角形分布而非指数, sigma=0.5 而非自适应
# P(k|y) ∝ max(0, 1 - |k-y|/sigma)
```
**推荐 C2 (MSE 回归)**：最简单，自动满足序数约束，1 个输出维度进一步减少参数。

**实验配置**: 2^3 = 8 种组合 (A/B/C 各 on/off)，但核心跑 4 种:
1. Baseline (原始 EXP-1，对照)
2. A only (降维)
3. A + B (降维 + score 加权)
4. A + B + C2 (降维 + score 加权 + MSE 回归)

**评估**: 5-fold × 5-seed OOF，对比 Total Score MAE、各字段 sMAE、Spearman ρ。

**工作量**: ~2 小时代码 + ~4 小时 GPU (4090 上 4 种配置)
**预期**: Total Score MAE 从 3.41 降到 2.6-2.9。严重低估率从 21% 降到 10-15%。

---

### 4.2 EXP-7: Backbone 解冻 (P1, 修 D3)

**前置**: EXP-6 最优配置已确定。

**目标**: 在 EXP-6 最优骨架上，解冻 DINOv2 最后 N 层，让特征适配甲襞领域。

**方案**:
```python
# 解冻最后 K 个 transformer block
for i, blk in enumerate(backbone.blocks):
    blk.requires_grad_(i >= len(backbone.blocks) - K)
backbone.norm.requires_grad_(True)  # 最后的 LayerNorm 也解冻
```

**超参搜索**:
| 参数 | 搜索范围 | 理由 |
|---|---|---|
| K (解冻层数) | {2, 4, 6} | 2 层 ≈ 14M params; 6 层 ≈ 42M |
| backbone LR | {5e-6, 1e-5, 2e-5} | 比 head LR (1e-3) 低 100 倍 |
| weight_decay | {0.05, 0.1} | 解冻后需要更强正则化 |

**实际跑 6 组** (K × LR 各取 3×2，weight_decay 固定 0.1):
- K=2, lr=5e-6 / 1e-5
- K=4, lr=5e-6 / 1e-5
- K=6, lr=1e-5 / 2e-5

**风险**: 186 case 微调 ViT-B 可能过拟合。监控指标: val_BA 的 epoch 曲线是否有明显 overshoot。如果 best_epoch 集中在 2-3，说明过拟合严重，需缩小 K 或加 data augmentation。

**数据增强加强** (配合解冻):
```python
# 当前: color_jitter=0, hflip=0.5, vflip=0
# 改为: color_jitter=0.2, hflip=0.5, vflip=0.5,
#        RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
#        RandomErasing(p=0.1)
```

**工作量**: ~1 小时代码 + ~8 小时 GPU (6 组 × 5-fold × 5-seed，但每组更慢因为解冻)
**预期**: 弱信号字段 (microthrombus, SVP, papilla) BA 提升 0.05-0.15。Total Score MAE 再降 0.3-0.5。

---

### 4.3 EXP-8: Case-Level 聚合 (P2, 修 D4)

**前置**: EXP-6 + EXP-7 最优配置。

**目标**: 替换帧间简单平均，用可学习的 case-level 聚合。

**方案 A — 两阶段冻结特征聚合 (推荐先做)**:

阶段一: 对全部 dev 图像用 EXP-7 最优 backbone 提取帧级特征 (保存为 .npy):
```
每帧 → backbone → CLS+GatedPool(patch) → 1536-dim (或降维后 64-dim)
```

阶段二: Case-level 模型:
```python
class CaseModel(nn.Module):
    def __init__(self, dim=64):
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.heads = nn.ModuleDict({f: nn.Linear(dim, N_classes[f]) for f in FIELDS})
    
    def forward(self, frame_feats):  # (num_frames, dim)
        x = frame_feats.unsqueeze(0)  # (1, num_frames, dim)
        x, weights = self.attn(x, x, x)  # self-attention across frames
        x = x.mean(1)  # (1, dim)
        return {f: h(x) for f, h in self.heads.items()}, weights
```

优点: 注意力权重可解释（哪些帧被重视）。可以学到"看到多个异常帧 → 严重"的模式。

**方案 B — Deep Sets (更简单的替代)**:
```python
class DeepSets(nn.Module):
    def __init__(self, dim=64, hidden=128):
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleDict(...)
    
    def forward(self, frame_feats):
        z = self.phi(frame_feats)   # per-frame transform
        z = z.sum(0, keepdim=True)  # permutation-invariant aggregation
        z = self.rho(z)
        return {f: h(z) for f, h in self.heads.items()}
```

**方案 C — 融合 YOLO 检测计数特征**:

EXP-2 的 YOLO 检测器虽然对 malformation_ratio 标签映射失败，但检测本身有效 (vessel/malformed/cross 三类检测框)。

```
每帧 YOLO 检测 → [n_vessel, n_malformed, n_cross,
                   ratio_malformed, ratio_cross,
                   avg_box_area, max_box_area]  (7-dim)
全帧聚合 → [mean, std, max, min] × 7 = 28-dim
拼接到 DINOv2 特征: 64 + 28 = 92-dim → case-level 模型
```

这为 capillary_count, crossing_ratio, malformation_ratio 提供直接的计数/比例信号，不依赖 DINOv2 从像素推断。

**推荐执行顺序**: 先 A (self-attention)，如果 capillary_count/crossing_ratio 没改善则加 C (YOLO 特征)。

**工作量**: ~3 小时代码 + ~2 小时 GPU (阶段二非常轻量)
**预期**: capillary_count/crossing_ratio/malformation_ratio sMAE 各降 20-40%。

---

### 4.4 EXP-9: 多任务分组 (P2, 修 D5)

**前置**: EXP-6 最优配置。可与 EXP-7/8 并行探索。

**目标**: 解决 12 字段共享特征空间导致的梯度冲突。

**方案 — 字段分组 + 独立 Adapter**:

按视觉信号类型分 4 组:

| 组 | 字段 | 共同视觉特征 |
|---|---|---|
| 全局 | clarity, blood_color, papilla | 整体模糊度、色调、表面纹理 |
| 局部 | microthrombus, hemorrhage, exudation | 局部异常点/区域 |
| 计数/比例 | capillary_count, crossing_ratio, malformation_ratio, SVP | 管袢密度和形态分布 |
| 动态 (fallback) | flow_state, rbc_aggregation | 微观动态 (当前仍用众数) |

每组独立的 bottleneck:
```python
self.adapters = nn.ModuleDict({
    'global':  nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Dropout(0.3)),
    'local':   nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Dropout(0.3)),
    'spatial': nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Dropout(0.3)),
})
# 每个字段的 head 从对应 adapter 的 64-dim 输出接
```

**替代方案: GradNorm 自动平衡**
不手动分组，而是让每个字段的 loss 权重自适应调整:
```python
# GradNorm: 监控每个 task 的 loss 下降速率
# 下降慢的 task 自动获得更高权重
```

**工作量**: ~2 小时代码 + ~4 小时 GPU
**预期**: clarity/blood_color 恢复到 Baseline v0 水平 (sMAE 降回 0.065/0.084)。整体 Total Score MAE 降 0.1-0.2。

---

### 4.5 EXP-10: 直接回归 Total Score (P2, 替代路径)

**目标**: 跳过逐字段分类，直接预测最终指标。

**动机**: 当前 pipeline 是 12 个独立分类 → 查表 → 加总，每个字段的误差独立累加，且查表丢失了类内的连续信息。如果最终目标是 total_score 和 overall_assessment，直接回归可能更高效。

**方案**:
```python
class DirectScoreModel(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone
        self.pool = GatedPool()
        self.bottleneck = nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Dropout(0.3))
        # 双头: 回归 total_score + 分类 overall_assessment
        self.score_head = nn.Linear(64, 1)        # 回归
        self.assess_head = nn.Linear(64, 5)        # 5 级分类
    
    def forward(self, x):
        z = self.backbone.forward_features(x)
        z = torch.cat([z[:, 0], self.pool(z[:, 1:])], 1)
        z = self.bottleneck(z)
        return self.score_head(z), self.assess_head(z)
```

**Loss**:
```python
loss = 0.5 * MSE(pred_score, true_total_score) + 0.5 * CE(pred_assess, true_assess)
```

**利弊**:
- 优点: 参数量极少 (64+1=65)，直接优化最终指标，不需要 score_rules 查表
- 优点: 186 个标注 case 全部可用 (不像逐字段时某些字段有缺失标签)
- 缺点: 丧失字段级可解释性（不知道哪个字段异常）
- 缺点: total_score 范围 0.3-14.4，分布严重偏斜，回归难度大

**折中方案**: 保留逐字段分类头 (为可解释性)，但加一个 total_score 辅助回归 loss:
```python
total_loss = field_classification_loss + 0.3 * MSE(sum_of_predicted_scores, true_total_score)
```
这让模型在学习每个字段时，同时感知"这些字段预测加起来应该接近 total_score"。

**工作量**: ~1.5 小时代码 + ~2 小时 GPU
**预期**: Total Score MAE 降到 2.5-3.0；如果配合 EXP-6 的 bottleneck，可能更好。

---

### 4.6 EXP-11: 视频信号（P3, 修 flow_state + rbc_aggregation）

**前置**: EXP-4A 光流特征已证明失败。需要新的视频表征方案。

**数据现状**:
- 247 个视频 (178 原始 AVI + 169 转码 MP4), 覆盖 128 case
- Dev set 中 95 case 有 flow_state 标签
- flow_state 5 类分布: c0=8, c1=38, c2=61, c3=61, c4=16

**方案 A — 预训练 Video Backbone Frozen Feature (推荐)**:

```python
# VideoMAE-v2 (ViT-B) 或 InternVideo2 frozen 特征
# 每个视频采样 8×16帧 clip → backbone → CLS token (768-dim)
# 8 个 clip → GatedPool → 768-dim case feature
# Linear head → 5 类 flow_state
```

选择依据: VideoMAE-v2 在 Kinetics 上预训练，学过"物体运动模式"，对血流运动可能有迁移性。InternVideo2 是多模态，可能更通用。

优先试 VideoMAE-v2 (参数少，推理快)。

**方案 B — 时空注意力 on DINOv2 帧特征**:

不引入新 backbone，而是对同一视频的连续帧用时间注意力:
```python
# 每个视频均匀采样 16 帧 → DINOv2 per-frame CLS → 16×768
# Temporal Transformer (2 层) → 时序聚合 → 768-dim
# Linear head → flow_state
```
优点: 不需要额外下载 video backbone。
缺点: DINOv2 不是为视频设计的，帧间信息可能还是不够。

**方案 C — 像素级运动统计 (改进版光流)**:

EXP-4A Farneback 失败可能因为:
1. 全帧光流被背景噪声淹没
2. 没有 mask 到血管区域

改进: 用 YOLO 检测 vessel ROI → 仅在 vessel bbox 内计算光流 → 提取管腔内流速/方向特征。

```
每帧 YOLO 检测 → vessel bbox → crop → Farneback 光流
→ ROI 内: flow_magnitude_mean, flow_direction_consistency, temporal_variance
→ 全视频聚合 → 特征向量 → 分类器
```

**推荐执行顺序**: C (成本最低, 基于已有 YOLO + 光流代码) → A (如 C 仍无信号)

**工作量**: C ~4 小时; A ~6 小时 (含下载 VideoMAE 权重)
**预期**: flow_state sMAE 从 0.492 降到 0.2-0.3 (C 方案) 或 0.15-0.25 (A 方案)

---

### 4.7 EXP-12: Microthrombus 高分辨率检测 (P3)

**保留 v2 的 EXP-5 方案，但加入 EXP-6 的 bottleneck**。

方案:
1. 用 instance mask crop 每根管袢 → resize 224×224
2. DINOv2 (可能已解冻的 EXP-7 版本) 提取 patch 特征
3. MIL (Multiple Instance Learning): case 标签弱监督
4. Case 级: 有栓 instance 计数 → 映射到 无/1-2/>2

**新增**: 如果 EXP-7 解冻 backbone 效果好，microthrombus 的 Lift 可能已从 0.085 提升到 0.15+。需先看 EXP-7 结果再决定是否需要专门的 patch pipeline。

---

### 4.8 EXP-13: 测量字段 (P3, 保留自 v2 EXP-3)

从 instance mask 骨架化提取 loop_length, apex_diameter, afferent/efferent_diameter。

已有: geometry_features (394-dim), instance_masks (N×768×128)。

**改进**: 与 EXP-8C 的 YOLO 特征合并，构建统一的"结构特征向量"。

---

## 5. 执行计划与依赖图

```
           ┌─────────┐
           │ EXP-6   │ ← P0, 修 D1+D2+D6
           │ 骨架修复  │    bottleneck + loss权重 + ordinal
           └────┬────┘
                │ 确定最优骨架配置
       ┌────────┼────────┐
       ▼        ▼        ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │ EXP-7  │ │ EXP-9  │ │ EXP-10 │  ← P1/P2, 可并行
  │ 解冻   │ │ 分组   │ │ 直接回归│
  └───┬────┘ └───┬────┘ └───┬────┘
      │          │          │
      ▼          ▼          ▼
  ┌──────────────────────────┐
  │ EXP-8: Case-Level 聚合   │ ← P2, 在最优 backbone 上做
  │ (self-attention + YOLO)  │
  └──────────┬───────────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│EXP-11 │ │EXP-12 │ │EXP-13 │  ← P3, 视需要
│ 视频  │ │microT │ │ 测量  │
└───────┘ └───────┘ └───────┘
```

### 时间预估

| 阶段 | 实验 | 代码 | GPU | 总计 |
|---|---|---|---|---|
| Phase 1 | EXP-6 (4 种配置, OOF) | 2h | 4h | **6h** |
| Phase 2a | EXP-7 (6 组超参, OOF) | 1h | 8h | **9h** |
| Phase 2b | EXP-9 (分组 adapter) | 2h | 4h | **6h** |
| Phase 2c | EXP-10 (直接回归) | 1.5h | 2h | **3.5h** |
| Phase 3 | EXP-8 (case-level) | 3h | 2h | **5h** |
| Phase 4 | EXP-11/12/13 (按需) | 4-8h | 6-12h | **10-20h** |
| Final | 最终集成 + test set | 2h | 1h | **3h** |
| **总计** | — | — | — | **~40-50h** |

**双卡并行**: GPU 0 跑 EXP-6 → EXP-7; GPU 1 跑 EXP-9 → EXP-10。Phase 1+2 可在 ~12h 内完成。

---

## 6. 成功标准 (基于 "90% 产品落地" 定义)

### 最终目标 (Locked Test Set 上):

| 维度 | 指标 | 目标门槛 | 当前值 | 差距 |
|---|---|---|---|---|
| 安全底线 | 严重低估率 (≥2级) | ≤ 5% | 21.3% | -16pp |
| 方向正确 | ±1 级准确率 | ≥ 85% | 70.2% | -15pp |
| 识别病人 | 灵敏度 (≥轻度) | ≥ 90% | 83.7% | -6pp |
| 不过度惊吓 | 特异度 | ≥ 70% | 25% (n=4) | -45pp |
| 排序能力 | Spearman ρ | ≥ 0.6 | 0.264 | -0.34 |

### 阶段性里程碑:

| 里程碑 | 达标标准 (OOF) | 对应实验 |
|---|---|---|
| M1: 骨架修复 | Total Score MAE < 3.0, Spearman ρ > 0.4 | EXP-6 |
| M2: 特征适配 | Total Score MAE < 2.5, 严重低估率 < 15% | EXP-7 + EXP-8 |
| M3: 产品可评估 | ±1 级 ≥ 80%, 灵敏度 ≥ 88% | Phase 2 全部 |
| M4: 产品落地 | 满足全部 5 项门槛 | Phase 3 + Final |

### 决策节点:

- **EXP-6 后**: 如果 bottleneck + score-weighted loss 带来 < 0.2 改善 → 说明问题不在分类头，需重新评估 backbone (优先做 EXP-7)
- **EXP-7 后**: 如果解冻 2 层 BA 显著提升但解冻 6 层反而下降 → 说明数据量确实不足以支撑深层微调，需要数据增强或自监督预训练
- **EXP-10 后**: 如果直接回归 Total Score MAE < 2.5 → 考虑以直接回归为主 pipeline，字段分类作为辅助可解释模块
- **Phase 2 完成后**: 用 Locked Test Set 做一次全面评估。如果 ±1 级 < 80% → 进入 Phase 3 (视频/高分辨率)；如果 ≥ 80% → 进入产品集成

---

## 7. 不再做的实验 (v2 遗留)

| 实验 | 原因 |
|---|---|
| 更多 Farneback 光流变体 | EXP-4A 已证明全帧光流对 flow_state 无信号 |
| YOLO → malformation_ratio 直接映射 | 标注语义与临床标签不对齐 |
| LoRA 微调 | 被 EXP-7 全层解冻替代 (更直接) |
| 表格学习 on geometry features | v1 已证明到顶 |
| EXP-1b exponential ordinal loss | 已证明实现缺陷导致高估 (被 EXP-6C 的 CORAL/MSE 替代) |

---

## 8. 资源清单

### 服务器
- SSH: `ssh -p 12956 root@connect.westc.seetacloud.com`
- 双卡 RTX 4090 (24GB × 2)
- Python: `/root/miniconda3/bin/python`

### 关键文件路径
- 标签: `/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv`
- 分数规则: `/root/nailfold/artifacts/labels/score_rules_v3.json`
- 图像索引: `/root/nailfold/artifacts/features/image_index.csv`
- DINOv2 权重: `/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth`
- EXP-1 结果: `/root/nailfold/artifacts/experiments/exp1_multiclass/`
- EXP-2 YOLO: `/root/nailfold/artifacts/experiments/exp2_yolo_det/`
- Test Set 评估: `/root/nailfold/artifacts/experiments/test_set_evaluation/`
- 视频数据: `/root/nailfold/data/recovered_archive*/*/wmv*.{avi,mp4}`
- Geometry features: `/root/nailfold/artifacts/features/geometry_dev/`

### 训练脚本
- EXP-1 基准: `/tmp/exp1_multiclass.py`
- Test set 评估: `/tmp/test_set_eval.py`
- EXP-2 pipeline: `/tmp/exp2_pipeline.sh`
- 光流提取: `/tmp/extract_flow_single.py`
