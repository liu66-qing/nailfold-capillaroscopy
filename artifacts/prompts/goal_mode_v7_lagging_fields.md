
# 补充：v6 结果修正 + 滞后字段第二轮

## 修正：v6 不是失败，以下字段已通过闸门

你的 v6 报告结论是"未发现稳定优于 v1 的新模型"，这是错的。按字段级闸门：

| 字段 | v3 BA | v6 BA | 提升 | 闸门 |
|------|-------|-------|------|------|
| exudation | 0.711 | **0.772** | +6.1pp | **PASS**（4/5 folds 获益）|
| papilla | 0.577 | **0.632** | +5.5pp | 检查 fold wins 和少数类召回 |
| SVP | 0.751 | **0.762** | +1.1pp | 检查 fold wins |

exudation 0.772 是目前所有字段中最接近 0.90 的。这个结果必须保留，不能因为"全局没有 v2"就丢弃。

当前最优字段级路由应更新为：
- clarity: 0.716（v3 YOLO 检测）
- blood_color: 0.710（v3 YOLO 检测）
- exudation: **0.772**（v6 DINOv2 LoRA rank4 30ep lr1e-4）
- SVP: 0.762（v6 DINOv2 LoRA）
- papilla: 0.632（v6 DINOv2 LoRA）
- hemorrhage: 0.500（v6，未改善）
- capillary_count: 0.529（v1）
- crossing_ratio: 0.345（v1）
- malformation_ratio: 0.340（v1）

## 滞后字段第二轮：上一轮的方案没有被正确执行

### hemorrhage：上一轮用了 v1 的方法，没有做专家模型

上一轮 hemorrhage 实验是 DINOv2 冻结均值 + ExtraTrees，这就是 v1——当然 BA 还是 0.50。
必须执行的方案（按顺序）：

方案 1：DINOv2 LoRA 帧级微调 + top-k 聚合（非 mean）
  - 复用 v6 的 finetune_dinov2_lora_binary_cv.py
  - 只保留 hemorrhage 一个 head
  - focal loss (gamma=2, alpha=0.75)
  - 关键改动：predict() 函数中的聚合从 np.mean 改为 top-k
    即：取该病例所有帧中阳性概率最高的 k 帧（k=1,2,3 都试），用这 k 帧的平均概率决定病例标签
  - 出血是"存在性"事件——一帧有出血就够了，不应该对所有帧取均值
  - 训练 30 epoch + early stopping

方案 2：如果方案 1 的 top-k 有提升但不够
  - 对 DINOv2 的 patch token（不是 CLS token）做空间注意力
  - 让模型学会关注帧内的出血区域
  - 搜索论文 "weakly supervised lesion detection MIL" 找实现

方案 3：如果帧级信号太弱
  - 搜索 "retinal microhemorrhage detection few-shot"
  - 搜索 "nailfold hemorrhage capillaroscopy"
  - 看有没有预训练模型或数据增强策略可以迁移

### crossing_ratio：检测框代理完全无效（Spearman -0.016），需要换思路

检测框 cross_vessel 数量和临床交叉率零相关，原因可能是：
1. YOLO 检测器对 cross_vessel 的检测不准确（只有 547 个训练框，且 mAP 0.44）
2. 临床"交叉率"的定义不是简单的"交叉血管数/总血管数"

**新方案：放弃检测框计数，改用端到端学习**

方案 1：DINOv2 LoRA 直接学 crossing_ratio
  - 和 exudation/SVP 一样，用 DINOv2 LoRA 端到端微调
  - 标签：序数分类（<=30%, 30-60%, 60-80%, >80%）
  - 类别不平衡严重（112:48:12:5），用 focal loss + class weight
  - 考虑合并稀有类：将 60-80% 和 >80% 合并为 ">60%"（17 例），变成 3 类
  - 30 epoch + early stopping

方案 2：分割 mask 骨架化 → 交叉点检测
  - 在 2,332 对 mask 上：二值 mask → 骨架化 → 检测交叉点（3×3 邻域有 ≥3 个连通方向的像素）
  - 交叉点数量/骨架总长度 = 交叉密度
  - 这是确定性算法，不需要训练，先验证和临床标签的相关性
  - 如果 Spearman > 0.3 → 作为特征输入分类器

### malformation_ratio：同样需要端到端学习

方案 1：DINOv2 LoRA 直接学
  - 同 crossing_ratio，序数分类
  - 类别合并为 3 类（如果稀有类 < 10 例）

方案 2：分割 mask 形态学特征
  - 从 mask 提取每个连通域的形态：圆度、紧凑度、宽度变化系数、弯曲度
  - "畸形"血管通常表现为异常宽、不规则弯曲、膨大
  - 异常形态比例 = malformation_ratio 的代理

### capillary_count：检测框计数思路没错，但需要改进映射

BA 0.207 说明硬阈值映射错了，但检测框数量本身可能和管袢数有正相关。

方案 1：检查相关性
  - 画散点图：每病例平均检测框数 vs 临床管袢数标签
  - 如果 Spearman > 0.3 → 用 LightGBM 学非线性映射，不用硬阈值
  - 如果 Spearman < 0.2 → 放弃检测计数路线

方案 2：DINOv2 LoRA 直接学
  - 序数分类，合并稀有类（<1 类只有 1 例，合并到 3-4）
  - 变成 3 类：>=7, 5-6, <=4

方案 3：DINOv2 LoRA + 检测特征联合
  - 拼接 DINOv2 CLS embedding + YOLO 检测特征 → MLP head

## 执行规则

1. 每个字段至少尝试 2 个方案，不要只试一个就放弃
2. hemorrhage 必须用 top-k 聚合，不能用 mean（这是上一轮的核心遗漏）
3. crossing/malformation 先试 DINOv2 LoRA 端到端（已验证路线），再试其他
4. 不要再用 ExtraTrees + 冻结特征的 v1 方法——v4 已证明到顶
5. 遇到瓶颈搜索论文，找到方法就试
6. 每个实验要有 .py 脚本 + 训练日志 + OOF

## 目标
- hemorrhage: 0.491 → ≥ 0.65
- capillary_count: 0.529 → ≥ 0.65
- crossing_ratio: 0.345 → ≥ 0.50
- malformation_ratio: 0.340 → ≥ 0.50
不要求 0.90，但必须比 v1 有实质提升。
