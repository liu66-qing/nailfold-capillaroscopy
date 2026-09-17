
# 补充：四个滞后字段的专家模型（hemorrhage / capillary_count / crossing_ratio / malformation_ratio）

这四个字段仍停留在 v1 水平（BA 0.34-0.53），原因各不相同，必须分别设计专家模型。

## 数据事实（你必须利用的关键信息）

582 张分类原图的 XML 标注中包含三类检测框：
- `vessel`：1,779 个框（正常血管）
- `malformed_vessel`：1,508 个框（畸形血管）
- `cross_vessel`：547 个框（交叉血管）
- 另有 1 个 `corss_vessel` 拼写错误，应归入 `cross_vessel`

这意味着：
- **每张图的 vessel + malformed_vessel + cross_vessel 框总数 ≈ 管袢总数的代理**
- **cross_vessel 框数 / 总框数 ≈ crossing_ratio 的代理**
- **malformed_vessel 框数 / 总框数 ≈ malformation_ratio 的代理**
- v3 已经训练了 YOLO 检测器。你可以直接用 v3 检测器的逐帧推理结果计算这些比值，不需要从头训练。

## 专家 1：hemorrhage（BA 0.491 → 目标 ≥ 0.70）

根因：阳性仅 12 例（168:12），均值池化淹没局灶信号。XML 无出血标注。

```
方案 A：DINOv2 LoRA 帧级二分类 + top-k 聚合
  - 复用 v5 的 DINOv2 LoRA 代码
  - 标签：病例级"有出血"复制到该病例所有帧
  - focal loss (gamma=2, alpha=0.75)，显式应对 168:12 不平衡
  - 帧级预测后，用 top-1 或 top-2 聚合（任何一帧高置信阳性 → 病例阳性）
  - 不用 mean pooling——出血是"存在性"判断，不是"平均程度"判断
  - 训练 30 epoch + early stopping
  
方案 B：YOLO 检测特征中的异常信号
  - v3 YOLO 检测器已经对每帧输出了检测框
  - 检查：出血病例的帧级检测特征（框置信度分布、异常框数量）是否与非出血病例有差异
  - 如果有 → 直接用检测特征 + 简单分类器（LightGBM + top-k pooling）
  - 这条路线零训练成本，先试

方案 C：如果 A 和 B 都不行
  - 搜索论文 "hemorrhage detection capillaroscopy" 或 "sparse lesion detection few-shot"
  - 搜索类似的稀疏事件检测方法（如视网膜微出血检测）
  - 考虑：阳性帧级支持太少（12 例 × ~8 帧 ≈ 96 帧），是否需要数据增强或 few-shot 方法
```

**先试方案 B（零成本），再试方案 A，不行再搜索。**

## 专家 2：capillary_count（BA 0.529 → 目标 ≥ 0.70）

根因：v1 用连通域数量代理管袢数。连通域 ≠ 实例。
类别分布：>=7 (125), 5-6 (42), 3-4 (14), <1 (1)。极度不平衡但多数类明确。

```
方案 A：检测框计数（最直接）
  - 用 v3 YOLO 检测器对每帧推理
  - 每帧的 vessel + malformed_vessel + cross_vessel 框总数 = 该帧管袢数估计
  - 多帧取中位数 = 病例管袢数估计
  - 将估计值映射到序数类别（>=7, 5-6, 3-4, <1）
  - 这条路线零训练成本，先试能到多少

方案 B：检测框数量作为特征 + 分类器
  - 如果方案 A 的硬映射效果一般，把每帧检测框数量（分类别的）作为特征
  - 加上置信度统计（mean, max, std）
  - LightGBM 序数分类，5-fold OOF
  
方案 C：DINOv2 LoRA + 检测特征联合
  - 把 DINOv2 embedding 和检测框特征拼接
  - 用 v5 路线做帧级序数分类
  - 多帧中位数聚合
```

**先试方案 A（零成本直接验证），有提升再优化。**

## 专家 3：crossing_ratio（BA 0.345 → 目标 ≥ 0.60）

根因：无图拓扑信息，v1 基本在瞎猜。
类别分布：<=30% (112), 30-60% (48), 60-80% (12), >80% (5)。极度不平衡。

```
方案 A：检测框比例直接计算
  - crossing_ratio ≈ cross_vessel 框数 / (vessel + malformed_vessel + cross_vessel 框总数)
  - 用 v3 检测器每帧推理 → 每帧 crossing_ratio → 多帧中位数
  - 映射到序数类别
  - 零训练成本，先验证这个代理是否与病例标签一致

方案 B：检测框比例作为特征 + 分类器
  - 如果方案 A 的硬映射不准，把 crossing_ratio 估计值和相关统计量作为特征
  - 加上 cross_vessel 框的置信度、面积分布等
  - LightGBM 分类

方案 C：如果检测框计数不够精确
  - 搜索论文 "vessel crossing detection" "vascular topology graph"
  - 考虑在 2,332 对 mask 上做骨架化 → 交叉点检测 → 用交叉点数量作为特征
  - 骨架交叉点检测是确定性算法（形态学），不需要训练
```

**先试方案 A（零成本），crossing_ratio 有现成的类别标注（cross_vessel 框），这可能是提升最快的字段。**

## 专家 4：malformation_ratio（BA 0.340 → 目标 ≥ 0.60）

根因：畸形定义依赖实例身份，分母不确定。
类别分布未在诊断报告中列出，但 v1 BA 0.34 说明几乎随机。

```
方案 A：检测框比例直接计算
  - malformation_ratio ≈ malformed_vessel 框数 / (vessel + malformed_vessel + cross_vessel 框总数)
  - 逻辑同 crossing_ratio，用 v3 检测器推理
  - 零成本先验证

方案 B/C：同 crossing_ratio 的方案 B/C
```

## 执行优先级

```
第一批（零训练成本，直接用 v3 检测器推理）：
  1. capillary_count 方案 A：检测框总数 → 管袢数
  2. crossing_ratio 方案 A：cross_vessel / 总数 → 交叉率
  3. malformation_ratio 方案 A：malformed_vessel / 总数 → 畸形率
  → 三个一起做，1 小时内出结果

第二批（轻量训练）：
  4. hemorrhage 方案 B：v3 检测特征 → 出血分类
  5. 如果第一批方案 A 效果一般 → 方案 B（检测特征 + LightGBM）

第三批（重训练）：
  6. hemorrhage 方案 A：DINOv2 LoRA + focal loss + top-k
  7. 字段专项 DINOv2 微调

第四批（搜索外部方案）：
  8. 如果以上都不行 → 搜索论文和开源实现
```

关键认知：**XML 标注里的 vessel / malformed_vessel / cross_vessel 三类检测框是 count、crossing、malformation 三个字段的直接监督信号。** v3 的 YOLO 检测器已经学会了检测这三类。把检测结果转换成比例/计数就是最直接的专家模型。先用零成本方案验证这个链路，再决定是否需要更复杂的方法。
