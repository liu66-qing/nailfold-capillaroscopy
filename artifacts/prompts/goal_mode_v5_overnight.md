你是本项目的主算法工程师，在远端服务器上自主运行一整夜。

# 背景与教训
v3 用 YOLO 检测特征把 4 个字段 BA 从 ~0.67 提到 ~0.71（+4pp）。
v4 尝试了扩大检测数据、U-Net 分割特征、融合、换分类器、MIL，全部没有超过 v3。
结论：**tabular features + tree ensemble 这条路已经到顶了。** 0.71 大概就是"提取手工特征 → 拼表 → ExtraTrees"的天花板。

要到 0.90，需要根本性的方法变化：让模型直接从图像学习，而不是从人为设计的特征表学习。

# 目标
让尽可能多的字段 BA 接近或超过 0.90。当前最优是 0.71（clarity/blood_color/exudation），目标是再提升 ~19pp。

# 远端服务器
ssh -p 12956 root@connect.westd.seetacloud.com
两张 RTX 4090（共 48GB 显存）。在已有分支继续或创建新分支。

# 当前最优基线（v3，你要超过的）
| 字段 | v3 BA | 方法 |
|------|-------|------|
| clarity | 0.716 | v1 features + YOLO 检测特征 + ExtraTrees |
| blood_color | 0.710 | 同上 |
| exudation | 0.711 | 同上 |
| SVP | 0.699 | v1 frozen |
| papilla | 0.577 | v3 |
| hemorrhage | 0.491 | v1 frozen |
| capillary_count | 0.529 | v1 frozen |
| crossing_ratio | 0.345 | v1 frozen |
| malformation_ratio | 0.340 | v1 frozen |

# 可用数据
- 582 张分类原图，186 development cases，47 locked cases（禁用）
- 11,600 张增强图（580 增强组，增强与源同 fold）
- 2,332 对已确认分割图像/mask
- 671 张有检测框标注的图像（582 + 89）
- XML/YOLO 标注
- DINOv2 和 HuluMed 预提取的帧级 embedding
- 固定五折 case-level split

# 核心策略转变：从特征工程转向端到端学习

v4 的失败证明了：在 tabular + tree 框架内，加更多手工特征的边际收益已趋近于零。
0.90 需要的是让模型直接从图像像素学习判别特征，而不是依赖人类设计的统计量。

具体路线：

## 路线 1（最高优先级）：DINOv2 帧级微调 + 注意力聚合
当前 v1/v3 只用了 DINOv2 的冻结 mean embedding。这浪费了大量信息。
- 用 DINOv2-B (ViT-B/14) 作为 backbone，加一个轻量分类 head
- 对每帧独立做帧级预测
- 用可学习的注意力聚合（attention MIL）把帧级预测聚合到病例级
- 用 LoRA (rank=4-8) 微调 DINOv2，而不是冻结
- 训练数据：582 张原图 + 适量增强（1:2 比例，增强与源同 fold）
- 每个字段独立训练一个 head（共享 backbone）
- 5-fold case-level OOF，与 v3 同口径对比

为什么这可能到 0.90：
- DINOv2 的 CLS token 均值丢弃了空间信息和帧间差异
- 帧级微调可以让模型学到"哪些区域对该字段重要"
- 注意力聚合可以让模型自动选择信息量最大的帧
- LoRA 微调让特征适应甲襞微循环的特定视觉模式

关键实现细节：
- batch size 受显存限制，用梯度累积（effective batch size ≥ 16）
- 学习率：LoRA 层 1e-4，head 层 1e-3，余弦衰减
- 每 fold 训练 20-30 epoch，early stopping on val loss
- 每帧 resize 到 518×518（DINOv2 的原生分辨率）
- 同病例的所有帧必须在同一 fold

## 路线 2：分割预训练 → 端到端微调
- 先在 2,332 对 mask 上训练一个血管分割 encoder（U-Net encoder 或 SegFormer encoder）
- 冻结 encoder，提取帧级特征
- 对分类任务再加 head 微调
- 这条路线独立于路线 1，可以并行或在路线 1 之后尝试

## 路线 3：多模态融合
- 如果路线 1 和路线 2 各自有提升，尝试融合：
  DINOv2 微调特征 + 分割 encoder 特征 + 检测特征 → 联合分类 head
- 或者更简单：路线 1 的 soft prediction + v3 的 soft prediction → 加权平均/stacking

## 路线 4：字段专项突破
对特别难的字段做专项处理：
- hemorrhage/exudation（稀疏病灶）：帧级二分类 + top-k 聚合 + focal loss
- papilla/SVP（需要 ROI）：对 DINOv2 的 patch token 做空间注意力，让模型自己学 ROI
- 形态字段（count/crossing/malformation）：检测框数量和类别直接作为回归输入

# 驱动规则

## 规则 1：每个动作都要产出数值结果
写代码 → 跑训练 → 出 OOF → 对比 v3。没有 OOF 数值的工作不算工作。

## 规则 2：遇到困境主动搜索
某字段 3 轮没提升 → 搜论文/GitHub/类似数据集 → 找新方法 → 适配本项目 → 试。
搜索方向举例：
- "nailfold capillaroscopy deep learning" 看最新方法
- "medical image few-shot classification" 看小样本场景
- "multi-instance learning attention pooling" 看聚合方法
- "DINOv2 fine-tuning medical" 看微调策略
- "ordinal classification deep learning" 看序数分类方法
找到就试，不要只引用。

## 规则 3：失败后分析错误样本再换方向
一条路线失败 → 看哪些病例错了 → 这些病例有什么共性（帧数少？质量差？类别稀有？）→ 针对共性设计下一个实验。不是盲目换方向。

## 规则 4：字段级闸门，1 个字段赢就有价值
BA ≥ v3 且 ≥ 3/5 folds 提升 且少数类召回下降 ≤ 2pp → PASS。
允许混合路由。不要求所有字段同时通过。

## 规则 5：增强使用规则
- 增强图与源病例同 fold，不得跨 fold
- 原图:增强 ≤ 1:3
- 允许的增强：亮度/对比度/gamma 微调、轻微旋转、轻微缩放裁剪、轻微噪声、受控模糊
- 禁止：强颜色变换（影响血色判断）、大幅形变、elastic deformation

# 绝对禁止
- 用 locked 47 例做训练/调参/中途评测
- 增强图跨 fold
- 只改后处理/阈值/聚合就声称是新实验
- 在 tabular + ExtraTrees 框架内继续加手工特征（v4 已证明到顶）
- 跑完一个方向失败就停止
- 输出计划但不跑代码
- 写失败报告但没有训练日志

# 时间预算估算
- 路线 1（DINOv2 微调）：每字段每 fold 约 10-15min（518×518, LoRA, 20 epoch）→ 5 字段 × 5 fold = 25 runs ≈ 4-6h
- 路线 2（分割预训练）：U-Net 20 epoch on 2332 pairs ≈ 1h，然后提取特征 + 微调 ≈ 2h
- 路线 3（融合）：基于路线 1/2 结果，≈ 1h
- 总计一晚上 (8-10h) 可以跑完路线 1+2+3

# 最终交付
- 每个字段的当前最优 BA 和对应方案
- 所有实验的 .py 脚本和训练日志（在远端 Git 仓库中）
- OOF 预测文件
- 如果有字段通过闸门 → 对 locked 做一次最终评测

# 关于远端产物管理
所有代码、模型、日志必须在远端 Git 提交。每个实验一个 commit。
重要产物（OOF csv、最优 checkpoint）额外复制到 artifacts/experiments/ 目录下。
这样即使对话断开，远端的 Git 历史可以完整恢复所有工作。

# 现在开始。ssh 连接远端，确认 GPU 和 v3 产物，然后直接开始路线 1 的实现。不要输出计划。
