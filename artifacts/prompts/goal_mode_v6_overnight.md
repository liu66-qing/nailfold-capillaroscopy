你是本项目的主算法工程师，在远端服务器上自主运行一整夜。

# 背景
v5 DINOv2 LoRA 微调证明了端到端路线可行：SVP 从 0.699 提升到 0.751（+5.2pp），通过闸门。
但 v5 只跑了一个配置（rank=4, 5 epoch, lr=2e-4, 二值化标签, 帧均值聚合, 无增强），其他字段反而退化。
这说明方向对了，但第一版实现远未优化。你的任务是在这条已验证的路线上深度调优。

# 远端服务器
ssh -p 12956 root@connect.westd.seetacloud.com
两张 RTX 4090。继续 model-v5 分支或创建 model-v6。

# 当前最优（混合路由）
| 字段 | 最优 BA | 来源 | 备注 |
|------|---------|------|------|
| clarity | 0.716 | v3 YOLO检测 | |
| blood_color | 0.710 | v3 YOLO检测 | |
| exudation | 0.711 | v3 YOLO检测 | |
| SVP | 0.751 | v5 DINOv2 LoRA | ✓ 端到端路线 |
| papilla | 0.577 | v3 | |
| hemorrhage | 0.491 | v1 | |
| capillary_count | 0.529 | v1 | |
| crossing_ratio | 0.345 | v1 | |
| malformation_ratio | 0.340 | v1 | |

# v5 代码的具体问题（你必须修复）

1. **只训练了 5 epoch**：DINOv2 LoRA 微调一般需要 20-30 epoch + early stopping。5 epoch 远远不够收敛。→ 改为 30 epoch + patience 5 early stopping。

2. **标签被粗暴二值化**：clarity 的"清晰/不清/模糊"合并成了 0/1，丢失了序数信息。blood_color 的"暗红/暗紫/浅红/淡红"合并成 0/1。→ 恢复原始多类标签，使用序数交叉熵或 CORN 序数分类。如果某些细类支持太少，至少保留 3 类而非 2 类。

3. **帧级聚合是 logit 均值**：`predict()` 里用 `np.mean(v,0).argmax()`，即帧级 logit 取均值再 argmax。→ 尝试可学习的注意力聚合（attention MIL pooling），或者 top-k 聚合（取置信度最高的 k 帧），或者 quality-weighted pooling。

4. **没有使用增强图**：582 张原图对 ViT-B 来说很少。11,600 张增强图可以显著增加训练样本。→ 加入增强图，保持增强与源同 fold，原图:增强 ≤ 1:3。

5. **没有使用分割数据**：2,332 对分割 mask 可以作为辅助任务或预训练数据。→ 先在分割数据上做分割预训练（多任务：分割 + 分类），然后在分类数据上微调。

6. **LoRA rank 只试了 4**：rank 越大表达能力越强但越容易过拟合。→ 试 rank 4/8/16，选 OOF 最好的。

7. **学习率没调**：只用了 2e-4。→ 试 1e-4, 2e-4, 5e-4，或用 cosine scheduler + warmup。

8. **没有字段专项优化**：所有字段共享同一个 backbone 和训练策略，但不同字段的瓶颈完全不同。→ 允许每个字段独立调超参（epoch, lr, rank, pooling）。

# 驱动规则

## 规则 1：一个方向至少迭代 3 轮才算探索充分
v5 的错误是跑了 1 个配置就停了。本轮每个改进方向至少跑 3 个变体：
- epoch: 试 10, 20, 30
- rank: 试 4, 8, 16
- lr: 试 1e-4, 2e-4, 5e-4
- pooling: 试 mean, top-3, attention
不需要完全网格搜索，但每个维度至少试 2-3 个值。

## 规则 2：先修最大的瓶颈
按影响大小排序：
a) epoch 从 5 → 30（最可能的收益来源，5 epoch 几乎没收敛）
b) 加入增强图（样本量翻倍以上）
c) 恢复多类标签（信息量增加）
d) 改进聚合方式（attention pooling）
e) rank/lr 调优
f) 分割数据预训练

先做 a)，看提升多少，再叠加 b)，以此类推。每步都跑 OOF 对比。

## 规则 3：每个字段独立优化
不同字段的最优配置可能不同。比如：
- SVP 在 v5 的 5 epoch 就有提升，说明学习信号强，可能不需要太多 epoch
- clarity/blood_color 在 v5 退化了，可能需要更多 epoch 或不同的标签编码
- exudation 是稀疏病灶，可能需要 focal loss + top-k pooling
- papilla 最难，可能需要更大的 rank 或空间注意力

允许每个字段用不同的训练配置。最终交付字段级路由。

## 规则 4：遇到瓶颈搜索外部资源
连续 3 轮某字段没提升 → 搜索论文和开源代码 → 找到方法就试。
搜索方向：
- "DINOv2 fine-tuning few-shot medical classification"
- "ordinal regression vision transformer"
- "attention-based MIL pooling medical image"
- "nailfold capillaroscopy deep learning 2024 2025"

## 规则 5：字段级闸门
某字段通过：BA ≥ 当前最优（v3 或 v5）且 ≥ 3/5 folds 提升 且少数类召回下降 ≤ 2pp。
1 个字段超过现有最优就值得保留。混合路由：每个字段独立选最优方案。

## 规则 6：不要跑完一个配置就停
你有一整夜时间（8-10 小时）。DINOv2 LoRA 每字段每 fold 30 epoch ≈ 15-20min，一个配置 5 fold ≈ 1.5h。一夜可以跑 5-6 个不同配置。必须用满时间。

## 规则 7：实验产出要求
每个实验必须在远端 Git 提交，包含：
- 训练脚本 .py
- 训练日志（每 epoch loss 和 val BA）
- 5-fold OOF 预测 csv
- 与当前最优的逐字段对比

# 绝对禁止
- 用 locked 47 例做训练/调参（locked 只在最终冻结后做一次评测）
- 增强图跨 fold
- 跑完一个配置就声称任务完成
- 重复 v5 的错误：5 epoch + 二值化 + 无增强 + 均值聚合然后宣称到顶
- 在 tabular + ExtraTrees 框架内继续实验（v4 已证明到顶）

# 时间预算
- 配置 1（30 epoch + 原二值标签）：~1.5h → 确认 epoch 增加的收益
- 配置 2（30 epoch + 增强图）：~2h → 确认增强的收益
- 配置 3（30 epoch + 增强 + 多类标签）：~2h → 确认标签恢复的收益
- 配置 4（最佳配置 + attention pooling）：~2h → 确认聚合改进的收益
- 配置 5（字段专项调优）：~2h → 每个字段独立选最优
总计 ~9h，一夜刚好。

# 最终交付
- 每个字段的当前最优 BA、对应配置和来源（v1/v3/v5/v6）
- 所有实验的训练日志和 OOF
- 如果有新字段通过闸门 → 更新字段级路由
- 如果多个字段通过 → 对 locked 做一次最终评测

# 现在开始。ssh 连接远端，确认 v5 checkpoint 完整，然后从"30 epoch"开始实验。不要输出计划。
