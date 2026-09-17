# 甲襞微循环四阶段实验总结（2026-08-03）

## 数据边界

- 权威清单：`code_latest/artifacts/manifest/locked_evaluation_v1.csv`
- 开发集：186 例，使用 `development_fold` 五折。
- 锁定集：47 例，未用于 schema、提示示例、阈值、训练、模型选择或路由。
- 开发集与锁定集重复组交集为 0；开发重复组未跨折。
- 报告图片及原始报告文本不作为模型输入。结构化报告字段只作为监督目标。

## Stage 1：SAM / MedSAM / 传统几何

50 图审计比较了：

- SAM2.1 Hiera Tiny
- SAM2.1 Hiera Base+
- MedSAM ViT-B
- 传统红色增强与脊线几何

Base+ 在可见管袢定位上整体最干净，但 50 图中仍有 15 例自动红旗。Tiny 有 22 例背景连通或纹理碎片风险；MedSAM 有 24 例过宽；传统方法 50/50 存在纹理碎片。

全开发集 186 例已生成四种方法的 mask、overlay 和像素域几何候选。候选包括覆盖率、面积、连通域、周长、宽度分布、骨架长度、端点和分叉。它们不是分割真值，不能直接解释为微米管径、医学管袢数或可靠交叉/畸形比例。

探索性五折结果：

- 管袢数：Base+ 宏 F1 0.422，均衡准确率 0.432。
- 交叉比例：传统几何宏 F1 0.282，均衡准确率 0.339。
- 畸形比例：Base+ 宏 F1 0.297，均衡准确率 0.301。
- 数值字段仍为报告值预测，不是校准测量。最佳 MAE 分别约为：输入支 6.94、输出支 4.44、袢顶 7.98、袢长 66.59。

## Stage 2：Qwen3-VL 零样本与少样本

Qwen3-VL-8B 的 JSON 合法率和重复稳定性均为 100%，但零样本产生严重常数输出。少样本主要改变输出先验，并未证明稳定视觉理解。没有字段达到生产路由门槛。

第一版枚举错误地包含 OCR/版式残片，已标记 `ABORTED`；第二版采用医学语义枚举并重新完成 186 例零样本及泄漏安全少样本五折。

## Stage 3：Qwen3-VL QLoRA

统一 Adapter：

- 4-bit NF4 QLoRA，冻结视觉编码器。
- 仅 assistant JSON token 计算损失。
- 语言侧注意力和 MLP LoRA，约 4365 万可训练参数。
- 五折训练每折 143--154 例；JSON 合法率均为 100%。
- 峰值训练显存约 15.24 GB；推理约 3.8--4.1 秒/病例。

相对零样本，管袢数、清晰度和出血通过平均改善与多数折方向门槛，但少数类支持不足，不能作强结论。相对少样本，只有管袢数同时改善宏 F1 和均衡准确率并满足多数折方向；其最小类支持仍不足。

字段拆分实验：

- 形态 Adapter：`clarity/capillary_count/crossing_ratio/malformation_ratio`
- 袢周 Adapter：`blood_color/exudation/hemorrhage/subpapillary_venous_plexus/papilla`

两者在 fold 0 仍明显坍缩到多数类。拆分没有解决样本量和类别不平衡问题，因此未扩大到五折。

## Stage 4：字段路由与融合

重新在 186 例开发集上评价 SigLIP2，并比较：

- SigLIP2
- Qwen3-VL QLoRA
- SAM2 Tiny / Base+ / MedSAM 像素几何
- 传统几何
- SigLIP2 + Base+ 特征融合
- SigLIP2 + 传统几何特征融合

冻结规则：分类字段要求宏 F1 和均衡准确率均改善、至少 3/5 折同向，且每折最小类支持不少于 5；数值字段要求 MAE 和中位绝对误差均改善、至少 3/5 折同向。

结果：

- `papilla`：冻结到 SigLIP2。
- 其他字段：`no_frozen_winner`，保持现有生产路由不变。
- 融合没有形成稳定的普遍增益，不应为了“用了更多模型”而上线。
- 锁定 47 例仍未打开，不用于本轮结论。

## 主要产物

- Stage 1 50 图：`artifacts/experiments/sam_stage2`
- Stage 1 全开发集：`artifacts/experiments/sam_stage2_full_dev_v2`
- Stage 2：`artifacts/experiments/qwen_stage2_cv_v2`
- Stage 3：`artifacts/experiments/qwen_stage3_qlora`
- SigLIP2 开发集五折：`artifacts/experiments/siglip2_development_cv_v1`
- 两分支融合：`artifacts/experiments/field_fusion_development_cv_v1`
- Stage 4 路由：`artifacts/experiments/stage4_routing_v1`

## 当前工程判断

1. 当前证据不支持用 Qwen3-VL 替代 SigLIP2 或几何分支。
2. Base+ 可保留为候选 mask 生成器，但没有像素真值和比例尺时不能宣称医学测量准确。
3. 下一轮最有价值的数据工作不是继续堆模型，而是补充少量高质量袢级 mask、比例尺和少数类病例。
4. 锁定集应在模型、阈值和路由完全冻结后只评估一次。
