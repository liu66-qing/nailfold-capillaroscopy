# 附件 C：根因假设与论文检索指引

## 必须验证的根因假设（按优先级排序）

| # | 根因假设 | 验证方法 | 对应实验 |
|---|---------|---------|---------|
| 1 | 病例均值池化稀释局灶事件 | 对比 mean vs top-k 在出血/渗出字段的 OOF | E2, E5 |
| 2 | 低质量帧拉低病例级预测 | 去掉 unreadable 帧后对比 OOF | E1 |
| 3 | 数量/形态用连通域代理而非实例+图结构 | 用骨架+端点+分支特征替代，对比 OOF | E4, E6 |
| 4 | 出血等少数类支持不足，类别权重无效 | focal loss + 阈值校准 vs baseline | E3 |
| 5 | 缺少像素标定，绝对微米测量无依据 | 检查标定信息，若无则标 HOLD | E8 |
| 6 | SVP/乳头无稳定 ROI | ROI proposal 实验 | E7 |
| 7 | 缺失/不可见/阴性/提取失败被混编 | 检查标签编码，如有问题先修标签 | 阶段 0 |
| 8 | 预计算特征/增强组/视频泄漏 | 哈希审计 + fold 隔离检查 | 阶段 0 |
| 9 | 多种子/多路线挑选偏差 | 固定种子，看 OOF 方差 | 所有实验 |
| 10 | 设备/光照/帧数分布漂移 | 按设备/会话分层看指标 | 阶段 1 |

每个根因必须通过代码、数据或受控实验验证，不能只凭推理。

## 论文检索方向
检索关键词（优先顶会和医学影像期刊）：
- medical image multi-instance learning
- weakly supervised frame aggregation
- vessel instance segmentation
- centerline and vascular graph reasoning
- ordinal classification for medical grading
- uncertainty and selective prediction
- few-shot medical segmentation
- topology-aware loss
- calibration and abstention
- quality-aware pooling for medical bag-of-frames

论文使用规则：
- 只提供候选方法和反方观点，不直接作为事实
- 记录：论文标题、与当前数据的相似性、关键假设、是否有官方代码、是否在本项目复现、失败原因

## 子任务并行（可选）
如果环境支持：
- 子任务 A：远端模型/checkpoint/训练脚本/Git 历史盘点
- 子任务 B：数据/哈希/fold 泄漏/增强组/特征来源审计
- 子任务 C：论文和相似方法检索
- 子任务 D：实验实现/CPU smoke test/配置验证

子智能体不能直接修改 v1、locked 或默认分支。所有结论由主代理用独立证据复核。

## DeepSeek 反方复核（可选）
如果环境中已合法配置 DeepSeek API，可在以下时机调用进行独立反方复核：
- 根因判断争议、方法选择不确定、实验结果异常、可能泄漏、过度解释风险
- DeepSeek 输出只作为候选假设，必须回到代码/数据/论文/实验验证
- 不得绕过认证、暴露 API key、将 key 写入 Git/日志
