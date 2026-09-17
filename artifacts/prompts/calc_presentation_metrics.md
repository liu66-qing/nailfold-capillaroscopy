连接远端服务器，计算当前最优模型的展示指标。

ssh -p 12956 root@connect.westd.seetacloud.com

# 任务：从已有 OOF 预测文件中计算 Accuracy（不是 BA）

对以下 5 个分类字段，找到当前最优的 OOF 预测文件，计算：
1. Accuracy（整体正确率，sklearn.metrics.accuracy_score）
2. Balanced Accuracy（已有，用于内部对比）
3. 每类的样本数和召回率

当前最优路由（从 field_routing_status.csv）：
- clarity → rank8_lr1e4（dev BA 0.7332）
- blood_color → v3 detector（dev BA 0.7105）
- exudation → rank4_lr1e4（dev BA 0.7719）
- SVP → rank8（dev BA 0.8076）
- papilla → rank16（dev BA 0.6606）

OOF 文件可能在以下位置：
- artifacts/experiments/model-v6-20260901/ 下各子目录
- artifacts/experiments/model-v5-20260901/
- artifacts/experiments/model-v3-20260901/
- artifacts/evaluation/

找到每个字段对应的 OOF 预测 csv（包含 truth 和 prediction 列），用 sklearn 计算 accuracy_score。

同时从 locked 评测结果中提取连续字段的归一化分数（如果有），以及固定值字段的命中率。

最终输出一张汇总表，格式：

| 字段 | 展示指标类型 | 展示数值 | BA（内部） | 样本数 | 来源文件 |

把结果保存到 artifacts/presentation_metrics.csv

不要训练任何模型，只读取已有文件计算指标。
