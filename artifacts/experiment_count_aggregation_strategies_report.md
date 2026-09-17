# 管袢数聚合策略扫描

基于已有 181 例 development OOF 的计数摘要，比较不同阈值与均值/中位数/q25/q75/max 统计量。没有重跑分割，也没有使用 locked 特征；`locked_cases_seen=0`，GPU 未使用。

最佳候选为 `count_t0p5_median`，BA `0.5973`；其次 `count_t0p5_q75`，BA `0.5920`。现有 `step7_count_gbt` OOF BA 为 `0.5287`，但两者不是同一个模型，不能宣称确定提升。

已证据支持：阈值和聚合统计量对计数分类有明显影响；`t=0.50` 中位数值得在独立开发实验中进一步复核。

尚不能判断：上四分位数或最大值是否在真实帧级推理中优于均值；当前只有病例摘要，没有原始帧实例/连通域，因此本实验不是已验证的后处理修复。

明细见 `experiment_count_aggregation_strategies/report.json` 和 `strategy_metrics.csv`。
