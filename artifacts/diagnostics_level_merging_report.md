# 等级合并实验

输入为现有 `geometry_deploy_v2.npy/.csv` 的病例级 mean/median/std 摘要和 `locked_evaluation_v1.csv` 的开发角色。五折 OOF 仅使用 186 例 development，`locked_cases_seen=0`，GPU 未使用，v1 未修改。

| 字段 | 原始 BA | 3 级 BA | 2 级 BA | 原始有效例数 |
|---|---:|---:|---:|---:|
| clarity | 0.457 | 0.457 | 0.698 | 185 |
| blood_color | 0.474 | 0.474 | 0.695 | 178 |
| exudation | 0.374 | 0.489 | 0.724 | 183 |
| hemorrhage | 0.500 | 0.500 | 0.500 | 180 |
| subpapillary_venous_plexus | 0.363 | 0.465 | 0.657 | 184 |
| papilla | 0.411 | 0.411 | 0.526 | 185 |

已证据支持：当前特征和固定病例折下，二级合并对除出血外的五个字段提高了开发 OOF BA；出血少数类 12 例且召回为 0，不能据此交付。

尚不能判断：这些提升是否来自产品定义更合理、类别支持增加或模型偏置；不能直接替换产品字段定义，也未测试强保护几何字段。

详细混淆矩阵、每类召回和每病例 OOF 见 `diagnostics_level_merging/report.json` 与 `oof_predictions.csv`。
