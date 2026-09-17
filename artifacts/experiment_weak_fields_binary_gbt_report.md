# 二级弱字段 GBT OOF 实验

仅在 186 例 development 上使用已有 `geometry_deploy_v2` 特征，五折病例 OOF；47 例 locked 未进入特征聚合（`locked_cases_seen=0`），GPU 未使用，v1 未修改。

| 字段 | 二级 OOF BA | 有效例数 | 判定 |
|---|---:|---:|---|
| clarity | 0.715 | 185 | 可进入候选交付 |
| blood_color | 0.694 | 181 | 可进入候选交付 |
| exudation | 0.757 | 183 | 可进入候选交付 |
| subpapillary_venous_plexus | 0.652 | 184 | 可进入候选交付 |
| papilla | 0.540 | 185 | 当前不可靠，需单独处理 |

已证据支持：二级标签 GBT 在前四个字段达到开发 OOF 可用水平，渗出最高；乳头二分类仍只有 0.540，且 flat 类召回 0.207。

尚不能判断：不能把开发 OOF 结果写成产品上线收益，也未验证强保护几何字段非退化；下一步应保留原始粒度对照并在新 locked 集一次性评估。

明细见 `experiment_weak_fields_binary_gbt/report.json` 和 `oof_predictions.csv`。
