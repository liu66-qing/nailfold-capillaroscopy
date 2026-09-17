# 评测病例集合一致性审计

本报告为只读审计，不训练、不推理、不用 locked 集做模型选择。

## 结论

- v1 权威清单：233 例（development 186 + locked 47）。
- 多模态预测文件：198 例，6 个模型/方法组合。
- 多模态集合包含 v1 locked：42 例；包含 v1 development：156 例。
- v1 locked 未出现在多模态集合：5 例。
- v1 development 未出现在多模态集合：30 例。
- 因此，多模态记录中的“排除 35 例 locked”不是 v1 权威 47 例 locked 集的同义表述。
- 远端多模态清单：233 例（development 198 + locked 35），其 locked 与 v1 locked 重叠 5 例。

## 集合统计

| 集合 | 病例数 | SHA-256（按排序 ID） | archive1/2/3 |
|---|---:|---|---|
| v1 全部 | 233 | e532c75182cd7c0e45af1ae41a762dcd3c1b194411595af66849a4b623cef5aa | {'recovered_archive1': 72, 'recovered_archive2': 98, 'recovered_archive3': 63} |
| v1 development | 186 | 4f3aeda93a87ceea2376cbf8ae7fcb01b60245c99fc85831d4141b2490520797 | {'recovered_archive1': 58, 'recovered_archive2': 78, 'recovered_archive3': 50} |
| v1 locked | 47 | 3da51fb4b791e57ff843c321e7c718d8fc47ed1f16d532d2c5d29527065aa5ca | {'recovered_archive1': 14, 'recovered_archive2': 20, 'recovered_archive3': 13} |
| 多模态预测 | 198 | e29b1dcb353f4d2e4c490f6c0538256ccd251f37d9fffe6bf44e7c4d92a4241d | {'recovered_archive1': 61, 'recovered_archive2': 83, 'recovered_archive3': 54} |

## 标签审计

多模态 true 字段与 v1 清单的原始字符串存在格式差异，但按 canonical mapping（括号、区间别名、颜色后缀）规范化后，9 个共同字段均为 0 个不一致，且没有病例出现多个真值。

| 字段 | 原始字符串不一致行数 | 规范化后不一致行数 | 多真值病例数 |
|---|---:|---:|---:|
| clarity | 0 | 0 | 0 |
| capillary_count | 0 | 0 | 0 |
| crossing_ratio | 12 | 0 | 0 |
| malformation_ratio | 6 | 0 | 0 |
| blood_color | 24 | 0 | 0 |
| exudation | 0 | 0 | 0 |
| hemorrhage | 0 | 0 | 0 |
| subpapillary_venous_plexus | 12 | 0 | 0 |
| papilla | 18 | 0 | 0 |

## 协议判定

当前多模态结果不能与 v1 locked 结果作严格 head-to-head 性能结论。下一步应冻结同一病例集合，先做 development 5 折 OOF，再对同一 locked 集一次性评测。
