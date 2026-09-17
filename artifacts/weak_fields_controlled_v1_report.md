# 弱字段分类器受控优化报告

## 已证据支持

- 严格使用 186 个 development 病例和既有 dual_seg 病例级特征；47 个 locked 病例未进入输入。
- 使用 5 折 OOF：每折 test 为当前 fold，validation 为下一 fold，其余为 train。
- 模型、类别权重、标准化和阈值均只由 train/validation 选择；test 仅产生 OOF 预测。
- 候选包括 LogisticRegression、ExtraTreesClassifier、HistGradientBoostingClassifier，参数搜索保持在预设有限网格内。
- 受控 baseline 交付字段均值 BA：0.6921。
- 优化后交付字段均值 BA：0.6657，变化 **-2.64 个百分点**。

| 字段 | baseline BA | optimized BA | 差值 |
|---|---:|---:|---:|
| clarity | 0.7205 | 0.6855 | -0.0349 |
| blood_color | 0.6340 | 0.6496 | +0.0156 |
| exudation | 0.7014 | 0.6863 | -0.0151 |
| SVP | 0.7126 | 0.6413 | -0.0713 |
| papilla（探索性） | 0.5903 | 0.6646 | +0.0743 |

4 个交付字段中仅 1 个提升；5 个 fold 的交付均值差值中仅 1 个为正。papilla 的提升不计入交付判定。

## 尚不能判断

- 本实验不能证明某个候选模型在 locked 集上表现；locked 集未被评估。
- 不能把 validation 上的局部提升外推为稳定泛化收益。
- blood_color 的单字段提升不足以抵消其他字段下降，也未达到整体替换门槛。

## 最终判定

**rejected**

按照预设回退规则，保留当前 dual_seg GBT baseline，不替换、不覆盖既有 step10/v1 结果。当前没有证据支持继续扩大分类器搜索空间；弱字段瓶颈继续记录为 186 例病例级弱标签限制。
