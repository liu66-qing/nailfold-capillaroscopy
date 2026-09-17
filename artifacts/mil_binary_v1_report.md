# 二级字段 Bag-level Attention MIL 诊断报告

## 执行边界

- 输入：DINOv2 `1708 x 768` 帧特征，186 个 development 病例。
- 训练：5 折分层 CV；每折 test fold、下一折 validation，其余三折 train；5 seeds（17/29/43/71/101）。
- 设备：CPU，报告 `gpu_used=false`。
- locked 隔离：权威清单 `locked_evaluation_v1.csv` 中 47 例未进入输入；每折 `locked_cases_seen=0`。
- 原 `train_mil_features.py`、v1 基线和既有结果未覆盖。
- 方法是病例 bag 内 attention pooling 后的 bag-level 监督，不是帧后预测聚合。

## 标签映射

- clarity：清晰 vs 不清/模糊
- blood_color：暗红/暗紫 vs 浅红/淡红
- exudation：无 vs +/++/+++
- subpapillary_venous_plexus：不见 vs 可见1排/可见2排/>2排,扩张
- papilla：平坦 vs 浅波纹状/波纹状（探索性，不参与 checkpoint objective，也不计入交付均值）

## 事实结果

| 字段 | MIL 5折5seed mean BA | 既有二级 GBT OOF BA | 判定 |
|---|---:|---:|---|
| clarity | 0.6574 | 0.7150 | 未显示收益 |
| blood_color | 0.6345 | 0.6940 | 未显示收益 |
| exudation | 0.6889 | 0.7570 | 未显示收益 |
| subpapillary_venous_plexus | 0.6933 | 0.6520 | 仅为独立协议下较高，不能据此宣称优越 |
| papilla（探索性） | 0.6406 | 0.5400 | 不作交付结论 |
| 4 个交付字段均值 | **0.6685** | **0.7045** | 当前配置不替代 GBT |

MIL 均值是 25 个 fold-seed test 结果的平均；不是把测试集用于训练或调参后的单一分数。每折同时保存病例级预测、混淆矩阵和 attention 记录。

## 已证据支持

1. 仅保留五个二级分类头可在 CPU 完成病例级 attention MIL 训练。
2. blood_color 的暗/浅映射、locked 隔离和 186 例开发集边界均被程序化校验。
3. papilla 作为探索性字段训练，但没有进入交付字段 checkpoint 目标。

## 尚不能判断

1. 当前实验不能证明 MIL 优于二级 GBT；整体结果反而较低。
2. 不能据此推断 attention 已解决帧与病例标签弱对应问题。
3. 不能外推到 47 例 locked 集；本阶段没有进行 locked 评估。

## 下一步判定

暂停用这版纯 DINOv2 MIL 替换 GBT。下一步仅在 development CV 内进行手工 geometry/color/texture 特征拼接或更强正则化的受控消融；每次仍须独立输出脚本和报告，不触碰 locked 集和 v1 基线。
