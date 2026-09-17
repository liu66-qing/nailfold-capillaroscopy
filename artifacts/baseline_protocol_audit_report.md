# dual_seg Baseline 协议统一审计

## 审计结论

当前正式 baseline 已被精确复现：审计脚本生成的 `deployment_all_non_test` 与 `current_dual_seg_binary_v1/oof_predictions.csv` 共 903 条字段-病例预测逐条一致，预测差异为 **0**。

统一后的正式协议：

- 数据：186 个 development 病例；47 个 locked 病例不读取、不评估。
- 特征：DINOv2 帧均值 + HuluMed 帧均值 + 105 个 seg feature + frame_count。
- 模型：ExtraTrees，300 trees，min_samples_leaf=2，class_weight=balanced。
- 每个 test fold 使用其余 4 folds 全部训练；不存在调参，因此不需要 validation fold。
- 随机种子：20260828 + test_fold。
- 主指标：合并全部病例 OOF 预测后计算一次 balanced accuracy；不使用各折 BA 的无权平均作为主指标。

正式 4 个交付字段 OOF mean BA：**68.477988%**。

## 差异来源

| 协议 | 全 OOF 交付均值 BA | 无权 fold 均值 BA |
|---|---:|---:|
| 正式：其余4折训练，seed=20260828+fold | 68.4780% | 68.6470% |
| nested：排除 validation，seed=20260828+fold | 70.5313% | 71.2424% |
| nested：排除 validation，seed=17+fold | 68.7672% | 69.2107% |

因此此前出现的 68.48%、69.21%、70.53% 分别来自不同协议，并非模型性能自然波动。

## 对既有实验判定的影响

- TTA v2 使用正式 baseline 训练协议，68.48% 对 68.63%，结论 `rejected` 仍有效。
- LoRA 使用严格 train/validation/test，直接 OOF 为 66.60%，低于正式 baseline；作为独立方法未显示收益，但与 ExtraTrees 的训练样本协议并非完全同构。
- 受控分类器 benchmark 的 optimized 模型在 train+validation 重训，而其 baseline 只用 train，baseline 对照不公平；其 `rejected` 方向性结论不能作为精确增益证据。
- 阈值校准脚本没有在选定阈值后用 train+validation 重训模型，因此 70.53% 对 68.32% 的差值不能作为正式结论，需标记为协议无效，而不是模型失败。

## 后续强制规则

任何需要 validation 选择参数或阈值的实验，必须：train 拟合候选、validation 选择、随后 train+validation 重训、test 只预测。其 baseline 必须使用同一 train+validation 样本和正式固定参数。所有报告统一使用合并 OOF BA，并另列 fold BA，不再用 fold 均值代替总体 OOF 指标。
