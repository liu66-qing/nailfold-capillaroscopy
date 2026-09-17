# dual_seg 字段专属阈值校准

## 已证据支持

- 严格 development-only 5 折 OOF，阈值只在 validation 选择。
- locked_cases_seen=0，gpu_used=false，输入仍为既有 dual_seg。
- baseline 交付字段均值 BA：70.53%。
- 阈值校准后交付字段均值 BA：68.32%，下降 2.22 个百分点。

| 字段 | baseline | calibrated | 差值 |
|---|---:|---:|---:|
| clarity | 72.05% | 67.39% | -4.66pp |
| blood_color | 63.40% | 64.87% | +1.47pp |
| exudation | 70.14% | 67.35% | -2.80pp |
| SVP | 71.26% | 68.38% | -2.88pp |
| papilla（探索性） | 59.03% | 58.10% | -0.93pp |

## 尚不能判断

该实验不能证明阈值在新数据或 locked 集上无效；只能证明在当前 186 例 development、固定 dual_seg 和严格 OOF 协议下，validation 阈值选择没有稳定泛化收益。

## 最终判定

**rejected**。保留默认 argmax 的 dual_seg 二级 GBT；不采用字段专属阈值校准。
