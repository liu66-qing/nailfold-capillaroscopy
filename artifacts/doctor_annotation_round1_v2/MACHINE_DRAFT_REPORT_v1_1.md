# 首轮局部金标准机器预标注报告

## 范围与边界

- 评估角色：development only。
- 病例数：10。
- locked 集：0 例；本轮未读取或使用既有 47 例 locked 集。
- 机器草稿不是金标准，不能用于训练、模型选择或性能宣称。
- 只有用户纠正并标记完成的 correction JSON 才能进入后续 gold subset。
- 几何坐标为像素坐标；设备比例尺未确认前不换算为微米。

## 生成规则

- 来源：`round0_multiclass_domain_adapted.json` 的 development 分割预测。
- loop 候选类别：`normal`、`abnormal`、`capillary`。
- actionable draft 阈值：confidence >= 0.25。
- 合格但低于阈值的候选不删除，只在每例的 `low_confidence_candidate_count` 中计数，作为待复核背景。
- `hemo` 预测仅记录为 hemorrhage candidate，不能视为出血阳性。
- 默认可测性只是尺寸启发式（bbox 高度 >= 25 且宽度 >= 3），必须由用户纠正。

## 事实结果

| 病例 | 原始 loop 候选 | 低置信背景 | actionable draft | hemo 候选 |
|---|---:|---:|---:|---:|
| recovered_archive1/1 | 24 | 17 | 7 | 8 |
| recovered_archive1/100 | 21 | 18 | 3 | 6 |
| recovered_archive1/24 | 6 | 2 | 4 | 8 |
| recovered_archive1/28 | 13 | 7 | 6 | 7 |
| recovered_archive1/29 | 70 | 49 | 21 | 6 |
| recovered_archive1/30 | 73 | 51 | 22 | 2 |
| recovered_archive1/31 | 48 | 37 | 11 | 3 |
| recovered_archive1/32 | 0 | 0 | 0 | 0 |
| recovered_archive1/33 | 47 | 33 | 14 | 16 |
| recovered_archive1/34 | 72 | 46 | 26 | 6 |
| **合计** | **374** | **260** | **114** | **62** |

## 用户纠正协议

请对每个可见 loop 做以下决定：保留、删除、合并或新增；并补充 `apex_point`、`afferent_branch_point`、`efferent_branch_point`。同时标记 `visible`、`measurable`，在 glare/shadow/不可用区域写入 `unusable_regions`。对出血候选逐项确认阳性或伪影；不要依据机器置信度直接确认。

首例建议从 `recovered_archive1/1` 开始：机器草稿 7 个 actionable loop，17 个低置信背景候选，8 个 hemo 候选。对应 overlay：`draft_overlay/recovered_archive1__1__CAPorg3.jpg`。
