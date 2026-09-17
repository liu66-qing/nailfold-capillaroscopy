# 17 个要求字段的可用性定论

字段清单来自 `artifacts/v1_field_specification.csv`（17 行）。此前几轮只覆盖了 11 个，
本轮补齐剩余 6 个（papilla + 4 个测量字段 + flow_velocity），所以每一行现在都有结论。

**结论：17 个字段中 4 个可用，1 个仅序数可用，11 个不可用，1 个无标签不可评。**

尺子按字段类型分：分类字段用 accuracy − 单一固定众数答案；回归字段用 MAE 对比
**固定训练折中位数**（回归没有众数基线，"永远回答中位数"才是无模型系统的真实水平）。
两者都在 2000 次案例级 bootstrap 的每次重采样内部重算基线，判定用 CI 下界 > 0。
案例级折来自 `development_fold`（NaN=locked-47，已排除），`locked_cases_seen: 0`。

## 1. 可用（4 个）

图级训练 + 5 池化等权集成，零选择。

| 字段 | 中文 | delta vs 固定众数 | 95% CI |
|---|---|---|---|
| clarity | 清晰度 | +0.330 | [+0.232, +0.373] |
| subpapillary_venous_plexus | 真皮下静脉丛 | +0.266 | [+0.179, +0.342] |
| exudation | 渗出 | +0.240 | [+0.137, +0.290] |
| blood_color | 血色 | +0.133 | [+0.033, +0.221] |

四个字段都额外满足：5 个集成成员各自单独过线；标签置换零假设为负。

## 2. 仅序数可用（1 个）

| 字段 | MAE 模型 | MAE 固定中位数 | 改善 | 95% CI | R² |
|---|---|---|---|---|---|
| loop_length 环长 | 66.40 | 87.02 | **+20.63** | [+13.45, +27.43] | 0.335 |

统计上确实跑过基线，且不是基线碰巧弱——176 例、135 个不同取值、真连续，
折间中位数稳定在 230~285。但 **MAE 66 相对 IQR 149 约占 45%**，
MAE/std = 0.62 反而**比另外三个管径字段都差**（0.33/0.51/0.57）。
它能排大小档，**不能给精确数值**，且无外部微米标定。
所以记为"仅序数可用"，不算精确测量可用。

## 3. 不可用（11 个）

**分类字段，测了没过线（7 个）**

| 字段 | 中文 | delta | 95% CI | 距过线 |
|---|---|---|---|---|
| papilla | 乳头（3 分类 42/66/77） | +0.049 | [−0.049, +0.124] | 0.049 |
| malformation_ratio | 畸形比 | +0.080 | [−0.006, +0.167] | 0.006 |
| microthrombus* | 微血栓 | +0.077 | [−0.016, +0.164] | 0.016 |
| capillary_count | 毛细血管数 | +0.049 | [−0.011, +0.110] | 0.011 |
| crossing_ratio | 交叉比 | −0.011 | [−0.068, +0.045] | — |
| rbc_aggregation* | 红细胞聚集 | −0.011 | [−0.027, +0.000] | — |
| overall_assessment* | 总体评价 | +0.000 | [+0.000, +0.000] | — |

\* 这三个不在 17 个要求字段内，是后加目标，列出以免与要求项混淆。
要求清单内实为 4 个：papilla、malformation_ratio、capillary_count、crossing_ratio。

**分类字段，算术上不可能（4 个）** — 少数类太少，任何分类器的最大可能增益都低于 0.08 噪声带，
且每一个都有整折零正例。不是模型问题，别再跑。

| 字段 | 中文 | 少数类/总数 | 最大可能增益 |
|---|---|---|---|
| hemorrhage | 出血 | 12/180 | +0.067 |
| sweat_duct | 汗腺 | 2/160 | +0.013 |
| wbc_count | 白细胞数 | 3/183 | +0.016 |
| vasomotion | 血管运动 | 3/182 | +0.016 |

**回归字段，跑不过固定中位数（3 个）**

| 字段 | 中文 | MAE 模型 | MAE 固定中位数 | 改善 | 95% CI |
|---|---|---|---|---|---|
| afferent_diameter | 输入支管径 | 8.47 | 5.74 | −2.73 | [−3.61, −1.97] |
| apex_diameter | 顶端管径 | 7.19 | 7.09 | −0.11 | [−0.74, +0.55] |
| efferent_diameter | 输出支管径 | 4.27 | 4.37 | +0.11 | [−0.37, +0.64] |

afferent 的 −2.73 不是我的配置问题：既有 artifact 里的模型 MAE 也是 7.70，同样跑不过 5.70。
根因是一个 **325 的极端离群值**（95 分位仅 20），std 25.9 几乎全由它造成。

## 4. 无标签不可评（1 个）

**flow_velocity 流速**。规格自己记录它是 `default=None` 的固定兼容规则，
备注 "placeholder has no visual measurement basis"，**没有标签列可评**。
这是"数据上不可评"，不是"测了没过"。`flow_state` 存在但是另一个 7 分类变量，不是替代品。

## 5. 复现命令

```bash
cd /root/autodl-tmp/nailfold
PY=/root/miniconda3/envs/nfc/bin/python
OUT=artifacts/experiments/variance_20260918

# 本轮补齐的 6 个字段
$PY scripts/eval_remaining_fields.py \
  --feat-dir artifacts/features_spatial/native \
  --oof-dir artifacts/experiments/threshold_tuned_20260916 \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --out $OUT/remaining_fields.json

# 此前 11 个二分类字段（交付配置）
$PY scripts/eval_imagelevel_ensemble.py --feat-dir artifacts/features_spatial/native \
  --oof-dir artifacts/experiments/threshold_tuned_20260916 \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --out $OUT/imagelevel_ensemble.json
```

## 6. 限制

1. development 186 例的结果，**不是产品能力**；locked-47 至今只允许做一次最终评估，本轮未碰。
2. **不做任何微米精度声明**；未使用 `calibration_factors.json` 作为标定证据
   （它是从标签反拟合的，存在四个互不一致取值）。
3. 回归标签量化到整数单位，这本身就限制了可达 MAE，与模型无关。
4. 病例标签广播到图，按构造带噪：某张图不一定呈现它所属病例记录的病变。
5. papilla 按原始 3 类评，未做类别合并。
6. loop_length 的"仅序数可用"是我基于 MAE/IQR 比例的判断，不是临床阈值论证——
   真正的界定需要临床方给出可接受误差范围。
7. 5 个集成成员共用同一冻结 DINOv2 编码器，误差相关，平均收益小于独立模型。
