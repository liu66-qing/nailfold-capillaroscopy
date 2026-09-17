# 交付规格 v2：图级训练 + 固定集成

前置结论（不变）：**可用字段仍是 4 个**。本轮没有把任何新字段推过门槛。
本轮真正的成果是：四个已可用字段的成绩在**完全不做选择**的前提下提高了，
而且两个临界字段（malformation_ratio、microthrombus）的"看起来过线"被自己的对照组否证。

尺子未变：accuracy − 单一固定众数答案；案例级折来自 `development_fold`（NaN=locked-47，已排除）；
基线在 2000 次案例级 bootstrap 的**每一次重采样内部**重算；判定用 delta 的 CI 下界 > 0。
无 class_weight、无阈值调优、无内层验证折。locked-47 全程 `locked_cases_seen: 0`。

## 1. 本轮改了什么

上一轮的诊断是"选择机制在加方差"。对症的做法是**平均**而不是挑选，而上一轮只测了挑选。
本轮两个机制，都在决定之前固定，对所有字段完全一致：

1. **图级训练**。此前每个病例被压成 1 个向量（n=186 行）。改为在 1708 张图上拟合
   （病例标签广播到它的图，预测再聚合回病例），拟合行数约 9 倍。
   评估单位仍是病例，尺子不变。按**病例**切折，没有病例的图跨折。
2. **固定等权集成**。5 种池化（mean/topk_mean/max/cls/std）各出一份概率，等权平均，阈值 0.5。
   成员、权重、聚合方式全部预先固定，没有可过拟合的自由度。

这不是已封闭的冻结特征 MIL：那条路在**视频帧**上池化并且用案例级探针；
这里的单位是不同手指/视野的静态图，图是训练行而不是被池化的输入，聚合发生在**预测**上而非特征上。

## 2. 交付表（4 个模型字段）

| 字段 | delta vs 固定众数 | 95% CI | 单成员独立过线 | 标签置换零假设 |
|---|---|---|---|---|
| clarity | **+0.330** | [+0.232, +0.373] | 5/5 | −0.004 |
| subpapillary_venous_plexus | **+0.266** | [+0.179, +0.342] | 5/5 | −0.045 |
| exudation | **+0.240** | [+0.137, +0.290] | 5/5 | −0.013 |
| blood_color | **+0.133** | [+0.033, +0.221] | 5/5 | −0.048 |

对比上一轮固定配置：clarity +0.308→+0.330，SVP +0.250→+0.266，exudation +0.235→+0.240，
blood_color +0.177→+0.133（**降低**，见 §4）。

四个字段都满足两个额外条件：5 个成员**各自单独**过线（成绩不是某一个幸运池化撑起来的），
标签置换后 delta 为负（没有病例身份泄漏）。

其余 10 个字段交付固定众数答案，与 v1 相同：四个算术不可能字段
（sweat_duct 2/160、wbc_count 3/183、vasomotion 3/182、hemorrhage 12/180）
加 capillary_count、microthrombus、malformation_ratio、crossing_ratio、overall_assessment、rbc_aggregation。

## 3. 两个临界字段：被自己的对照组否证

| 字段 | 单池化最好（挑出来的） | 五池化等权平均 | 判定 |
|---|---|---|---|
| malformation_ratio | std +0.123 CI[+0.037,+0.204] 过线 | **+0.080 CI[−0.006,+0.167]** | 不可用，差 0.006 |
| microthrombus | max +0.115 CI[+0.033,+0.197] 过线 | **+0.077 CI[−0.016,+0.164]** | 不可用，差 0.016 |

两个字段都只有 2/5 和 1/5 的池化能单独过线。读那个最好的池化就是上一轮刚否证的选择偏差，
所以这里以等权平均为准：**都没过**。差距 0.006 和 0.016，比 v1 的 0.027/0.043 近了，但仍在线下。

## 4. 图级训练的真实归因

同一段代码、同一估计器，只改"训练行是图还是病例均值"：

| 字段 | 图级增益（等权集成下） | 5 池化中图级胜出数 |
|---|---|---|
| clarity | +0.032 | 5/5 |
| capillary_count | +0.033 | 5/5 |
| malformation_ratio | +0.031 | 3/5 |
| rbc_aggregation | +0.022 | 5/5 |
| subpapillary_venous_plexus | +0.011 | — |
| exudation | +0.000 | 3/5 |
| microthrombus | −0.005 | 3/5 |
| blood_color | −0.011 | 0/5 |
| crossing_ratio | −0.011 | 3/5 |

图级训练在多数字段上有小幅正增益，中位数 +0.011~+0.044（按池化算），
但**不是普遍有效**：blood_color 上 0/5 胜出、净 −0.011。
所以 blood_color 的交付值从 +0.177 降到 +0.133——我按统一配置报数，不为它单独换配置，
否则就是逐字段选择，那已被证明是负收益。
如果只看这一个字段，案例级 +0.144 更好；差异 0.011 在噪声带内，不足以支持给它开特例。

## 5. 病例数：没有隐藏的可用数据

`cases.csv` 有 267 例，评估清单只有 233 例（186 dev + 47 locked）。差的 34 例已核查：
33 例 `report_present_field_count = 0`（完全无标签），第 34 例有 20 个字段但 0 张可用图。
**"加病例"在现有数据里是死路**，三个临界字段缺的是真实新病例，估计 microthrombus 需 n≈340。

## 6. 复现命令

```bash
cd /root/autodl-tmp/nailfold
PY=/root/miniconda3/envs/nfc/bin/python
OUT=artifacts/experiments/variance_20260918

# 交付配置（图级 + 五池化等权集成，含案例级对照与置换零假设）
$PY scripts/eval_imagelevel_ensemble.py \
  --feat-dir artifacts/features_spatial/native \
  --oof-dir artifacts/experiments/threshold_tuned_20260916 \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --out $OUT/imagelevel_ensemble.json

# 图级 vs 案例级逐池化归因（对照 A/B/C）
$PY scripts/imagelevel_control.py --feat-dir artifacts/features_spatial/native \
  --oof-dir artifacts/experiments/threshold_tuned_20260916 \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --out $OUT/imagelevel_control.json
```

## 7. 限制

1. development 186 例的结果，**不是产品能力**；locked-47 本轮一次未碰。
2. 病例标签广播到图，按构造就是带噪的：某张图不一定呈现它所属病例记录的病变。
3. 5 个成员共用同一个冻结 DINOv2 编码器，误差相关，平均的收益小于独立模型。
4. 每例图数 4~23 不等，病例对拟合的贡献不均等。
5. 置换零假设只做 20 次，够查明显泄漏，不足以给出紧的零分布区间。
6. 等权是预先规定的，不是拟合的；拟合权重会把本设计要避免的选择方差放回来。
7. 不做任何微米精度声明，未使用 `calibration_factors.json` 作为标定证据。
