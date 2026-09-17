# 字段可用性提升实验结果 · 2026-09-17

目标（用户要求）：**让尽可能多的字段变得可以用**。

结论先说：**可交付字段数量没有增加，仍是 4 个。** 两条模型侧路线被自己的对照组否证；
但 4 个已可用字段的 delta 显著提高，且 14 个字段第一次被分成了四类互不相同的处境，
其中 4 个字段被证明在 n=186 下**无论用什么模型都不可能达标**。

所有数字均为 development 集 out-of-fold，**不是产品能力**，不得对外引用。
locked-47 全程未参与：每个特征集 `locked_cases_seen: 0`，每个字段断言 OOF 无 locked 病例。

---

## 1. 最终字段判决表

判决用两把尺子，分开测、都报告：

- `delta` = 准确率 − **每折训练集众数基线**（每次 bootstrap 内重算）→ 能否作为**标签**交付
- `AUC` = out-of-fold 判别力，与患病率无关 → 该字段**是否存在任何信号**

| 字段 | 判决 | 正例/n | 患病率 | delta | delta 95%CI 下界 | AUC | AUC 95%CI |
|---|---|---|---|---|---|---|---|
| clarity | **可交付** | 94/185 | 0.508 | +0.384 | +0.303 | 0.893 | [0.844, 0.937] |
| subpapillary_venous_plexus | **可交付** | 105/184 | 0.571 | +0.250 | +0.152 | 0.878 | [0.825, 0.926] |
| blood_color | **可交付** | 81/181 | 0.448 | +0.182 | +0.094 | 0.829 | [0.769, 0.887] |
| exudation | **可交付** | 90/183 | 0.492 | +0.284 | +0.191 | 0.801 | [0.738, 0.861] |
| microthrombus | 仅有信号 | 74/183 | 0.404 | +0.087 | −0.011 | 0.734 | [0.661, 0.799] |
| capillary_count | 仅有信号 | 57/182 | 0.313 | −0.005 | −0.093 | 0.664 | [0.574, 0.751] |
| malformation_ratio | 仅有信号 | 70/162 | 0.432 | +0.074 | −0.031 | 0.646 | [0.561, 0.728] |
| rbc_aggregation | 仅有信号 | 150/182 | 0.824 | −0.082 | −0.143 | 0.631 | [0.518, 0.737] |
| crossing_ratio | 仅有信号 | 65/177 | 0.367 | −0.023 | −0.119 | 0.617 | [0.530, 0.700] |
| overall_assessment | 无信号 | 160/185 | 0.865 | −0.032 | −0.076 | 0.613 | [0.473, 0.742] |
| sweat_duct | **n 不足** | 2/160 | 0.013 | −0.008 | −0.024 | 0.770 | [0.697, 0.844] |
| wbc_count | **n 不足** | 3/183 | 0.016 | −0.011 | −0.027 | 0.402 | [0.115, 0.717] |
| vasomotion | **n 不足** | 3/182 | 0.016 | +0.000 | +0.000 | 0.371 | [0.215, 0.530] |
| hemorrhage | **n 不足** | 12/180 | 0.067 | −0.078 | −0.117 | 0.335 | [0.200, 0.474] |

判决口径：

- **可交付**：预先指定 pooling 上 delta 的 CI 下界 > 0。
- **仅有信号**：AUC 的 CI 下界 > 0.5，但 delta 的 CI 含 0。存在真实的排序信息，
  但在该患病率下**无法承受被压成硬标签**。只能用于排序/复核分诊，**不构成标签授权**。
- **n 不足**：少数类占比 ≤ 0.08（种子噪声带）。即使完美分类器，
  delta 上限也只有 0.013~0.067，**低于噪声带**。见 §4。
- **无信号**：AUC 的 CI 含或低于 0.5。

---

## 2. 已可用字段的提升（唯一真实增量）

| 字段 | 审计 delta | 新 delta | 变化 | 新 CI | 特征 |
|---|---|---|---|---|---|
| clarity | +0.297 | +0.384 | **+0.086** | [+0.303, +0.465] | native_hi:topk_mean |
| subpapillary_venous_plexus | +0.114 | +0.250 | **+0.136** | [+0.152, +0.348] | square_baseline:topk_mean |
| exudation | +0.240 | +0.284 | +0.044 | [+0.191, +0.383] | native:topk_mean |
| blood_color | +0.149 | +0.182 | +0.033 | [+0.094, +0.282] | native:topk_mean |

只有 clarity(+0.086) 和 SVP(+0.136) 超过 0.08 种子噪声带；exudation 和 blood_color
的变化在噪声带内，**不应算作提升**。

---

## 3. 两条模型侧路线被否证（含否证方式）

### 3.1 视野假说 —— 被我自己的对照组推翻

已有特征管线是 `Resize(518) → CenterCrop(518)`，而全部 1708 张图原生 1024×768，
所以历史特征**丢弃了 25.0% 的宽度**。这是真实缺陷，已实测。

但修好它没有带来收益。关键是 `square_control`：用**新管线**跑 518×518，
看到的是和历史特征**完全相同的 75% 裁剪**。三档对比（`nat − sq` 列隔离视野效应）：

| 字段 | square(同样裁剪) | native(全视野) | native_hi | nat − sq |
|---|---|---|---|---|
| clarity | +0.362 | +0.324 | +0.362 | **−0.038** |
| SVP | +0.277 | +0.228 | +0.228 | **−0.049** |
| blood_color | +0.193 | +0.177 | +0.182 | **−0.017** |
| exudation | +0.306 | +0.317 | +0.317 | +0.011 |

视野效应最大 +0.049，且在所有真正变动的字段上**为负**。
增益来自新管线的聚合方式，不是来自多看了 25% 画面。**视野假说否证。**

### 3.2 稀疏病灶 / 空间池化假说 —— 部分成立，但救不活任何字段

机制假设：死掉的字段多是稀疏局部病灶（出血、微血栓、红细胞聚集、交叉），
一个病灶可能只占 1024×768 的不到 1%，而全局 CLS / 1813 个 patch 取均值会把它稀释约 100 倍。
于是额外提取 `max` / `topk_mean`(k=16) / `std` 三种 patch 级统计。
这与已封闭的冻结特征 MIL 机制不同——后者在**帧**上池化，从未在**空间**上池化。

结果：空间池化确实有效，但**只对已经可用的字段有效**。

| 字段 | mean | topk_mean | max |
|---|---|---|---|
| clarity | +0.368 | +0.384 | **+0.389** |
| exudation | +0.268 | +0.284 | **+0.295** |
| microthrombus | +0.066 | **+0.087** | +0.077 |
| overall_assessment | −0.076 | −0.032 | **−0.005** |
| hemorrhage | −0.078 | −0.078 | −0.067 |
| rbc_aggregation | −0.082 | −0.082 | −0.082 |
| crossing_ratio | −0.017 | −0.023 | −0.011 |

正是为之设计的稀疏病灶字段（hemorrhage、crossing_ratio、rbc_aggregation）**仍然为负**。
`std`（空间异质性）对 SVP 最好（+0.293），符合"画面均匀还是斑驳"的判读语言，
但同样没有救活任何死字段。**被救活的字段数 = 0。**

---

## 4. 4 个字段在 n=186 下不可能达标（这是本次最有决定性的发现）

这不是模型问题，是算术问题。少数类占比就是任何模型能拿到的 delta 上限：

| 字段 | 正例/n | 患病率 | 完美分类器的 delta 上限 | 最小折正例数 |
|---|---|---|---|---|
| sweat_duct | 2/160 | 1.2% | **+0.013** | 0 |
| wbc_count | 3/183 | 1.6% | **+0.016** | 0 |
| vasomotion | 3/182 | 1.6% | **+0.016** | 0 |
| hemorrhage | 12/180 | 6.7% | **+0.067** | 0 |

四者上限全部低于 0.08 种子噪声带，且**最小折正例数为 0**——按折训练本身就是退化的。
**换模型、换特征、加 GPU 都不改变这一点。**
出路只有两个：把这些字段的正例病例数补上去，或从产品字段表中移除。

`sweat_duct` 与 `wbc_count` 的审计模型对 100.0% 的病例都输出众数——这是塌缩，不是难题。
`sweat_duct` 的 AUC 0.770 不可信：只有 2 个正例，bootstrap 中缺少某一类的重采样被跳过，
CI 因此被人为收窄，不应解读为"有信号"。

---

## 5. 一个被拒绝的选项：rule-out 阈值

在 90% 灵敏度目标下，14 个字段中有 **8 个**只能在特异度 < 0.20 时达到，
即靠**把几乎所有病例都标为阳性**换来的。那里出现的 NPV ≈ 1.0 是"全部报阳"的算术产物，
不是筛查能力。**不予采纳，不进入任何交付口径。**

---

## 6. 复现命令

```bash
# 空间池化特征提取（2 卡并行，各约 1 分钟）
python scripts/extract_dinov2_spatial.py \
  --index artifacts/features/dinov2/index.csv \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --image-root /root/autodl-tmp/nailfold/data \
  --output-dir /root/autodl-tmp/nailfold/artifacts/features_spatial/native \
  --weights /root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth \
  --preset native --device cuda:0 --batch-size 16

# 字段级评估（标签逐字复用 threshold_tuned OOF，只换特征）
python scripts/eval_spatial_fields.py --features-root <F> \
  --oof-dir artifacts/experiments/threshold_tuned_20260916 \
  --roles artifacts/manifest/locked_evaluation_v1_reviewed.csv \
  --presets native square_baseline --out <O>/summary.json

# 判决与诊断
python scripts/diag_dead_fields.py
python scripts/eval_signal_auc.py --features-root <F> --preset native --oof-dir ... --out <O>/auc_native.json
python scripts/final_field_verdict.py
```

产物（新服务器 14170）：
`/root/autodl-tmp/nailfold/artifacts/experiments/spatial_20260917/`
`summary.json`、`summary_hi.json`、`auc_native.json`、`auc_native_hi.json`、
`dead_field_diagnosis.json`、`final_field_verdict.json`

特征：`/root/autodl-tmp/nailfold/artifacts/features_spatial/{native,native_hi,square_baseline}/`
各含 `features_{cls,mean,max,topk_mean,std}.npy`（float16, 1708×768）、`index.csv`、`metadata.json`。

## 7. 限制

1. 全部为 development 集 out-of-fold 结果，**不是产品能力**，外部验证仍为 0。
2. 每字段跨 55 个（preset × pooling）视图取最优，post-hoc 选择使 `best` 列乐观有偏；
   本文正文只引用**预先指定**的 `topk_mean`，`best` 仅作透明性披露。
3. 冻结编码器：`max`/`topk_mean`/`std` 只能暴露 DINOv2 在无微调时已经分开的东西，
   它们不是学出来的检测器，**不构成病灶定位**，patch 下标未保留。
4. AUC 与 delta 回答不同问题。AUC > 0.5 **不代表**字段可以上线。
5. `IMPOSSIBLE_N` 是关于 n=186 与类别分布的陈述，不是关于该体征在临床上是否真实存在。
6. 患病率是 development 集患病率，不是预期部署患病率。
