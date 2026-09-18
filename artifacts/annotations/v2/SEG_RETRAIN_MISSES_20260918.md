# 补标漏检血管 + 重训分割器：结论是否证

日期 2026-09-18 · development only（186 例 / 1708 图）· locked-47 一张未用（每个环节 `locked_cases_seen: 0`）

## 0. 一句话结论

**补标 + 重训没有救活任何字段，也没有纠正"分割器漏检畸形血管"的反向梯度。**
两个互相独立的 class-agnostic 提议器（SAM、vesselness 滤波）各自召回约 6~7 个/图的"漏检"，
把它们并入训练标签重训 5 折后，实例数涨了 27%（17.4 → 20.45 /图），但
`frac_low_circ` 与畸形等级的相关系数只从 **−0.257 动到 −0.251**（p 仍 0.001），符号照旧反着。
可用字段数仍是 **4**（clarity / SVP / exudation / blood_color），本轮 11 个几何相关字段里通过的仍是 2 个（clarity、exudation），与重训前完全一致。

这一步是我上一轮设计的判别性检验。它现在倒向解释 (b)：
**那些畸形血管在图像里本就难以成形为可分割目标，是信息缺失而非标注缺失。** 补标这条路应关闭。

## 1. 做了什么

| 阶段 | 内容 | 产出 |
|---|---|---|
| 提议 | SAM（本地，`third_party_api_used: false`）对 1708 图做 class-agnostic mask，减去 YOLO c=0.25 已检出者 | 11,400 misses，6.67/图（YOLO 51.5/图） |
| 提议（第二路） | vesselness 滤波，同样口径 | 6.05/图 |
| 相关性预检 | miss 密度 vs 畸形等级 | SAM rho=+0.0948 p=0.229；vess rho=+0.1412 p=0.0722 |
| 特征侧融合 | 把 miss 统计量拼进图级几何（69 → 99 维）重跑尺子 | 见 §3 |
| 重训 | YOLO c=0.25 检出 ∪ SAM misses 作为伪标签，5 折各自排除本折病例，60 epoch | `models/aug_fold{0..4}` |
| 配对对照 | 同一管线但 `ai_miss_polygons_added = 0`（只有 YOLO 伪标签），仅 fold0 | `models/ctrl_fold0` |
| 重训后推理 | 折匹配（fold k 的病例只用 aug_fold{k}），c=0.25，imgsz 1024 | 34,932 实例，0 失败 |

## 2. 重训后的分割质量：没有变好

验证集是**人工全视野标注**的 valid split（AI 伪标签从不进 val）。

| 模型 | mask mAP50 | mAP50-95 |
|---|---|---|
| aug_fold0 | 0.7036 | 0.2305 |
| **ctrl_fold0（无 miss 标签）** | **0.7101** | **0.2347** |
| aug_fold1 | 0.7022 | 0.2275 |
| aug_fold2 | 0.7104 | 0.2371 |
| aug_fold3 | 0.6923 | 0.2218 |
| aug_fold4 | 0.6965 | 0.2325 |

唯一可配对的 fold0 上，**加了 10,166 个 miss 多边形的模型比不加的略低**（0.7036 vs 0.7101）。
在人工标注这把尺子下，这些补进去的目标不是被漏掉的血管，更像噪声。

⚠️ 对照只跑了 1 折，这是 n=1 的配对比较，不能当显著性结论；但它至少排除了"补标带来明显分割增益"。
⚠️ 这里的 mAP 与 handoff 的 0.952 不可比：口径、验证集、imgsz 都不同，别做跨表对照。

## 3. 反向梯度没有被纠正（决定性证据）

畸形等级 n=163。三套特征来源对比：

| 特征来源 | 实例数/图 | `n_inst` rho | `frac_low_circ` rho | `circularity_mean` rho |
|---|---|---|---|---|
| 原 c=0.25 | 17.4 | −0.1905 (p=0.015) | **−0.2574 (p=0.0009)** | +0.2969 (p=0.0001) |
| 原 c=0.05 | 56.6 | −0.1193 (p=0.129) | **−0.2586 (p=0.0009)** | +0.2916 (p=0.0002) |
| **重训 aug** | **20.45** | −0.1004 (p=0.202) | **−0.2507 (p=0.0012)** | +0.2228 (p=0.004) |

按畸形等级 0/1/2/3 分组的 `frac_low_circ` 均值，重训后是 0.428 / 0.417 / 0.255 / 0.334 ——
畸形越重，检出的血管反而越圆、越少，和重训前同一个方向。

正对照仍在（说明管线没坏，只是目标没被学到）：
`n_inst` vs capillary_count rho −0.382 → −0.263（p=0.0003），`nn_dist_median` +0.363 → +0.300。
正对照略微**变弱**，与实例数变多但多出来的不是真血管一致。

## 4. 特征侧融合：同样无增益

图级训练、固定 `concat` 为预注册臂、C=0.03、2000 次 bootstrap、seed 20260918。

| 字段 | 基线 c05 | +SAM miss | +vess miss | 重训 aug |
|---|---|---|---|---|
| clarity | +0.2919 | +0.2973 | +0.2865 | **+0.3081** [0.2108, 0.3514] |
| exudation | +0.2350 | +0.2295 | +0.2350 | +0.2295 |
| **malformation_ratio** | **+0.0920** | +0.0982 | +0.0859 | +0.0798 |
| capillary_count | +0.0440 | +0.0440 | +0.0495 | +0.0440 |
| crossing_ratio | +0.0056 | +0.0168 | −0.0112 | −0.0279 |
| blood_color | +0.0526 | +0.0351 | +0.0175 | +0.0175 |
| papilla | −0.0216 | −0.0378 | −0.0270 | +0.0054 |
| afferent_diameter | −0.0303 | −0.0121 | −0.0303 | −0.0303 |
| efferent_diameter | +0.0183 | +0.0183 | −0.0061 | +0.0122 |
| apex_diameter | −0.0296 | −0.0118 | −0.0178 | −0.0296 |
| loop_length | +0.0170 | +0.0227 | +0.0227 | +0.0170 |
| **通过数 / 11** | **2** | **2** | **2** | **2** |

- 三条新路线里 malformation_ratio 的最好值是 +0.0982 CI[0.000, 0.1902]，CI 下界仍压在 0，差 0.006 这件事没变。
- crossing_ratio 在重训后变成 −0.0279，比原来更差。
- clarity 的 +0.3081 比基线 +0.2919 高 0.016，但 CI 大幅重叠，且 clarity 早已是交付字段，**不构成新增能力**。

## 5. 为什么这判给"信息缺失"而不是"候选太脏"

上一轮我把两个解释列为未分开：(a) 候选大多不是真血管、信号被稀释；(b) 畸形血管在图像里本就不可见。
现在三条证据同向指 (b)：

1. **两个机制完全不同的提议器给出同一个非显著结果**（SAM +0.095/p=0.229，vesselness +0.141/p=0.072）。若真有一批可见但未标的畸形血管，换提议器应该能显著召回其中一批。
2. **重训是最强的一次"给模型答案"**：把 miss 直接写成训练目标，实例数涨 27%，反向梯度只动了 0.006。模型学到了新目标，但新目标与畸形等级无关。
3. **人工标注 valid 上没有 mAP 增益**（配对折 0.7036 vs 0.7101）。补进去的目标在人工尺子下不被承认。

若是 (a)，第 2 条应该出现梯度松动；它没有。

## 6. 尚未验证 / 不能声称

- miss 多边形**从未经人工确认**，不是金标准，也不得写回任何既有标注文件（本轮所有产出只写入新目录树）。
- 我**没有目视核对** `reports/visual/miss_{0..3}.jpg`（4 张叠加图已备份到 `artifacts/features/segmiss_v1/visual/`），所以"这些候选不是血管"是从 mAP 与相关性推出的，不是看图看出来的。需要人工过一眼这 4 张图来独立确认。
- 配对对照只有 1 折。
- 全部数字是 development，**不是产品能力**；locked-47 未做任何评估。
- 几何是像素尺度；`calibration_factors.json` 故意未用（label 反拟合，四个互不一致取值）。
- 畸形/交叉等级是区间字符串的序数重编码，不是连续真值。

## 7. 复现

```bash
# 提议（本地 SAM，不外发任何图像）
python segmiss_propose.py --mode sam --out proposals/sam_full
python segmiss_propose.py --mode vess --out proposals/vess_full
# 判别性预检
python segmiss_phase3.py --miss-case proposals/sam_full/miss_case.csv  --out reports/phase3_sam.json  --tag sam
python segmiss_phase3.py --miss-case proposals/vess_full/miss_case.csv --out reports/phase3_vess.json --tag vesselness
# 构建重训集（aug = YOLO ∪ miss；ctrl = 仅 YOLO）
python segmiss_build_retrain.py --misses proposals/sam_full/ai_proposed_misses.jsonl --out seg_aug
python segmiss_build_retrain.py --misses proposals/empty_misses.jsonl              --out seg_ctrl
# 重训与折匹配推理
python train_any.py seg_aug  {0..4} 1 aug
python extract_instance_geometry.py --weights-dir models --prefix aug_ --out features/aug_instance
python eval_geom_imagelevel.py --geom features/aug_instance/image_geometry.csv --out reports/ruler_retrained_aug.json
```

权重 sha256 见 `artifacts/features/segmiss_v1/aug_meta.json`（5 个 fold，各 45,156,982 字节）。

## 8. 对路线的影响

- **"改标注重训分割器"这条路关闭。** 这是 `nailfold-seg-misses-abnormal` 留下的唯一建议方向，现已被自身的判别性检验否证。
- crossing_ratio / malformation_ratio / capillary_count 三个字段的瓶颈不在分割器，也不在下游建模，而在**图像本身是否承载了这些信息**。要推进只能换采集（更高倍率/更好聚焦）或换标注体系（逐血管人工勾畸形），两者都不是建模工作。
- malformation_ratio 仍是最接近门槛的一个（+0.092~+0.098，CI 下界 0）。它的短板是 n=163、少数类占比 0.429 下 CI 半宽约 ±0.09 —— **加病例，不是加模型**。
