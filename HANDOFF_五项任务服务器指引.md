# 五项任务 · 服务器该看什么

登录：`ssh -p 12956 root@connect.westc.seetacloud.com`
Python：`/root/miniconda3/bin/python`（不要用系统 python）

**SSH 坑（必读）**：内联 heredoc 写 Python 会被 shell 吞掉反斜杠和引号。
一律先写本地文件 → `scp -P 12956 x.py root@connect.westc.seetacloud.com:/tmp/` → 再跑。
脚本里要字面反斜杠时用 `chr(92)`。

**红线**：`evaluation_role=='locked_test'` 的 47 例一律不许读。
它是最后一次评估的唯一凭据，看一眼就废了。

---

## 任务 1 · 核对 periloop 异常病例的原始报告

### 结论已部分确定，别重复劳动

`total_score` = morphology + flow + periloop 三者之和，
**99.4% 的病例误差 < 0.001**（已验证，不必再查）。

规则重算 vs 报告分数，按分组拆开：

| 分组 | MAE | 偏差>1.0 的例数 |
|---|---|---|
| morphology_score | 0.166 | 5 |
| flow_score | 0.082 | 0 |
| **periloop_score** | **0.492** | **51** |

**噪声集中在 periloop 一个分组**，另两组基本自洽。
periloop 组内留一法：去掉 `exudation` 后 MAE 0.492→1.174，它是主贡献字段。

### 已查实的一个机制（只解释 5 例）

RTF 结论文本写着"管袢周围有明显的血浆成分渗出"，
但 manifest 里 `exudation` 抽成了 **"无"**。

- 这类矛盾共 **5 例**，平均 gap **−2.775**，其余 181 例 |gap| 仅 0.441（差 6 倍）
- 但 51 例大 gap 中这类只占 **8%（4 例）**
- **其余 47 例原因未知，这是任务 1 的真正目标**

### 具体该看什么

```bash
# 交叉核对表已生成，先看它
/root/nailfold/artifacts/experiments/rtf_vs_field.csv
# 列：case, rtf_has_exudation, rtf_strong, exu_val, exu_score, periloop_rep, gap
```

原始报告在 `/root/nailfold/data/<archive>/<num>/`：
- `rep_rch1.rtf` / `prn_rch1.rtf` —— **只有结论文本，没有字段值和分数**
- `rep.jpg.jpg`、`rep_<YYMMDD><idx>.jpg.jpg` —— **分数在这张扫描图里，必须人眼看**
- `CAPorg*.jpg` —— 显微图像，与核对无关

RTF 解码脚本已写好：`/tmp/dump_rtf.py`（源 `scripts/dump_rtf.py`），
用法 `python /tmp/dump_rtf.py recovered_archive1/45 recovered_archive2/189`。
注意 RTF 是 GBK 编码 + `\'XX` 十六进制转义，直接 cat 是乱码。

**优先核对这批**（报告 periloop 分数高但五个字段全接近 0，矛盾最尖锐）：

| case | 报告 periloop | 字段合计 | gap |
|---|---|---|---|
| recovered_archive1/45 | 3.60 | 0.00 | −3.60 |
| recovered_archive2/189 | 4.00 | 0.40 | −3.60 |
| recovered_archive3/257 | 3.40 | 0.10 | −3.30 |
| recovered_archive2/130 | 4.10 | 1.40 | −2.70 |
| recovered_archive1/46 | 5.30 | 2.80 | −2.50 |

**核对时要回答的问题**（按顺序）：
1. 打开 `rep*.jpg.jpg`，报告上 periloop 那一栏实际列了几个字段？
   是不是有第 6 个字段没进 manifest？（`groups` 里只有 5 个：
   exudation, hemorrhage, subpapillary_venous_plexus, papilla, sweat_duct）
2. 报告上写的字段值和 manifest 里的值一致吗？
3. periloop 分数是简单加和，还是有加权/取最大值？
4. 那 47 例非渗出矛盾的病例，共同点是什么？

### 判据

修完后重算，`periloop_score` 的 MAE 应降到 morphology/flow 的量级（**< 0.2**）。
降不到就说明机制没找对，别急着改数据。

---

## 任务 2 · 固化并审计评分规则与缺失值处理

### 已发现的三个缺陷

**缺陷 A：规则是反推的，不是原始规则。**
`score_rules_v3.json` 自述：
```json
"provenance": "Thresholds reconstructed from the ordered score/assessment distribution;
               boundary-noise cases remain in the OCR labels."
```
`afferent_diameter` 的 `training_mae` = **3.99e-17** —— 在自己的训练数据上完美拟合，
这是过拟合反推的典型迹象。**这些阈值没有临床依据，必须拿原始报告核对。**

**缺陷 B：两个字段完全没有规则。**
```bash
/root/miniconda3/bin/python -c "
import json; r=json.load(open('/root/nailfold/artifacts/labels/score_rules_v3.json',encoding='utf-8'))
print(r['field_rules']['output_input_ratio'])   # {'type':'missing','cases':0}
"
```
- `output_input_ratio`：有规则条目但 `type='missing'`，186 例全部无法评分
- `flow_speed_um_s`：**置信度全部 0.000，字段全空**

**缺陷 C（我踩过的坑，务必避免）：NaN 会静默拿到最高分。**
`numeric_tree` 用 `x <= threshold` 遍历。NaN 参与比较永远返回 False，
于是一路走 `right` 分支拿到最高分值。
`float('nan')` 不会报错，`str(nan)=='nan'` 也能通过字符串检查。

后果：修正前测得标签噪声 MAE **0.914**，修正后 **0.612**。
我曾据此错误地宣称"total_score 不可信"，实为脚本 bug。

**正确写法**（参考 `scripts/probe_noise_fixed.py`）：
```python
import math
def score_of(f, v):
    r = FR.get(f)
    if r is None or r['type'] == 'missing': return None
    s = str(v).strip()
    if s in ('', 'nan', 'NaN', 'None', '<NA>'): return None   # 必须
    if r['type'] == 'categorical_lookup': return r['mapping'].get(s)
    if 'tree' not in r: return None                            # 必须
    try: x = float(s)
    except ValueError: return None
    if math.isnan(x): return None                              # 必须
    node = r['tree']
    while 'threshold' in node:
        node = node['left'] if x <= node['threshold'] else node['right']
    return node.get('value')
```

### 五级切分：含等号（已验证，别改回去）

`assessment_rule`: `ordered_thresholds`, thresholds `[1.0, 2.0, 4.0, 8.0]`,
labels `["正常","大致正常","轻度异常","中度异常","重度异常"]`。

**切分含等号（`<=`）**：dev 一致率 98.92%（2 例冲突）vs 严格 `<` 的 95.14%（9 例）；
locked_test 95.74% vs 89.36%。
用真实 total_score 含等号切分复现 `overall_assessment`：**183/183 = 100%**。

**含等号只用于把预测分数转等级。真实等级一律直接用 `overall_assessment` 列，
不要从 total_score 反推。**

已知不可约冲突（标记为异常，不要试图修）：
dev `archive2/180`(ts=7.8)、`archive3/263`(ts=7.3)；
test `archive2/182`(ts=3.4)、`archive3/244`(ts=4.1)。

### 每字段有独立的抽取质量列

`<field>__confidence` 和 `<field>__status`（共 102 列）。

**重要发现：OCR 抽取质量不是瓶颈。**
前 7 个高方差字段置信度全在 0.93–0.97，`high_consensus` 占 174–181/186。
低置信度反而集中在低方差字段（sweat_duct 0.832、malformation_ratio 0.847）。
`engine_conflict` 全库仅 11 例。
`corr(方差%, 平均置信度) = 0.258`。

**所以别把精力放在"提高 OCR 置信度"上，那不是噪声来源。**

### 物理越界值（另一类问题，量小）

甲襞毛细血管管径正常 8–20 μm、管袢长 150–400 μm。实测越界：
`afferent_diameter` 最大 **325**（正常值 16 倍）、`apex_diameter` 113、
`efferent_diameter` 89、`loop_length` 648。共 19 例含至少一个越界值。

影响有限：越界组重算 MAE 1.189 vs 正常组 0.882。
这是报告上写错（单位混淆/小数点错位），OCR 忠实抄录，不是抽取错误。

### 交付物

把审计过的规则固化成 `score_rules_v4.json`，必须包含：
- 每条阈值的出处（原始报告依据 or 反推，标明）
- 显式的缺失值语义（缺失 → 不计分，不是 0 分，不是最高分）
- `output_input_ratio` 的真实规则，或明确声明弃用
- 一个回归测试：对 dev 186 例重算，断言分组 MAE 均 < 0.2

---

## 任务 3 · 回归单任务 / 梯度平衡的最小消融

### 这件事已经做完了，结论是阴性，不要重跑

**EXP-C，三臂完整，3 seeds × 5 folds，n=183：**

| arm | 配置 | MAE | ρ | pred_sd | sd_ratio |
|---|---|---|---|---|---|
| A | 纯回归, ts/20 | 2.796 | 0.377 | 1.896 | 0.523 |
| B | 纯回归, z-norm | **2.737** | 0.357 | 1.630 | 0.449 |
| C | 多任务, 损失平衡 | 2.775 | 0.343 | 1.625 | 0.448 |
| — | 常数中位数基线 | 2.940 | — | — | — |
| — | EXP-10 原版 | 2.856 | 0.354 | — | 0.585 |

真实 sd = 3.609。

**三臂跨度仅 0.059。** 去掉分类头、平衡损失量级、z-score 归一化，
加起来只值 0.06 MAE，且 sd_ratio 全都没超过 EXP-10 原版的 0.585。
**损失量级不平衡不是瓶颈，这条路已封闭。**

产物：
```
/root/nailfold/artifacts/experiments/exp_c_v2_g0/{A_reg_only_raw,B_reg_only_znorm}_oof.csv
/root/nailfold/artifacts/experiments/exp_c_v2_g1/C_multi_balanced_oof.csv
# 列：case, true_ts, pred_ts, true_level, pred_level_from_score
日志：/root/nailfold/logs/exp_c_v2_g0.log, exp_c_v2_g1.log
脚本：/tmp/exp_c_v2_g0.py（源 scripts/exp_c_loss_fix.py）
```

### 顺带已封闭的其他路径

| 方向 | 结果 | 结论 |
|---|---|---|
| 渐进解冻 K=2, lr 1e-5 | MAE 3.993, ρ 0.025 | n=186 灾难性过拟合 |
| 渐进解冻 K=2, lr 5e-6 | MAE 4.128, ρ 0.017 | 同上，**保持全程冻结** |
| 视频补 flow_state（EXP-B 自对照）| 静态 BA 0.192 / 视频 BA 0.204 | **均低于随机 0.250** |
| 64维 bottleneck + 跨度加权损失 | +0.45 MAE | 更差 |
| 12 头共享 trunk 多任务 | 相对随机仅 +0.08~0.21 | 到顶 |

EXP-B 归档说明：`E:\甲劈微循环\scripts\EXP_B_ARCHIVED.md`
产物 `/root/nailfold/artifacts/experiments/exp_b_video_flow/`（ST_both 臂已终止）。

### 真正该做的消融（尚未做）

**按方差排序做单字段专项模型**，而不是继续调 total_score 的损失。

修正 NaN bug 后的正确优先级：

| 字段 | 方差% | 累计 | 众数占比 | 可观测 |
|---|---|---|---|---|
| microthrombus | **42.64** | 43% | 0.59 | 是 |
| exudation | 13.56 | 56% | 0.51 | 是 |
| rbc_aggregation | 8.11 | 64% | 0.43 | **否** |
| capillary_count | 7.15 | 71% | 0.69 | 是 |
| flow_state | 6.58 | 78% | 0.34 | **否**（EXP-B 已证否）|
| loop_length | 6.28 | 84% | 0.47 | 连续值, r_total≈0.05 可疑 |
| papilla | 5.09 | 89% | 0.42 | 是 |

**注意：按"跨度"排序是错的。** 我曾把 capillary_count（跨度 6.0）当第一优先级，
但它方差只占 7.2%；真正的第一是 microthrombus，占 **42.6%**。

**8 个字段可直接固定成众数**（方差贡献合计 < 2%，零风险）：
sweat_duct 0.00%（众数 0.99）、vasomotion 0.00%（0.98）、wbc_count 0.01%（0.98）、
blood_color 0.36%、clarity 0.53%、hemorrhage 0.55%（0.93）、crossing_ratio 0.68%，
外加 `flow_speed_um_s` 全空。

### 为什么值得做（关键论据）

天花板（top-k 字段完美 + 其余取众数）：

| 场景 | MAE | 精确等级 | ±1级 | **中度+ BA** | 重症灵敏度 | QWK |
|---|---|---|---|---|---|---|
| 当前 arm B | 2.737 | 42.6% | 86.3% | **0.511** | 0.310 | 0.157 |
| microthrombus+exudation 做对 | 2.380 | 48.6% | 91.3% | **0.800** | 0.224 | 0.498 |
| 前 5 字段做对 | 1.638 | 63.4% | 95.6% | **0.896** | 0.517 | 0.757 |
| 21 字段全对 | 0.642 | 82.0% | 97.8% | 0.883 | 0.862 | 0.875 |

**只做对两个字段，中度+平衡准确率从 0.511 跳到 0.800。**
而 MAE 几乎没动（2.737 → 2.380）—— **MAE 把这个最大的机会完全藏住了。**

标签自洽地板：21 字段全对时 MAE 仍有 **0.642**（不是 0），
与独立算出的 `/root/nailfold/artifacts/experiments/exp_e_ceiling/ceiling.json`
中 `S2_all_oracle` = 0.6418 一致。

---

## 任务 4 · 明确产品人群与验收指标

### 这是当前最大的阻塞项，且不需要服务器

以下四个问题至今无答案，而它们决定"MAE 2.737 算不算够"：
1. 产品定位：A. total_score 连续分 / B. 五级标签 / C. 健康建议 / D. A→B→C 全链路
2. "90% 准确率"是硬性验收，还是口头目标？在哪个测试集上算？允不允许差一级？
3. 哪类误判最不能接受？（重症漏检 vs 正常人误报，代价完全不同）
4. 面向已就诊异常人群，还是普通健康筛查？

### 服务器上已经算出的、能支撑这个决策的数据

```
/root/nailfold/artifacts/experiments/exp_h_clinical_metrics/{summary.json,report.txt}
```

**必须先看的一个数字：**

> **±1 级准确率 = 86.3%（arm B）。常数中位数基线 = 86.3%。一位不差。**

那个基线就是"对所有 183 例都预测中度异常"。
精确等级准确率 42.6% vs 常数 41.5%，只赢 1.1 个点。

**所有能到 70% 以上的指标，都能被一个不看图的常数预测器复刻。**

各等级召回率（arm B）：

| 真实等级 | n | 召回 |
|---|---|---|
| 正常 | 9 | **0.000** |
| 大致正常 | 16 | **0.000** |
| 轻度异常 | 24 | 0.042 |
| 中度异常 | 76 | 0.776 |
| 重度异常 | 58 | 0.310 |

183 例中 140 例被预测为"中度异常"。**25 个健康/接近健康的病例，一个都没认出来。**

筛查切点（arm B，原阈值）：

| 切点 | 患病率 | 灵敏度 | 特异度 | PPV | BA |
|---|---|---|---|---|---|
| 任何异常 | 0.863 | **1.000** | **0.000** | 0.863 | 0.500 |
| 中度及以上 | 0.732 | 0.940 | 0.082 | 0.737 | 0.511 |
| 重度 | 0.317 | 0.310 | 0.896 | 0.581 | 0.603 |

灵敏度 1.000 不是本事 —— 特异度 0.000，它从不说任何人正常。
重症漏检 0/58 同理，是"从不说正常"的免费副产品，常数基线也是 0。

### 建议的主指标

**中度及以上切点的平衡准确率**，且必须与常数基线并列在同一张表里。
arm B = 0.511，基线 = 0.500。这个指标不受患病率影响。

**次要指标：C-index = 0.622**（基线 0.500）—— 唯一有真实信号的地方，
因为它与阈值无关。QWK 0.157 / LWK 0.114（基线均 0.000）。

阈值重校准试过了（按患病率匹配，in-sample 偏乐观）：
重症召回 0.310→0.500、QWK 0.157→0.280，
但精确准确率 42.6%→35.5%、±1 级 86.3%→80.3%、重症漏检 0%→5.2%。
**重校准只是搬运误差。** C-index 和 Spearman 与阈值无关，
始终 0.622/0.357 —— 排序能力才是硬约束。

### 一个结构性限制（必须让决策者知道）

**本队列异常率 86.3%**（正常 9 + 大致正常 16 = 25/183）。

**在这个队列上做不出任何筛查声明，无论换什么模型** —— 里面几乎没有健康人，
灵敏度和 PPV 天然虚高。

- 若面向健康筛查：**必须补健康人数据**，这是数据问题不是模型问题
- 若面向已就诊人群：该报的不是"筛查准确率"，而是分级一致性（QWK/C-index）

### 无法确定的事（别声称）

**没有任何重复标注**，所以 QWK 的人类可复现上限**测不出来**。
QWK 0.157 只能对着 0.000 的基线读，**不能说"接近人类水平"**。

要建立这个上限：抽 30–50 例，两人独立标，算 QWK。这是当前最大的认知盲点。

---

## 任务 5 · 重建未参与选型的测试集

### 先认清现状：患者身份不可恢复

| 线索 | 状态 |
|---|---|
| `patient_id` 列 | **完全为空** |
| 姓名 | 无 |
| 文件 mtime | 全是归档时间，无信息 |
| `duplicate_group` 列 | **已损坏，别用** |
| 检查日期 | **可恢复** |

`duplicate_group` 的损坏程度：213 个单例 + 一个 20 成员的组，
该组跨越 ts 0.6→13.2、五个等级全覆盖，且全部落在 dev fold 3。这是错的。

**日期可从文件名恢复**：`rep_<YYMMDD><idx>.jpg.jpg`
→ 105 个日期，覆盖 240 例；4 例内含两个不同日期。

泄漏风险实测：
- **26 个日期横跨 dev 和 locked_test**
- 42 个日期横跨不同 fold

**结论：患者级重复既不能确认也不能排除。** 别声称"患者级独立"。

### 已做的敏感性分析：没发现泄漏导致的虚高

EXP-10 case 级 MAE 2.856 / QWK 0.133
→ 按 `date#laterality` 分组后（n=100）MAE **2.636** / QWK **0.197**

**指标反而变好了。** 所以现有 OOF 的问题不是泄漏，是选型污染。

脚本：`scripts/exp_d_grouped_eval.py`，里面的 `metric_suite()` 是
项目的标准指标函数（含 MAE、macro-MAE、Spearman、C-index、QWK/LWK、BA、
±1、重症低估、各级召回，且每次自动附常数基线），**新实验直接复用它**。

### 现有 CV 划分是干净的（不必怀疑）

`/tmp/exp10_direct.py`（源 `scripts/exp10_direct_server.py`）中：
```python
valfold  = (fold + 1) % 5
trainids = dev.loc[~dev.development_fold.isin([fold, valfold]), 'exam_case_id'].tolist()
valids   = dev.loc[dev.development_fold.eq(valfold), 'exam_case_id'].tolist()
testids  = dev.loc[dev.development_fold.eq(fold), 'exam_case_id'].tolist()
```
case 级隔离正确，**没有 test 偷看**。

**但这批 OOF 已被反复用于选型（至少 10 轮实验），不能当最终泛化结果。**

### 该做什么

1. **locked_test 的 47 例继续封存**，只用一次，且必须在规则修好、
   主指标定下来之后。现在花掉就没有第二次机会。
2. 用日期做分组 CV，作为**敏感性分析**，不能声称患者独立。
   建组键：`日期#手别`。
3. 若能拿到原始设备导出记录或就诊台账，去补 `patient_id`。
   这是唯一能真正解决患者级独立的办法。
4. 新的选型一律在 dev 的嵌套 CV 内做，**外层保持未被触碰**。

### 关键文件清单

```
manifest       /root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv
               读时务必 dtype={'exam_case_id': str}，102 列，233 行（186 dev + 47 test）
帧索引         /root/nailfold/artifacts/features/image_index.csv （仅 2 列）
评分规则       /root/nailfold/artifacts/labels/score_rules_v3.json
原始数据       /root/nailfold/data/recovered_archive{1,2,3}/<num>/
backbone       /root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth
实验产物       /root/nailfold/artifacts/experiments/
日志           /root/nailfold/logs/
```

帧统计：2207 帧 / 248 例，每例 min 1 / max 23 / mean 8.90 / median 8。
**标签是 case 级，被广播到每一帧** —— 对 capillary_count 这类计数字段
几乎肯定是错的假设，尚未验证。

**只有图像进入模型。** `image_index.csv` 只有 2 列；
244–247 段视频从未被主模型使用。

### 跑长任务必须用 nohup

裸 `ssh` 会在连接重置时把进程带走。
```bash
nohup /root/miniconda3/bin/python /tmp/xxx.py --gpu 0 > /root/nailfold/logs/xxx.log 2>&1 &
```
**每个 epoch 都要 `print(..., flush=True)`** —— 曾有实验静默 40 分钟无法判断死活。

### 机器是共用的

`/root/autodl-tmp/medprm/` 下有另一个项目的作业在跑。
曾出现负载 54、显存被占 14.5/40 GB，导致每 fold 从 700s 涨到 1200s。
排长任务前先 `nvidia-smi` 和 `cat /proc/loadavg`。

---

## 一句话优先级

**任务 4（定人群和验收指标）> 任务 1+2（修 periloop 规则）> 任务 3 的单字段模型 > 任务 5。**

任务 3 的损失消融和任务 5 的泄漏排查**都已做完且都是阴性**，
不要重做。任务 4 不做，其余四项的结果都无法解释。
