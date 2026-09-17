# 甲襞分类评估与失败模式审计

日期：2026-09-08

## 范围

本审计只使用 development 数据、既有 OOF 预测和模型 checkpoint。未读取或使用 locked test 预测进行选模、阈值或聚合调节。

## 1. 数据划分与标签

- 清单包含 233 个 case：186 development、47 locked test；图像索引共 2207 帧。
- `patient_id` 233 行全部为空，因此无法证明患者级隔离。当前只能证明 `exam_case_id` 和 `duplicate_group` 隔离。
- 214 个 duplicate groups；没有 duplicate group 跨 development/locked，也没有跨 development folds。
- 一个 duplicate group 包含 20 个 case，均被放在 development fold 3，说明已对已知重复组进行隔离。
- 三个 archive 都同时分布在 development 和 locked test，archive 不是隔离单位。
- reviewed 标签相对 pre-dirty-fix 版本的五字段差异数：clarity 33、blood_color 57、exudation 30、SVP 32、papilla 36。这些是版本间差异数，不等同于人工脚本记录的 173 次 corrections，因为还包含 dirty-fix 和空值处理。
- `apply_review.py` 证明人工复核只覆盖 development；其记录为 173 corrections、2 fills、8 blanks。locked labels 未被该人工复核流程修改。

## 2. 选模规则

- LoRA baseline v2：单 seed；case 标签复制到每帧，以 frame CE 训练；每个外测 fold 使用相邻 fold 验证；最佳 epoch 按前四字段平均 BA 选择，明确排除 papilla；最后又在四个 LoRA 配置中按字段事后挑选最佳配置。
- Progressive complexity：5 seeds；0.75 case CE + 0.25 frame CE；所有五字段参与最佳 epoch 选择；固定配置比较。
- Patch pooling：与 progressive complexity 使用相同 folds、5 seeds、loss 和选模规则，因此可以与 A_frozen 公平比较。
- LoRA v2 的 0.7729 是逐字段事后路由的五字段平均，不是一个预先固定配置的无偏性能估计。各 metrics 文件的 `delivery_mean_ba` 实际排除 papilla；baseline JSON 中的 0.7729 则包含 papilla，命名不一致。

## 3. 当前 patch pooling 结果

实验已全部完成。相对同流程 A_frozen（mean BA 0.7356）：

| 配置 | Mean BA | 结论 |
|---|---:|---|
| A CLS only | 0.7356 | control |
| B CLS + patch attention | 0.7486 | +0.0130；当前同流程最佳 |
| C patch only | 0.7173 | 下降 |
| D TTA | 0.7270 | 下降 |
| E multicrop | 0.7345 | 基本持平 |

CLS + patch attention 相对 frozen：clarity +0.0165、blood_color -0.0370、exudation +0.0435、SVP +0.0221、papilla +0.0203。它有研究意义，但不能直接与 LoRA v2 的 0.7729 比，因为训练和选择规则不同。

## 4. 同流程 LoRA 与 frozen 对照

在相同 folds、5 seeds、case/frame loss、all-field epoch objective 下：

- LoRA r4 相对 frozen：五个字段全部下降（约 -0.007 至 -0.049 BA）。
- LoRA r8 相对 frozen：仅 exudation +0.0102；clarity -0.0316、blood_color -0.0367、SVP -0.0221、papilla -0.0035。

因此，现有证据支持“在公平 protocol 下，LoRA 没有稳定优于 frozen”。它不支持把 LoRA v2 的逐字段 0.7729 当作可重复的公平优势。

## 5. 排序、阈值与帧分歧

Frozen 的 AUC：clarity 0.885、blood_color 0.826、exudation 0.833、SVP 0.863、papilla 0.667。前四字段有较强排序信号；papilla 明显较弱。

仅在内部验证 fold 选择阈值后，frozen BA 变化为：clarity 0.7892 -> 0.7930、blood_color 0.7577 -> 0.7554、exudation 0.7324 -> 0.6894、SVP 0.7940 -> 0.7956、papilla 0.6044 -> 0.6010。阈值不是主要瓶颈；小验证折选出的阈值不稳定。

LoRA v2 逐帧重算复现了所有基线 BA。不同帧跨越 0.5 的病例比例：clarity 57.3%、blood_color 74.0%、exudation 57.9%、SVP 66.8%、papilla 43.2%。病例内证据差异很大。

但在内部验证 fold 选择 mean-logits、mean-probability、median、max、top-3 后，交叉选择的 BA 均未超过原 mean-logits：clarity 0.8038 vs 0.8371、blood_color 0.7330 vs 0.7392、exudation 0.7939 vs 0.7993、SVP 0.8559 vs 0.8622、papilla 0.5708 vs 0.6268。简单更换聚合规则没有解决问题。

## 结论

最严重的未决风险是患者级隔离无法核实，因为 `patient_id` 全空。其次是 LoRA v2 的性能估计受到单 seed、papilla 排除于 epoch objective、按字段事后挑配置的选择偏差影响。病例内帧分歧是真实存在的，但现有 max/top-k/median 方案不能稳定利用它。

当前可保留的研究结果是 CLS + patch attention 相对严格 frozen control 提升 0.013 mean BA，尤其改善 exudation 和 SVP；它需要预先固定后再做独立验证，不能继续用 locked test 调整。
