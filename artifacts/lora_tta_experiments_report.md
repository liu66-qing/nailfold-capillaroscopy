# LoRA 与 TTA 受控实验报告

## 基线

Git 回退点：`current-dual-seg-binary-v1`，提交 `f9b50a8`，标签同名。固定 `dual_seg` ExtraTrees 二级 OOF，4 个交付字段均值 BA **68.48%**。

## TTA

使用原图、水平翻转、中心 90% 裁剪；不做色彩扰动。训练仅使用原始 DINOv2 特征，增强视图只在 test 推理时平均概率。

- baseline：68.48%
- TTA：68.63%
- 变化：+0.15 个百分点
- clarity +0.57pp，blood_color +0.54pp，exudation +1.03pp，SVP -1.53pp
- 判定：`rejected`，未达到 +1.0pp 且未满足稳定性门槛

## DINOv2 LoRA pilot

rank=4；冻结原始 DINOv2，仅对最后 4 个 block 的 qkv/proj 添加低秩增量；五个二级头使用病例标签进行帧级监督。每折仅用 train，validation 选择 epoch，test 只评估。

- LoRA 交付字段均值 BA：66.60%
- 相对基线：-2.88 个百分点
- 判定：`rejected`

该 pilot 不能证明 LoRA 的理论方法无效，只能证明当前 186 例病例级弱标签、帧级监督和 5 折协议下没有稳定收益。不能把结果外推到 locked 集。

## 最终回退

TTA 和 LoRA 均不替换 baseline。保留所有实验脚本和结果用于审计；正式路径继续使用 `dual_seg` 二级 GBT。后续不扩大 LoRA/TTA 搜索空间，除非获得新的标签或独立验证证据。
