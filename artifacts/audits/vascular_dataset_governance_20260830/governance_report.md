# 血管数据集续治理结果（2026-08-30）

## 最终状态

本轮治理使用既有只读审计作为输入，未使用 GPU，未修改 `data/` 原始文件、既有 v1 基线、模型、标签或评测结果，也未用 locked 集进行训练、调参、阈值或模型选择。当前无条件可直接训练资产仍为 **0**。

## 已自动确认

- locked 排除清单扩展为 1344 个文件：32 张 locked 原图、32 个对应 XML、640 张增强图和 640 个增强 YOLO；所有状态为 `EXCLUDE_LOCKED_OVERLAP`。
- 分类原图按 MD5 建立 572 个哈希组，其中真正的完全重复组为 10 组；10 组重复图像的 XML 签名均不一致，已回退到 `HOLD_DUPLICATE_LABEL_CONFLICT`，canonical 仅作索引规则。增强按 original_id 建组，582 组中 580 组为 20 张，禁止独立拆分。
- 582 个 XML 均按 basename 原图证据生成派生元数据修复副本并通过尺寸/框边界校验（其中 5 个有用户视觉确认），原 XML 哈希保留；7 个零面积 YOLO 行已在派生副本隔离并复验剩余行格式。
- Excel 含 582 行、582 个唯一 image_id，5 个弱字段列全部非空且取值集合与标注说明 PDF 一致；这证明结构一致，不替代临床语义认可。
- 分割 JSON 中 2332 个达到配对证据门槛，其中 84 个来自连续缺失批次的“-2 序列偏移 + 尺寸 + mask 几何”重建，其余来自嵌入图像像素证据；这只是配对确认，不等于临床标签或训练放行。

## 仍需人工审核

- 829 个分割 JSON 配对仍为 `HOLD_PAIRING`（包括无法通过几何证据的无 `imageData` 记录）；不得猜测图像或 mask。
- 10 个重复组的冲突 XML 标签、`vessel/malformed_vessel/cross_vessel` 的临床语义认可、来源/病例/权限和物理标定仍需人工决定；XML filename/size 元数据本身已完成派生修复。
- 弱字段列的 PDF 规则与 Excel 取值已完成结构一致性检查，但“正常/异常/不可见”等临床操作定义仍需具备资质的人员最终认可。
- 机器配对通过的分割样本仍需代表性视觉抽查、mask 完整性检查、独立复核一致性和病例级 split 审核。

## 用途判定

- 血管检测预训练：仅可在排除 locked、完成来源/语义/异常框审核、canonical 去重和病例级 split 后作为 `PASS_AUXILIARY`；当前仍 HOLD。
- 二值血管分割预训练：仅对 `AUTO_CONFIRMED_PAIR` 子集，在人工 mask 抽查和 split 审核通过后作为 `PASS_AUXILIARY`；当前仍 HOLD。
- 管袢数、交叉率、畸形率改善：只能作为候选辅助监督/特征来源，需统一定义、病例级聚合和独立验证；不可直接作为临床金标准。
- 弱字段分类：Excel 对 SVP、红细胞聚集、乳头、汗腺导管和出血提供 582 行图片级辅助标签，结构审计已通过；完成临床语义、来源和 locked 排除后可作 `PASS_AUXILIARY`，不能当最终金标准。对 clarity、blood_color、exudation 没有直接标签。
- 最终临床金标准：不可用。外部/同学标注不等同临床金标准。

## 机器可读输出

见 `governance_summary.json`、`final_status.json`、`manual_review_queue.csv`、`manual_review_consistency.json`、`weak_field_label_audit.json`、`source_case_mapping.csv`、`locked_exclusion_manifest.csv`、`classification_duplicate_groups.csv`、`duplicate_annotation_consistency.csv`、`augmentation_group_manifest.csv`、`xml_anomalies.csv`、`yolo_anomalies.csv`、`segmentation_pair_reconstruction.csv`、`candidate_training_manifest.csv`；派生 XML、YOLO 隔离副本和哈希审计见 `artifacts/derived/vascular_dataset_governance_20260830/`。
