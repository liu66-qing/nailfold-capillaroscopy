# 派生可训练辅助资产

这些目录只包含由治理证据筛出的辅助监督候选，不是临床金标准。

- `detection/`：分类原图和由派生 XML 转换的 YOLO 框。
- `weak_field/`：与 Excel image_id 对齐的图片级字段。
- `segmentation/`：机器证实 JSON-image-mask 配对。

所有条目仍需病例级拆分、临床语义认可和独立质量抽查后才能训练。增强图未作为独立样本，locked 重叠未纳入。
