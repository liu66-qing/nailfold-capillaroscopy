# 血管数据集只读治理审计（2026-08-29）

机器可读明细：`audit.json`。

## 核心计数
- 分类：扩充前图片 582、XML 582、扩充后图片 11600、YOLO 文件 11600；XML 目标 3835。
- 分类 XML 类别：{'vessel': 1779, 'malformed_vessel': 1508, 'cross_vessel': 547, 'corss_vessel': 1}；YOLO 类别 ID：{2: 29960, 0: 35540, 1: 10860}。
- 分割：图片 3159、mask 3159、LabelMe JSON 3161；形状类别 {'vessel': 3203, '_background_': 980}；shape_type {'polygon': 4183}。
- recovered 图像精确哈希重叠 123 条（感知哈希候选 186 条）。
- 既有清单：233 个病例，locked_test 47，development 186。
