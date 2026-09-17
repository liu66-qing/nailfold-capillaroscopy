# 三组重复 XML 人工语义裁决

请分别打开 `D340_A_B_overlay.png`、`D486_A_B_overlay.png`、`D494_A_B_overlay.png`。每张图左右是同一张原图：红框=A，绿框=B；框内 `序号:类别` 对应 XML object。

重点不是比较框数量，而是判断：

1. 框是否覆盖真实可见血管；
2. `vessel/malformed_vessel/cross_vessel` 类别是否合理；
3. 两边是否各有独特且正确的框；
4. 合并是否会重复框或制造错误标签。

裁决填写在 `artifacts/audits/vascular_dataset_governance_20260830/duplicate_group_decisions.csv` 的 `decision` 列：`A`、`B`、`MERGE` 或 `HOLD`。逐组说明见 `manual_review_guide.csv`。
