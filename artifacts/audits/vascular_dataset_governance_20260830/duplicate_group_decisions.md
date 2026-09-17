# 重复组决策入口

逐组决策表：`duplicate_group_decisions.csv`。

当前 AI 默认决策为 `HOLD`，因为每组图像 MD5 相同但 XML 框/类别签名不一致。请在 `decision` 列填写 `A`、`B`、`MERGE` 或保持 `HOLD`，并补充 `reviewer`、`review_time`、`decision_reason`。
