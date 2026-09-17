# 管袢数自动错误归因

输入为远端已有开发 OOF `step7_count_gbt.cases.csv`、开发 `seg_features.parquet`、`per_video_audit.csv`、`per_case_measurements.csv`；按角色再次过滤后 181 例进入计数 OOF，locked 47 例仅用于边界校验，`locked_cases_seen=0`。未使用 GPU，未修改 v1。

事实结果：121/181 正确，60/181 错误，其中 overcount 36、undercount 24。阈值计数从 `t=0.05` 到 `t=0.50` 的病例均值跨度中位数 25.76，相对 `t=0.225` 中位数 2.92；高跨度尾部阈值偏移候选 39 例。帧计数变异系数中位数 0.547，帧间不稳定候选 53 例。漏检候选 14 例（其中 14/24 undercount），碎片/FP 候选 15 例（其中 15/36 overcount），粘连候选 0 例。

已证据支持：阈值敏感性和帧间方差是可量化因素；漏检候选与 undercount、碎片/FP 候选与 overcount 方向一致，可用于人工复核排序。

尚不能判断：是否粘连主导、漏检主导或某设备主导。当前输入没有实例级真值 mask、真实连通域、骨架分支数和设备 ID；`mask_count_proxy`/`connected_component_proxy` 是检测器实例计数，不是医学管袢数，骨架分支字段明确为 unavailable。

逐病例字段、规则旗标、分布和错误方向混淆表见 `diagnostics_count_attribution/report.json` 与 `per_case_attribution.csv`。
