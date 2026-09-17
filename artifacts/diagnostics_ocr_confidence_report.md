# OCR 一致性高置信子集

一致性分组由 `rapidocr_all_clean_v2.jsonl` 与 `qwen_all.jsonl` 的同病例报告副本重建，仅作 OCR 两路一致性代理，不是医生 kappa。开发集 186 例，`locked_cases_seen=0`，GPU 未使用，v1 未修改。

开发 OOF 分组支持：清晰度 agreement 185/missing 1；血色 agreement 170/missing 16；渗出 agreement 175/disagreement 3/missing 8；出血 agreement 180/missing 6；静脉丛 agreement 181/missing 5；乳头 agreement 181/missing 5。不一致组除渗出外没有开发支持，不能做稳定子集结论。

未加权到 agreement 加权（1.5；disagreement 0.5；missing 1.0）的整体 BA 变化：清晰度 0.462→0.452，血色 0.471→0.508，渗出 0.351→0.354，出血 0.500→0.500，静脉丛 0.341→0.357，乳头 0.432→0.361。

已证据支持：血色是唯一达到明显正向变化的权重候选；静脉丛仅小幅正向，其他字段无普遍收益。

尚不能判断：OCR 一致性是否对应医生标签质量；不一致样本太少，不能支持删样本或普遍降权，也不能称为医生一致性评估。

详细分组 BA、类别召回和逐病例预测见 `diagnostics_ocr_confidence/report.json` 与 `oof_group_predictions.csv`。
