# dual_seg_geo 二级 GBT OOF

geometry image features 已按 development 清单重新提取：1708 帧、186 病例、394 维，`locked_cases_seen=0`。

新增路线：`DINOv2 aggregate + HuluMed aggregate + seg_features + geometry aggregate`。

5 折 OOF 交付字段均值（clarity、blood_color、exudation、SVP）：

- `dual_seg_geo`: **0.6759**
- 同脚本对照 `dual_seg`: **0.6863**

当前 geometry 增量没有带来收益，较对照低约 1.05 个百分点。该结果是一次固定分类器协议的诊断，不覆盖既有 step10/v1 基线；papilla 仅作探索性记录。Logistic 出现收敛警告，后续若继续优化需先做训练集内标准化或改用收敛稳定的分类器，再进行独立消融。
