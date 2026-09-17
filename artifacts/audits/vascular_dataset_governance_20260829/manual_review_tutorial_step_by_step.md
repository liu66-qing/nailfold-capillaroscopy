# 血管数据集人工审核保姆级教程（逐步执行版）

本教程把 `data/血管数据集` 的人工审核拆成可执行的闸门。审核期间不覆盖原始图片、XML、YOLO、JSON、mask，也不修改 v1 基线或 locked 集。

## 0. 准备工具和记录表

使用 Excel/LibreOffice Calc 记录审核；使用 CVAT、LabelMe 或能显示 Pascal VOC/YOLO 框的工具看框和多边形；普通图片查看器只能看图，不能用来猜坐标。

打开目录：

`E:\甲劈微循环\artifacts\audits\vascular_dataset_governance_20260829`

先查看 `governance_report.md`、`manual_review_queue.csv`、`governance_manifest.csv`。把 `manual_review_decisions_template.csv` 复制成同目录下的 `manual_review_decisions.csv`，审核结果只写这个副本。

每条记录至少填：`review_id, asset_path, case_id, source_layer, issue_category, reviewer, review_time, decision, reason, corrected_value, evidence_path`。

`decision` 只允许：`PASS`、`PASS_AUXILIARY`、`HOLD`、`EXCLUDE_LOCKED`、`EXCLUDE_DUPLICATE`、`REPAIR_REQUIRED`、`REJECT`。不能只写“看过了”。

## 1. 建立只读工作副本

1. 在 `artifacts/audits/` 新建工作目录，例如 `vascular_dataset_manual_review_20260830`。
2. 只复制清单、模板和需要查看的少量图像；不要在 `data/血管数据集` 内保存修复结果。
3. 在审核记录首行写审核人、开始时间、工具版本，并注明“原始数据只读”。
4. 如果发现原始文件修改时间变化，立即停止审核，先恢复备份。

## 2. 第一闸门：先排除 locked

打开 `governance_manifest.csv`，筛选 `locked_overlap=yes`。应确认：17 个 locked 病例、32 个本地分类原图、640 个增强派生物。

对每个病例记录 `EXCLUDE_LOCKED`：原图和所有 `<original_id>_<augmentation_id>.jpg` 都要列出。不要打开这些图像挑样本，也不要用它们制定阈值或抽样规则。

**通过条件：** locked 原图和全部增强派生物均已排除；少一个就停在本步骤。

## 3. 第二闸门：确认来源和病例归属

对队列中的 449 张未直接映射分类原图，逐张查找：病例号/受试者号、采集日期、设备和倍率、是否同一病例多帧、是否与 186 个 development 或 47 个 locked 重复、使用权限。

证据按强弱排序：原始导出清单或采集记录 > 原始压缩包路径和哈希 > 视觉比对 > 文件名猜测。仅凭文件名不能放行。

- 证据完整且未与 locked 重叠：先记 `HOLD`，待后续标签和抽查通过再改 `PASS`；
- 确认为 development 的重复帧：记录病例号，通常只能 `PASS_AUXILIARY`；
- 无法证明来源：`HOLD` 或 `REJECT`；
- 4 个未入清单的重叠病例：来源确认前不得并入 development。

## 4. 第三闸门：先批准标签定义

单独写一页操作性定义，并让标注负责人确认。必须明确：

- `cross_vessel`：真实相交还是投影重叠；
- `malformed_vessel`：弯曲、扩张、分叉、断裂的归类；
- 出血：阳性最小证据，反光/阴影排除规则；
- SVP：不可见、1 排、2 排、扩张的边界；
- 乳头：正常、异常、不可见边界；
- 汗腺导管：观察窗口、模糊图和不可判定处理。

每个字段准备“明确阳性、明确阴性、边界”三类示例。边界样本可以 `HOLD`，不要强行二分类。同学 Excel/XML 和模型预测都只是候选证据，不能直接称为金标准。

## 5. 第四闸门：审核分类框

数据位置：

- 图片：`data/血管数据集/分类数据集/扩充之前/images`；
- XML：`data/血管数据集/分类数据集/扩充之前/annotations`；
- YOLO：`data/血管数据集/分类数据集/labelYOLOs`。

先处理 9 个异常 XML 框和 7 行异常 YOLO。用标注工具叠加显示图片和框，检查：坐标是否在图内、左上角是否小于右下角、框内是否真是血管、类别是否符合定义、附近是否漏标。

- 坐标和目标都能明确判断：记录 `REPAIR_REQUIRED`，新坐标写 `corrected_value`；
- 目标不存在：`REJECT`；
- 类别依赖未批准定义：`HOLD`。

修复文件只能写到 `artifacts/derived/vascular_dataset_reviewed_v1/` 等新目录，不覆盖原文件。

对 485 个 XML 尺寸不一致样本：必须同时确认 XML 声明尺寸和实际图片确为同一图像的等比例缩放，且所有框可使用同一比例变换；否则 `HOLD`。对 46 个 filename 无法配对 XML：用哈希、导出清单或视觉证据重建，不能按数字猜测。

## 6. 第五闸门：审核分割三元组

数据位置：`data/血管数据集/分割数据集/{images,mask,annotations}`。

当前 3,161 个 JSON 的 `imagePath` 都不能在本机直接解析；1,557 个样本存在尺寸/索引漂移，109 个 JSON 无 `imageData`。所以不能直接按同名文件训练。

对每个候选三元组同时确认：JSON 内容或来源对应本地 JPG；JPG、mask 尺寸完全一致；polygon 是血管前景；断裂/遮挡按定义处理；二值 mask 没有系统性漏标或填入背景。

把结果写入独立配对表，至少包含：`image_path, json_path, mask_path, pair_evidence, dimension_check, semantic_check, reviewer, decision`。`_background_` 多边形和二值 mask 的语义必须通过图片抽查确认，不能只看类别名称。

## 7. 第六闸门：代表性抽样

至少审核 100 张分类原图和 100 对分割 JPG/mask/JSON。分类抽样覆盖所有标签类别、不同尺寸、不同来源、异常文件和重复组；分割抽样覆盖有/无 `imageData`、尺寸一致/漂移、polygon 数量高低。

每张记录漏标数、错标数、不可判定、配对错误、图像质量和耗时。两名审核者有分歧时，保留双方意见和最终裁决。

建议门槛：关键错误率 >5% 继续 `HOLD`；1%–5% 最多 `PASS_AUXILIARY`；<1% 且无系统性错误才可申请 `PASS`。这是数据质量门槛，不是模型准确率。

## 8. 第七闸门：物理标定

在任何长度、直径、密度或“微米”指标进入训练/报告前，确认设备型号、倍率、像素到微米关系、裁剪/缩放历史以及不同来源是否同一标定。无法确认时只能报告像素或相对量，不能写成微米级临床测量。

## 9. 每批审核后的继续/回退规则

每批结束检查：是否出现新的 locked 重叠；是否有病例归属、标签语义或配对冲突；抽样错误率是否超门槛；修复是否只写在派生目录。

任一问题出现，该批次全部改回 `HOLD`，回到对应步骤。不得继续训练，不得修改 v1 或既有评测结果。

## 10. 最终交付物

审核结束必须有：`manual_review_decisions.csv`、标签定义文档、派生修复文件或配对表、抽样统计和分歧记录、新候选训练 manifest，以及一页 locked 排除/增强分组/物理标定声明。

在所有 P0 项关闭前，数据集状态保持 `HOLD`。完成后也只能把它称为“经审核的血管检测/二值分割候选监督”，不能直接称为临床金标准或弱字段最终标签。
