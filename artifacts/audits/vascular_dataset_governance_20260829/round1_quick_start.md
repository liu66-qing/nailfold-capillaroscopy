# 第 1 轮审核：只做三件事

## A. 确认 locked 已排除

打开 `locked_exclusion_manifest.csv`。

- 总行数应为 **672**；
- `classification_pre` 应为 **32** 行；
- `classification_aug` 应为 **640** 行；
- `linked_recovered_cases` 去重后应为 **17** 个病例。

你不需要逐张看图。只需确认：同一个病例的原图和所有 `<原图编号>_<增强编号>.jpg` 都在表中。然后在 `manual_review_decisions.csv` 写一条总审核记录：

```text
decision=EXCLUDE_LOCKED
reason=已核对17个locked病例，32个原图及640个增强派生物全部列入排除表
evidence_path=locked_exclusion_manifest.csv
```

如果不是 672/32/640/17，立即停止，不要训练。

## B. 看 6 张分类图

打开 `round1_visual_review_form.csv`，按行打开 `asset_path`。同时打开对应的 XML：

`data/血管数据集/分类数据集/扩充之前/annotations/<编号>.xml`

每张图只回答两个问题：

1. **图像可判读吗？** 至少能看清一条完整血管，且不是大面积模糊、反光或暗角遮住目标。看不清就写“不可判读”。
2. **XML 真配对吗？** XML 的 `<filename>`、`<size>` 和图像实际文件必须一致；框必须落在对应血管上。不能只因文件都叫数字就认为配对。

填写规则：

- 可判读且配对无明显问题：`decision=HOLD`，`reason=视觉上可判读，等待来源/语义总审核`；
- 不可判读：`decision=HOLD`，`reason=模糊/暗角/反光导致无法可靠判定`；
- XML 明显错配或尺寸为 0：`decision=REPAIR_REQUIRED`，把问题写清楚；
- 原始文件不要改。

已知重点：`315.xml` 的 filename 是 `351.jpg`，尺寸是 `0×0`，不能放行；`100.xml` 也必须核对其 filename 与本地图像是否一致。

## C. 看 4 对分割图、mask、JSON

每一行同时打开：

- `分割数据集/images/<编号>.jpg`；
- `分割数据集/mask/<编号>.png`；
- `分割数据集/annotations/<编号>.json`。

如果安装了 LabelMe：用 **Open** 打开 JSON，它会叠加显示 polygon；再把 mask 与原图并排看。没有 LabelMe 时，用图片查看器并排打开原图和 mask，JSON 用文本查看器检查 `imageWidth`、`imageHeight` 和 `shapes`。

每对只回答：

1. JSON 的尺寸、JPG 实际尺寸、mask 实际尺寸是否相同？
2. polygon/mask 是否覆盖图中同一条血管，而不是背景或另一张图？
3. 是否存在明显漏标、错配或空 mask？

填写规则：

- 三者一致且覆盖关系明确：仍填 `HOLD`，理由写“局部视觉一致，等待全量配对和病例来源审核”；
- 尺寸不一致、JSON 指向不存在路径、看不出对应关系：`REPAIR_REQUIRED` 或 `HOLD`；
- 不要把“看起来像”写成 `PASS`。

## 完成后交给我什么

只需要把填好的两个文件放回同一目录：

- `manual_review_decisions.csv`；
- `round1_visual_review_form.csv`。

我会据此生成第一版派生配对表和候选训练清单。P0 项未关闭前，原始数据仍保持 `HOLD`，不会进入训练。
