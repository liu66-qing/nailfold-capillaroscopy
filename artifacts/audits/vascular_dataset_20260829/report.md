# 血管数据集只读治理审计（2026-08-29）

## 执行摘要

本审计只读运行、未使用 GPU，未用 locked_test 进行模型选择或调参，未修改既有 v1 基线、模型、标签或评测结果。机器可核对明细见 audit.json、source_layers.json，审计脚本见 audit_vascular_dataset_20260829.py、audit_source_layers_20260829.py。

结论：这批数据确实包含此前流程未纳入的“血管分类/检测框”和“血管分割/多边形”监督，但它们不是当前临床字段的直接金标准。可作为独立的血管检测/分割预训练数据；要映射到 capillary_count、crossing_ratio、malformation_ratio 及几何字段，必须先完成病例级去重、标注错配修复、坐标/放大校准和独立验证。对 clarity、blood_color、exudation、hemorrhage、subpapillary_venous_plexus、papilla 没有直接标签支持。

最重要的阻断风险：分类 XML 有 485/582 个尺寸声明与当前本地副本不一致、9 个异常框；分割 JSON 的 imagePath 3,161/3,161 均指向不存在的路径，且 JSON 与 images、mask 从约 1595 号开始发生顺序漂移（1,557 个尺寸不匹配），3160、3161 两个 JSON 无对应图像/mask。XML 中的 C:\\Users\\Nicolas、C:\\Users\\ming 等是同事机器的来源路径，不是本地目录缺失证据；46 个 filename 不能直接配对只表示当前导出包缺少同名映射，需用原图内容/哈希重建。未经修复不得把文件级随机划分或当前同名配对用于训练。

## 数据资产盘点

| 资产 | 事实数量/格式 | 证据 |
|---|---:|---|
| 分类扩充前 | 582 .jpg + 582 Pascal VOC .xml | data/血管数据集/分类数据集/扩充之前/{images,annotations} |
| 分类扩充后 | 11,600 .jpg、11,600 YOLO .txt，580 个原图编号组、每组 20 张 | data/血管数据集/分类数据集/{image,labelYOLOs}；audit.json classification |
| 扩充缺失 | 原图 418、423 无对应 _0.._19 组 | 扩充之前/images/418.jpg,423.jpg 与扩充组统计 |
| 分割 | 3,159 .jpg、3,159 .png mask、3,161 LabelMe .json | data/血管数据集/分割数据集/{images,mask,annotations} |
| 分割 JSON 内嵌图像 | 3,052 有 imageData，109 无 | audit.json segmentation.json_stats |
| 压缩包 | 分割数据集.zip 36,446,574 B、data.zip 8,133,315 B | data/血管数据集/分割数据集*.zip |
| 既有病例清单 | 233 unique cases = 186 development + 47 locked_test | artifacts/manifest/locked_evaluation_v1.csv |
| ANFC-THU 本地数据 | train 257 图像/5,363 标注，valid 64 图像/1,392 标注；COCO polygon+bbox，全部 640×640 | data/ANFC-THU.v1i.coco/{train,valid} |

图像尺寸不是单一放大倍率：分类扩充前 XML/图像出现 244 种尺寸组合（例如 844×617、1024×768、3072×2048 等），扩充组内部尺寸一致但组间不同；分割 crop 尺寸高度/宽度高度可变。ANFC-THU 是独立 Roboflow 导出层，README 明确自动方向校正并 Fit 到 640×640 黑边、无图像增强，不能与同事血管包或本地 recovered 原始病例混为一套。

## 来源分层（修正）

- 同事标注包：data/血管数据集。XML 绝对路径来自同事电脑，仅作来源元数据；本地审计改用文件内容、尺寸和哈希，不把这些路径直接当作本地缺失。
- 本地原始病例层：data/recovered_archive1、recovered_archive2、recovered_archive3，对应现有 233 例角色清单，是病例级重叠/泄漏审计的参照。
- 本地 ANFC 层：data/ANFC-THU.v1i.coco，Roboflow/CC BY 4.0 导出，321 张 640×640 图像、6,755 个 COCO 标注，321 个来源组在 train/valid 间无交叉；与血管包和 recovered 图像均未发现 MD5 精确重叠。类别为 capillary、abnormal、aggregation、blur、hemo、normal；类别 0 capillary 在实际实例中为 0。详细证据见 source_layers.json。

ANFC 的类别定义与同事包的 vessel/malformed_vessel/cross_vessel 不同，不能直接合并标签 ID；它适合作为独立 capillary/abnormal/aggregation 检测或分割预训练层。

## 重叠与泄漏审计

- 对分类扩充前、扩充后和分割图像共 15,341 个文件，与 recovered_archive1/2/3 共 3,226 个图像做 MD5 比对，发现 123 个 recovered 文件精确重叠，涉及 71 个 recovered 病例目录：50 个 development、17 个 locked_test、4 个不在 233 例清单中的目录。明细在 audit.json 的 overlap_with_recovered.exact_overlap；这 17 个 locked 病例只记录，不得训练、调参或模型选择。
- 分类扩充前内部有 10 组完全相同图像（20 个文件），例如 124.jpg=330.jpg、190.jpg=400.jpg、364.jpg=505.jpg、388.jpg=543.jpg；证据为 audit.json 的 exact_duplicate_image_hash_groups=10。应按原图级 dedup 后再划分。
- 扩充图按 原图编号_增强序号 归组，580×20=11,600，不是 11,600 个病例。任何划分必须先按原图编号/病例分组，再在组内使用增强版本，禁止把增强图分散到 train/val/test。
- 分割数据没有病例 ID；JSON imagePath 保留来源 crop 名称（如 ..\\images\\100_1.jpg），目录内实际文件为顺序编号 1.jpg 等，导致路径不可解析。不能仅按 JSON stem 与图像/mask stem 连接。

## 分类标注审计

### XML（扩充前）

- 3,835 个目标：vessel 1,779、malformed_vessel 1,508、cross_vessel 547、拼写变体 corss_vessel 1。corss_vessel 必须人工核对，不能静默并入。
- truncated=1 共 48 个（vessel 24、malformed 18、cross 6）；difficult 均为 0。
- 9 个框存在越界、零宽/零高或解析异常；完整列表见 audit.json 的 invalid_boxes。
- 485/582 XML 的 size 与当前本地图像副本不一致；46 个 XML 的 filename 在当前导出包中找不到同名文件。XML 还保留同事电脑绝对路径（C:\\Users\\Nicolas...、C:\\Users\\ming...），这不是本地原始数据缺失的证明，必须回到同事原始目录或用图像哈希/像素内容重建映射。
- 目标粒度是单个血管/异常血管/交叉血管的矩形框，不是病例级临床评级，也没有标注者、置信度或时间信息。

### YOLO（扩充后）

- 76,360 个框行，类别 ID 分布：0=35,540、1=10,860、2=29,960。与 XML 对照可推断 0 vessel、1 malformed_vessel、2 cross_vessel，但数据包没有正式 classes.txt，纳入前应固定并记录映射。
- 7 行异常（宽或高为 0，集中在 156_5.txt、156_8.txt、437_0.txt、456_10.txt、456_14.txt、566_4.txt、76_9.txt）；完整路径在 audit.json 的 yolo_bad。
- 这是检测框监督；增强后框数量约为扩充前 20 倍，不能按框数当作独立样本量。

## 分割标注审计

- 4,183 个 LabelMe polygon，标签为 vessel 3,203、_background_ 980；全部 shape_type=polygon，结构性点数/数值检查未发现坏多边形。
- 每个 mask 为单通道 L，像素值仅 0/255（统计见 audit.json mask_value_top20），适合二值血管前景；但 _background_ 多边形与 mask 的语义需人工抽查确认，不能直接假设为负类真值。
- JSON 3,161 个、图像/mask 各 3,159 个；3160.json、3161.json 无对应图像和 mask。所有 JSON 的 imagePath 均无法按字面路径解析；从约 1595 号起同名图像/mask 与 JSON 声明尺寸错位，共 1,557 个不匹配，提示索引漂移。应以 imageData、像素哈希/尺寸和原始压缩包重新建立一一对应关系。
- 3,052 个 JSON 含 base64 imageData，可用于恢复图像；109 个无内嵌图像，必须从原始包或人工补齐后才可用。分割 crop 尺寸从几十像素到数百像素不等，需保留原始尺寸并显式记录放大倍率。

## 与现有任务字段的映射

| 当前字段 | 直接支持 | 需要转换/验证 | 结论 |
|---|---|---|---|
| clarity | 无 | 可用于视觉预训练或清晰度候选 | C，仅辅助 |
| blood_color | 无颜色类别 | RGB 外观可作预训练 | C |
| exudation、hemorrhage | 无 | 无对应病灶类别 | D，不可直接使用 |
| subpapillary_venous_plexus、papilla | 无 | 无解剖结构类别 | D |
| capillary_count | 血管实例/前景 | 从框或 mask 计数，需去重、连通域/骨架规则及病例级聚合 | B，需转换 |
| crossing_ratio | cross_vessel 框 | 需定义分母、交叉事件与视野级聚合 | B |
| malformation_ratio | malformed_vessel 框 | 需统一 typo、分母及截断框处理 | B |
| 几何字段（直径、loop length、afferent/efferent、output/input 等） | vessel mask/框提供形态线索 | 需像素到物理单位校准、骨架/中心线算法和独立验证 | B/C（候选特征） |

分类/分割监督此前在现有病例级字段流水线中确实没有作为独立监督源纳入；但由于病例 ID、放大倍率和标注规范缺失，不能直接并入 233 例标签表。

## 可用性分级

- A（可直接训练）：仅限完成配对校验后、把任务定义为独立血管检测或二值分割；当前目录没有可无条件 A 的文件子集。
- B（需转换）：完成原图/病例级 dedup、修复 XML/YOLO 坐标、重建分割 JSON-image-mask 映射后，用于 capillary_count、crossing_ratio、malformation_ratio 和几何候选特征。
- C（只能辅助/预训练）：对 clarity、blood_color 及所有临床字段做视觉表示预训练、候选区域提议或误差分析；不能当作这些字段标签。
- D（不可用/阻断）：7 个异常 YOLO 行、9 个坏 XML 框、corss_vessel 未核对记录、3160/3161 无图像/mask 的 JSON、109 个无 imageData 且无法从原包恢复的 JSON，以及任何无法确认病例归属或与 locked 重叠的样本。

## 风险、结论与安全纳入清单

1. 外部/同学标注没有标注者资质、版本、协议或一致性统计；本报告将其称为外部监督，不称为金标准。data/微循环培训资料 中的 PDF/CAJ/DOC 是背景资料，未发现与本数据包绑定的正式标注规范文件；PDF 中文字体缺失导致文本抽取不可靠，需人工阅读原件确认。
2. 任何训练/验证划分必须先按 recovered 病例目录和分类原图组 dedup，再按病例划分；增强版本只能跟随原图所属 split。locked 相关的 17 个病例及其所有增强派生物永久排除出训练和调参。
3. 建议安全纳入顺序：先保留 audit.json 作为只读基线；对 71 个 recovered 重叠病例和 4 个未入清单病例建立排除表；仅对剩余、可追溯且完成图像-标注一一核验的原图组建立候选训练清单；分割先用 imageData/原始 zip 重建配对，再由独立人员抽查 mask；最后以 development 内部交叉验证评估，禁止触碰 locked 结果。
4. 在修复和独立复核完成前，推荐用途仅为 C 级预训练/候选区域生成；不要把 11,600 增强图报告成病例数，也不要把检测/分割类别直接改写为临床字段标签。
