# 拯救计划增补：可回退的医学视觉编码器实验

日期：2026-09-21。交接对象：Claude。状态：**首轮冻结特征探针已执行完毕（2026-09-22）**，结果见文末「执行记录」。计划正文以下保持原样未改，只在文末追加实验结果。

## 任务范围

用户授权将医学视觉预训练加入原拯救计划：值得尝试，效果不好回退，其余工作照旧。本文件只增加隔离的编码器实验，不替换当前部署，不改变字段定义、固定字段、RAG 输入契约，不重开已失败的普通池化、相对分档、LoRA 搜索。

纠正旧结论：DINOv2/LoRA/HuluMed 的已有失败不能证明所有医学编码器都无效；反过来，医学预训练也不保证适用于甲襞。以下均为待检验假设。clarity/exudation/SVP/blood_color 是基线保留字段，不应称为已经通过上线验收。

## 1. 模型选择与证据

首轮只测两个新候选，避免模型动物园。

1. MedSigLIP：google/medsiglip-448。官方用于医学图像分类/检索的双塔编码器，视觉分支约 400M，448 输入；覆盖皮肤、眼科、病理、胸片与 CT/MRI 切片。只使用图像分支，不生成报告，不调用文本零样本提示。模型卡未给出甲襞性能证据。HF 需要登录接受 HAI-DEF 条款，非免条件下载；不能代表用户擅自签署条款或绕过 gate。
2. RETFound-DINOv2：固定 RETFound_dinov2_meh（ViT-L/14）。选择眼底彩照预训练，不选择 OCT。另加通用 DINOv2 ViT-L/14 控制臂，因为当前基线是 ViT-B/14。否则不能把容量增加解释为医学预训练收益。不因事后成绩换成 shanghai 权重；备选替换必须在读取评估结果前记录。

备选：RETFound-MAE natureCFP（如 DINOv2 版权重无法取得，仅用一个预先记录的替代）；MedSigLIP 不可取得时，先检查 BiomedCLIP 是否已有等价实验，若已有则复用，不重跑。两者访问均受阻时报告受阻，继续原计划其他工作。

RETFound-Green 可作后续低成本候选，不加入首轮。CT 专用 RADAR、PANDA、LiON 不进入甲襞编码器首轮。Cell-DINO 存在非商业研究限制，且细胞成像并非甲襞，本轮不选。HuluMed 已测，不包装成新候选。

来源：
- https://huggingface.co/google/medsiglip-448
- https://huggingface.co/google/medsiglip-448/tree/main
- https://www.nature.com/articles/s41467-026-70077-z
- https://github.com/rmaphoh/RETFound
- https://www.nature.com/articles/s41467-025-62123-z
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE_CELL_DINO_MODELS
- https://github.com/alibaba-damo-academy/damo-radar

注意：RETFound 仓库也列出了权重访问申请流程。开源代码不等于权重无门槛、不等于部署许可；记录每个实际 checkpoint 的条款与版本。

## 2. 实验与回退隔离

先读项目 AGENTS.md 和已有脚本，再落实实现。保留所有未提交用户改动。新建独立实验配置、环境、特征目录；不要原地升级旧环境或覆盖部署权重。

建议产物目录 artifacts/experiments/medical_encoder_transfer_20260921/，内含：
- protocol.yaml：运行前冻结字段、候选、切分、选择规则和门槛；
- model_registry.json：来源、revision、哈希、许可、输入规范；
- baseline_manifest.json：旧权重、旧预处理、PCA、分类头、阈值、字段路由及哈希；
- features/<encoder>/<view>/<split>/；
- predictions_oof.csv、loao_predictions.csv；
- paired_metrics.json、candidate_selection.json、decision.md；
- rollback_check.json：恢复旧配置后固定样本的概率与原输出一致。

默认不切换服务；失败时只标记候选 rejected，并继续旧模型。不要删除实验结果。即使通过研发门槛，也只获得候选资格，不自动上线。

## 3. 数据纪律

只用 development_fold；locked-47 不用于选模型/阈值，不纳入自监督训练。不要宣称整个项目历史 locked_cases_seen=0，只报告本次运行是否访问。患者身份未知的限制保留，不把检查级独立称为已证明患者级独立。

原始显微图与 rep_* 报告扫描图严格隔离。排除与外层测试/验证检查重合的人工框、增强图、伪标签和视频帧；无法判定来源的数据不得进入适配训练。

冻结公共编码器可一次提取全部开发图特征，但 scaler/PCA/分类器只能在训练折拟合。任何本地 SSL、LoRA、adapter 必须每折独立训练；不能先在全部开发图适配再交叉验证。LOAO 的留出档案同样不能参加任何适配，即使不读标签。

保留当前图级训练及检查级评估方式，不改成新 MIL。检查是统计单位，多帧、多次增强、种子不能增加独立样本数。

## 4. 迁移设计：先避免实现制造负迁移

### 输入和细节

不使用眼底的圆形裁剪、视盘定位、眼底背景去除及 OCT 预处理；甲襞不是眼底照片。保留 RGB，不做强颜色标准化、直方图匹配、血管图风格转换、AI 超分辨率。

首轮每张图使用同一个原始视野范围；各编码器采用自己的官方归一化和支持的分辨率。统一几何规则：等比例缩放后补边，补边颜色为归一化中性值。记录实际血管尺度与 patch 大小，补边 patch 池化时排除（如接口支持）。位置编码只按官方支持方式处理。

保存原部署 DINOv2-B 作为 anchor，另跑同一新加载器的 DINOv2-B 对照；不能把加载器差异归因于预训练。

首轮采用全局视觉向量，DINOv2-B/L 与 RETFound-DINOv2 统一 CLS 读出，MedSigLIP 用官方图像 embedding。各模型的 readout 差异属于系统比较，不能声称完全隔离了预训练因果效应。保留原五池化基线另作 anchor。

如果检查显示整图缩小使多数已有人工框仅覆盖很少 patch，允许一个预注册的细节实验：全图 + 固定 2×2 重叠瓦片（原图相对范围固定，20% 重叠），瓦片向量取均值后与全图拼接。通用 DINOv2 对照必须用完全相同视图。瓦片不增加样本数，不据字段结果反复改变裁剪范围；本实验与原空间池化失败路线的差别是保留输入局部像素细节，而不是同一特征图换池化。

### 解剖与类别

只迁移编码器特征，不迁移眼底疾病头或把分叉点当甲襞袢顶。papilla 不等于血管 apex；未明确对象定义前不拿畸形框监督 papilla。HF 的整袢类别与本地局部框类别不直接合并。

### 颜色、清晰度

血色与清晰度本身是目标，颜色抖动、灰度化、模糊增强可能主动抹掉信号。首轮不做这些增强；后续形态适配分支也不能默认接管颜色/清晰度字段。

## 5. 首轮：冻结特征探针

候选：当前部署 anchor、同加载器 DINOv2-B、DINOv2-L、MedSigLIP、RETFound-DINOv2-MEH。

主要救援终点固定 malformation_ratio；papilla 为次要终点，保留当前类别体系。clarity/exudation/SVP/blood_color 为保留能力监测。rbc_aggregation 可作探索性终点；静态 microthrombus 结果仅作相关性探索，不能证明事件检测。四个固定字段、空流速、绝对测量不作为本实验成功标准。

外层用固定 development_fold；内层只选预注册候选与极小分类头参数集，例如 C={0.003,0.03,0.3}。训练折 scaler -> PCA（最多64且不超过训练秩）-> LogReg；PCA whitening 设置与基线一致。头、帧聚合、类别映射和阈值规则统一。冻结特征确定性推理不机械跑五次相同 seed；bootstrap 使用检查级重采样。

外层测试只评估内层选出的方案；并列报告预先固定候选的 OOF 对照，但不得从这些结果事后挑模型再把该分数叫无偏性能。整个内层选择流程嵌入 LOAO，留出档案不参与选择。

保存所有概率、样本 ID、实际训练 ID、候选选择记录；报告 accuracy、delta、BA、每类召回、AUROC（二分类或明确的多分类定义）、校准和配对差。缺少类别时标记不可估计，不静默跳过后输出过窄 CI。

## 6. 第二轮：仅在有信号时做甲襞适配

不把小数据 SSL 当必做步骤。首轮没有目标字段收益，先检查权重加载、归一化、视野、细节；实现正确且细节对照仍失败，则关闭本模块，继续原计划。

若候选初步有用，最多追加一个轻量 adapter 实验；训练中保留初始化约束、使用弱几何增强，禁止颜色/模糊不变性。采用已验证可运行的 adapter 实现，不直接复制旧 DINOv2 LoRA 超参到不同模型。只允许一个预注册配方、3 个种子，epoch 用内层验证选择。选择完整嵌入外层评估，不在外层结果上重选种子。

有可用人工区域类别时优先衔接原计划的局部形态监督；全图与区域分支分开评估，先不增加自由 attention。如果缺少可靠局部语义，保留冻结胜者即可，不强制加入自监督训练。无标签数据能调整表示，不能创造关键点或医学金标准。

## 7. 继续、停止、采用

以下是研发投资规则，不是医学上线标准。

继续门槛：主要救援字段的配对 BA 点估计提高至少0.03，至少3/5外折同向，且没有少数类召回崩塌；这是进入有限适配的理由，不是宣称统计显著。

候选采用门槛：内层选择流程的主要救援字段 BA 提高至少0.05，检查级配对95% CI下界>0；LOAO平均改善>0，任一档案BA下降不超过0.03。它们是预设工程取舍，不是领域通用界值。仍须同时报告实际敏感度/特异度等，改善不等于绝对性能合格。若转而宣称次要字段成功，需独立复验或多重比较处理，不能事后交换主次终点。

若全局替换，四个保留字段BA各自不能下降超过0.02；如只适用于形态分支，可保留旧全局编码器，但分支路由必须在内层选定并评价完整流程，不能按外层测试逐字段挑赢家。

CI跨0、效果不稳定：判证据不足，不强称无效或有效；本轮默认不采用、不扩展参数搜索。全失败就回退。若仅DINOv2-L有效而RETFound-L不优于通用L，只能归因候选系统/容量收益，不能宣传医学预训练有效。

## 8. 与原拯救计划衔接

报告实际积分恢复/监督修复仍继续；不把积分作为推理输入。局部人工形态监督、视频时空信息、独立验收数据与字段级产品标准照旧。冻结医学编码器筛选可作为独立低成本实验先做，无需等待全部OCR；一旦修正标签，所有比较臂必须使用同版标签重评，不跨版本比。

本模块不能解决物理标定、静态缺少时间信息、患者身份不明、少数阳性不足、无独立验证。成功时只增加候选模型；真实上线仍要求未参与选择且有对应参考真值的数据验收。

Claude最终交付：一张同病例配对结果表，分清数据/预处理/容量/医学预训练/适配各因素；明确 adopted / rejected / inconclusive / blocked_access；附回退校验和本次是否接触locked。不要把本计划写成已完成实验，不要承诺字段数或90%准确率。

# 原拯救计划主体（本增补必须与以下路线一起执行）

## A. 字段治理和产品边界

当前字段按输入条件和证据状态分组，不把所有字段当同一个视觉任务。

### A1. 保留为当前基线的静态字段

- clarity
- exudation
- blood_color
- subpapillary_venous_plexus / SVP

这些字段已有视觉信号，但只能作为开发基线，不能因为开发 OOF 超过众数就称为上线通过。必须继续报告档案外稳定性、校准、拒答和低患病率 PPV/NPV。

### A2. 固定为 `not_modelled` 的字段

- vasomotion：单位为次/min，静态图没有时间基，阳性仅3例，AUROC 0.117。
- wbc_count：单位为个/15s，静态图没有时间基，阳性仅3例，AUROC 0.276。
- sweat_duct：单位为个/一指甲襞，需要完整视野，development 仅2例异常，locked 异常为0。
- hemorrhage：单位为管襞/一指甲襞，需要完整视野，阳性约12例，AUROC 0.331。

固定众数不是患者测量结果。接口必须输出 `status=not_modelled` 或 `source=default`，禁止将默认值送入 RAG 作为“无异常”事实。

`flow_speed_um_s` 删除，不固定：整列为空、没有可用标签和物理标定。

### A3. 暂停而非固定

`microthrombus` 不得固定为无：约74例有非零标签，固定会漏掉全部异常。它的单位为个/min，优先检查视频与报告观察过程是否对应；没有对应视频就不得声称完成该字段。

### A4. 结构形态优先目标

- malformation_ratio：已有人工框和折外检测器显示视觉信号，优先做区域形态监督。
- papilla：作为次要目标，先定义“乳头”与“管襻顶端/异常顶端”的对应关系。
- rbc_aggregation：只有在局部形态或视频时序提供新增信息后继续。
- crossing_ratio：现有人工框/检测证据弱，暂停专项攻关。

### A5. 测量字段

affferent/efferent/apex diameter、loop_length、capillary_count 暂不输出绝对微米或条/mm，除非补齐设备标定和参考定义。可以研究相对等级，但不得把相对同群偏粗/偏细改名为临床异常。

## B. 第一阶段：冻结数据和评估协议

1. 只使用 `development_fold`，locked-47 永不用于模型选择、阈值选择、自监督适配或特征选择。
2. 固定唯一标签版本、病例级切分、字段映射、预处理、PCA、分类头和阈值。
3. 禁止按测试结果逐字段挑模型。任何候选必须在训练折内选择，然后一次性评估外折。
4. 每个实验保存概率、病例ID、折号、训练病例集合、配置哈希和是否访问locked的审计文件。
5. 每个字段同时报告 accuracy、相对众数delta、AUROC、BA、每类召回、灵敏度、特异度、校准、档案分层结果和95% CI。
6. 研究改善不等于上线通过；最终必须有未参与选择且有对应参考真值的独立验收数据。

## C. 第二阶段：恢复逐项医生积分和报告标准

从报告图片中提取字段原值、正常范围、逐项积分、组小计、总分和综合判断。OCR结果必须人工抽查；不能用总分反推字段并称为新金标准。

验证：

- 真实积分是否与 `score_rules_v3.json` 一致；
- 相同字段值是否总对应相同积分；
- 逐项积分能否重现 periloop/flow/morphology 小计；
- 真实积分是否包含字段表没有记录的信息。

分支：

- 若积分完全由现有字段决定，只用于加权评估，不增加监督；
- 若发现可证明的解析或规则错误，单独修正标签后重跑旧基线；
- 若积分含有现有字段未记录的信息，才做一次“原字段目标 + 真实积分辅助损失”实验，主指标仍是字段能力。

不得用加权总分正确掩盖具体字段错误。积分只能在训练折作为辅助目标，不能作为推理时输入。

## D. 第三阶段：局部人工区域监督

使用已有人工框和类别标签训练局部形态表示：

```text
人工实例/类别
→ 局部血管区域编码
→ 形态类别与几何特征
→ 固定病例级聚合
→ malformation_ratio / papilla / rbc_aggregation
```

不只使用框数量，也不把所有局部实例简单平均。候选统计包括类别概率、异常实例比例、曲率、宽度、面积、长宽比、空间分布和质量分数。聚合规则必须在训练折决定。

主终点为 `malformation_ratio`，次终点为 `papilla`。若人工区域指标上升但病例字段不升，则停止。若此前已有完全等价区域监督实验，也不得换名称重跑。

## E. 第四阶段：医学视觉预训练和眼底结构迁移

医学编码器实验按本文件前半部分执行，但必须明确其作用边界：

眼底迁移能帮助：

- 血管分割；
- 骨架提取；
- 端点、分叉点和曲率表示；
- 结构形态预训练；
- 作为甲襞局部形态模型初始化。

它不能直接解决：

- 甲襞疾病标签不足；
- 医生报告主观字段；
- microthrombus、vasomotion 等时间单位字段；
- hemorrhage、sweat_duct 等完整视野字段；
- 甲襞专属类别定义。

眼底迁移仅在以下管线中有意义：

```text
RETA/DRIVE/VETO 血管拓扑预训练
→ 甲襞无标签域适配
→ 本地人工框训练 malformation/papilla
→ 与当前 DINOv2 初始化配对比较
```

不能把眼底疾病分类头迁移到甲襞，也不能把眼底分叉点当甲襞袢顶点。若 `malformation_ratio` 或 `papilla` 的病例级 OOF 没有稳定提升，停止眼底迁移。

## F. 第五阶段：视频时空建模

只对能确认病例对应、有效时长、帧率、非重复、同次检查的视频执行。先做稳定画面和血管追踪，沿中心线构建时空图/kymograph，提取位移、连续性、停滞、强度波动和周期特征。

设置三臂：

1. 静态图像特征；
2. 视频帧平均特征；
3. 运动/时空特征。

主攻 `flow_state`、`rbc_aggregation`，条件性加入 `microthrombus`。只有当运动特征超过帧平均，且视频事件语义与报告单位匹配，才进入下一轮。不得把运动变化直接命名为微血栓。

## G. 第六阶段：测量字段和完整视野

对测量字段先分析实际预测跨正常范围的错误，而不是继续拟合原始微米值。只有明显异常可稳定识别时才考虑相对等级和拒答；否则保持不支持。

对 hemorrhage/sweat_duct，只有能够恢复同一检查的完整指甲襞视野、明确分母、并有足够阳性病例，才建立全局计数模块。不能用局部裁剪图预测“一指甲襞”的计数。

## H. 第七阶段：采用与回退

候选进入有限适配的研发门槛：主要字段配对 BA 提高至少0.03，至少3/5外折同向，且没有少数类召回崩塌。

候选采用门槛：主要字段提高至少0.05，配对95% CI下界>0，LOAO平均改善>0，任一档案不下降超过0.03；四个现有保留字段全局替换时各自下降不得超过0.02。

这些是项目投资门槛，不是医学通用上线标准。CI跨0或结果跨档案不稳定时，标记 `inconclusive`，不扩展搜索。

全失败、访问受阻、许可不适合或回退校验不通过：候选标记 `rejected`/`blocked_access`，恢复原部署模型。保留所有实验产物，不删除旧权重。

## I. 最终产品形态

Tier A：经字段级和独立验证支持，可输出概率、适用范围和拒答。

Tier B：只有排序或开发信号，只能输出“建议复核”，不能生成确定医学事实。

Tier C：`not_modelled`，包括四个固定字段、空流速及未解决的时间/完整视野/标定字段。

RAG 只读取 Tier A；Tier B 和 Tier C 必须带状态和原因。模型不得输出疾病诊断、总分或综合判断作为图像事实。

## J. Claude 交付格式

最终交付一张候选对照表，至少包含：候选编码器、数据来源、预处理、容量对照、医学预训练因素、人工区域监督、视频因素、字段、OOF/LOAO指标、CI、档案分层、是否访问locked、许可证、状态（adopted/rejected/inconclusive/blocked_access）和回退校验。

不要把开发集改善写成上线达标，不要承诺字段数量或90%准确率。原拯救路线和医学编码器路线必须作为一个整体执行；医学编码器只是其中一个可回退分支。

---

# 执行记录

本段由 Claude 在执行时追加。每次实验只往下追加，不改写上面的计划正文。所有数字均为 **development（186 例）** 折外结果，**locked-47 本轮完全未读取**（四个臂的 `metadata.json` 均记录 `locked_cases_seen: 0`）。开发集改善不等于上线达标。

## 执行记录 1：首轮冻结特征探针（2026-09-22）

产物目录：`artifacts/experiments/medical_encoder_transfer_20260921/`

复现命令：

```bash
PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm anchor_dinov2b_deployed --batch-size 6
PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm dinov2b_newloader --batch-size 8
PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm dinov2l_capacity --batch-size 8
PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm biomedclip_medical --batch-size 8
PYTHONIOENCODING=utf-8 python scripts/eval_medical_encoders.py
PYTHONIOENCODING=utf-8 python scripts/check_medical_encoder_gates.py
PYTHONIOENCODING=utf-8 python scripts/isolate_capacity_vs_geometry.py
PYTHONIOENCODING=utf-8 python scripts/write_medical_encoder_deliverables.py
```

本地 GPU（RTX 5070 Laptop 8 GB），python 为 `anaconda3/envs/pytorch_gpu`。服务器无卡模式，本轮未使用服务器。

### 1.1 §1 候选访问结论：两个计划中的医学候选都拿不到

| 候选 | 状态 | 证据 |
|---|---|---|
| google/medsiglip-448 | `blocked_access` | 匿名 401，**带账号现有 token 仍 403**；需接受 HAI-DEF 条款 |
| RETFound_dinov2_meh | `blocked_access` | 带 token 403 |
| RETFound_mae_natureCFP（预登记备选） | `blocked_access` | 同为 gated 仓库 |
| BiomedCLIP ViT-B/16 | 已运行 | 依 §1 回退条款启用：MedSigLIP 不可得，且 grep 确认此前无任何 BiomedCLIP 实验 |

没有代替用户签署任何条款，没有绕过 gate。**因此本轮没有真正测到计划想测的 MedSigLIP / RETFound**，医学预训练假设只被 BiomedCLIP 这一个（且分辨率被混淆的）系统检验过。

### 1.2 四个臂

| 臂 | 角色 | 权重 | 几何 | 维度/网格 |
|---|---|---|---|---|
| `anchor_dinov2b_deployed` | anchor | DINOv2 ViT-B/14 | 直接 resize 518×686，无 padding（复现部署 `native`） | 768 / 37×49 |
| `dinov2b_newloader` | loader 对照 | 同上，同权重 | 保比例 resize + 中性 pad 到 518×518 | 768 / 37×37 |
| `dinov2l_capacity` | 容量对照 | DINOv2 ViT-L/14 | 新 loader | 1024 / 37×37 |
| `biomedclip_medical` | 医学候选 | BiomedCLIP PMC-15M | 新 loader，224×224 patch16 | 768 / 14×14 |

四臂均 1708 图 / 186 例，零 NaN。读出为 `cls`，头为 StandardScaler→PCA(64)→LogisticRegression，C∈{0.003,0.03,0.3} 仅在训练折内选（nested），`class_weight=null`。

**一个已修的错误必须记录**：anchor 最初和 loader 对照 **逐字节相同**（`np.array_equal` → True），对照等于什么都没测。原因是部署几何是 518×686 直接 resize 而新 loader 是 pad 到方形。已在任何指标算出之前修正（`protocol.yaml` 修正案 A1），修正后两臂 row0 相关 0.904、非同一文件。另外 `.tmp_probe/feat/dinov2/features.npy` 没有 `metadata.json`、与 anchor row0 相关仅 0.416，来源不可考，**未**用作部署 anchor（修正案 A2）。

### 1.3 主终点 malformation_ratio（n=162）

| 臂 | acc | delta（对众数） | BA | AUROC | 漏检率 | 配对 BA 增益 vs anchor | 95% CI |
|---|---|---|---|---|---|---|---|
| anchor DINOv2-B 部署几何 | 0.6296 | +0.0617 | 0.6090 | 0.7130 | 0.5429 | — | — |
| BiomedCLIP（医学） | 0.6605 | +0.0926 | 0.6430 | 0.6655 | 0.4857 | +0.0340 | [−0.036, +0.105] |
| DINOv2-B 新 loader | 0.6296 | +0.0617 | 0.6124 | 0.6991 | 0.5143 | +0.0034 | [−0.050, +0.053] |
| DINOv2-L（容量） | 0.6914 | +0.1235 | 0.6804 | 0.7287 | 0.4000 | **+0.0714** | **[+0.011, +0.136]** |

### 1.4 §H 门槛判定（`gate_check.json`）

| 臂 | 研发门槛（≥0.03、≥3/5 同向、少数类不塌） | 采用门槛（≥0.05、CI下界>0、LOAO均值>0、任一档案不跌>0.03、四保留字段不跌>0.02） | 判定 |
|---|---|---|---|
| BiomedCLIP | 通过（+0.034，3/5，少数类召回 0.457→0.514） | **不通过**：CI 含 0；LOAO 均值 −0.0288；archive2 −0.1158；SVP −0.0776 | `rd_gate_passed` |
| DINOv2-B 新 loader | 不通过（+0.003，2/5） | 不通过 | `inconclusive` |
| DINOv2-L | 通过（+0.0714，**5/5**，少数类召回 0.457→0.600） | **全部通过**：CI 下界 +0.011；LOAO 均值 +0.0684；最差档案 +0.0435；保留字段最差 −0.0163 | `adopted_by_numbers` |

LOAO（C 只在保留档案内选）：DINOv2-L 在三个档案上 BA 0.636 / 0.633 / 0.704，delta +0.098 / +0.100 / +0.098，**三个档案同向**——这是 malformation_ratio 第一次出现档案无关的正增益（此前 archive3 门槛一直不过）。

### 1.5 归因：赢的不是医学预训练，是容量（`attribution.json`）

把三个差值重新配对，才能分开 loader、容量、医学三个因素：

| 对比 | 隔离的因素 | malformation_ratio | clarity | exudation | SVP | blood_color | papilla |
|---|---|---|---|---|---|---|---|
| `geometry_only`（B新loader vs anchor） | 只换 loader/几何 | +0.003 | +0.022 | +0.005 | −0.003 | −0.030 | +0.020 |
| `capacity_only`（L vs B，同 loader） | 只加容量 | **+0.068 [+0.014,+0.123]** | +0.027 | −0.022 | +0.003 | +0.057 | −0.072 |
| `medical_only`（BiomedCLIP vs B，同 loader） | 医学权重（混淆 224/patch16） | +0.031 | −0.022 | +0.017 | **−0.074 [−0.138,−0.007]** | +0.039 | −0.040 |

结论写清楚：**主终点上唯一 CI 不含 0 的正效应来自 ViT-B→ViT-L 的容量，不是医学预训练**。几何改动几乎没有影响（全字段 CI 含 0），也就是说 §4「实现制造负迁移」这一风险在本轮没有兑现。医学臂唯一 CI 不含 0 的效应是 SVP **下降** 0.074。

BiomedCLIP 的 null 不能读成「医学预训练无用」：它同时是 224×224、patch 16，token 数 196 对 1813，分辨率与容量都被削。可以说的只有——**这个医学系统在我们的字段上没有比通用 DINOv2 更好**。

### 1.6 其它字段（同一配置，附带得到）

| 字段 | n | anchor BA | 最好臂 | 最好 BA | delta（对众数） |
|---|---|---|---|---|---|
| clarity | 185 | 0.7887 | DINOv2-L | 0.8371 | +0.3297 |
| exudation | 183 | 0.7600 | BiomedCLIP | 0.7819 | +0.2732 |
| SVP | 184 | 0.7750 | anchor | 0.7750 | +0.2120 |
| blood_color | 181 | 0.7142 | DINOv2-L | 0.7415 | +0.1878 |
| rbc_aggregation | 182 | 0.4967 | 无 | — | −0.0055（全臂塌成单类，L 臂 `collapsed=True`） |
| papilla（3 类） | 185 | 0.4141 | B 新 loader | 0.4336 | +0.0432 |

rbc_aggregation 和 papilla 的结论与既有记录一致：**没有被医学编码器救活**。§A4 写的「rbc_aggregation 只有在局部形态或视频提供新增信息后继续」依然成立，本轮不构成继续的理由。

### 1.7 本轮的实际结论与下一步

1. 按 §H 的算术，**DINOv2-L 满足采用门槛**，但它是 `capacity_control`，不是医学候选。采用它等于承认「上一步该做的是换更大的通用编码器」，这不是本增补要验证的假设。
2. **医学预训练假设仍未被真正检验**：计划指定的两个候选都是 403。只有拿到 MedSigLIP 或 RETFound 权重（需用户自己接受条款）才能填这一格。
3. 主终点的绝对水平仍然低：最好 BA 0.6804、AUROC 0.7287、**漏检率 0.40**。malformation_ratio 的真值是医生分档的报告值，不是逐血管计数比例；这个成绩不支持任何上线声明。
4. locked-47 本轮未读，也**不建议**现在读——预算已超支（7 次消耗、3 次在其上做过模型选择）。
5. §H 的「四个保留字段各自下降不得超过 0.02」在 DINOv2-L 上通过（最差 exudation −0.0163），但 exudation 的方向是负的，全局替换会牺牲一个目前唯一站得住的字段的一部分。

### 1.8 §2 产物清单（已写出）

`protocol.yaml`（含修正案 A1/A2）、`model_registry.json`（4 臂：来源/哈希/许可/输入规范）、`baseline_manifest.json`、`candidate_selection.json`（6 候选，3 个 `blocked_access`）、`rollback_check.json`（`deployment_changed: false`，回退是空操作）、`predictions_oof.csv`、`paired_summary.csv`、`paired_metrics.json`、`gate_check.json`、`attribution.json`、`loao_predictions.csv`（24 行）。

**已知局限**：部署当年加载的是 facebook `.pth`，本轮加载的是 timm safetensors 转换版，两者哈希不同（`baseline_manifest.json` 已记录 `hashes_match: false`），所以 anchor 是对部署几何的**重实现**，不是逐位重放；跨模型族读出方式不同，本轮是系统比较，不能把预训练单独归因；`medical_only` 分辨率混淆见 §1.5。
