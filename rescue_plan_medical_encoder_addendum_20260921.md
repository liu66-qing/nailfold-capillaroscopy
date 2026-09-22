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

## 执行记录 2：范围更正（2026-09-22，用户核对后追加）

用户核对执行记录 1 后指出范围问题。以下四条为更正，优先于执行记录 1 中任何相反读法。

**1. 本轮只完成 7 个字段，不是「除固定字段外的完整字段实验」。**

实际评估过的字段仅有：`malformation_ratio`、`clarity`、`exudation`、`subpapillary_venous_plexus`、`blood_color`、`papilla`、`rbc_aggregation`。这 7 个是我在 `eval_medical_encoders.py` 里写死的 `BINARY_MAP` + `MULTICLASS`，不是按字段清单推导出来的。执行记录 1 的 §1.6 标题写「其它字段」有误导性，它只是这 7 个中的另外 6 个。

**2. 医学编码器假设尚未覆盖全部非固定字段。**

至少 `capillary_count`、`afferent_diameter`、`efferent_diameter`、`apex_diameter`、`loop_length`、`crossing_ratio`、`flow_state`、`microthrombus` 这 8 个非固定字段在本轮**完全没有跑过任何臂**。它们在本轮既不是「没有信号」也不是「已关闭」，而是**未测试**。此前记录里关于这些字段的结论来自其它实验、其它配置，不能当成本轮医学编码器的结论。

**3. DINOv2-L 的 `adopted_by_numbers` 只是 malformation_ratio 上的容量对照结果。**

它不表示医学预训练成功——L 臂是通用 LVD-142M 权重，医学因素为零。它也不能直接作为全局部署模型：
- 该判定来自 7 个字段中的 1 个主终点，其余 14 个非固定字段未参与；
- `attribution.json` 显示赢的因素是容量（ViT-B→ViT-L），不是本增补要验证的医学预训练；
- 追加的部署读出复核（执行记录 3）显示这个增益在固定配置下不成立。

**4. BiomedCLIP 的结果不能用来判定医学预训练无效。**

它同时受 224×224 输入、patch 16、以及 CLIP 体系与 DINOv2 自监督体系的差异影响（token 数 196 对 1813）。本轮能说的只有「这一个医学系统在这 7 个字段上没有比通用 DINOv2 更好」。计划 §1 指定的两个医学候选（MedSigLIP、RETFound）带账号现有 token 仍 403，**医学预训练假设本身仍未被检验**。

**后续纪律**：补齐必须先产出完整字段矩阵（标签是否存在、适合何种目标、输入模态、能否合理评估），确认后才训练。不得按字段事后挑最好模型；不得把未测试字段写成无信号或已关闭。

## 执行记录 3：部署读出下容量增益反向（2026-09-22）

在暂停前已完成一条复核，结果必须记录，因为它推翻了执行记录 1 的采用判定。

追加一个臂 `dinov2l_deployed_geometry`（ViT-L/14，**部署几何** 518×686 直接 resize），使容量成为与 anchor 之间唯一变化的因素；然后用**部署固定配置**（5 池化 mean/topk_mean/max/cls/std，概率平均，C 固定 0.03，PCA 64，seed 20260917，无任何调参）重跑两臂。

复现：

```bash
PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm dinov2l_deployed_geometry --batch-size 4
PYTHONIOENCODING=utf-8 python scripts/recheck_capacity_deployed_readout.py
```

| 字段 | ViT-B BA | ViT-L BA | 配对增益 | 95% CI |
|---|---|---|---|---|
| malformation_ratio | 0.6304 | 0.6036 | **−0.0269** | [−0.083, +0.026] |
| clarity | 0.8376 | 0.8211 | −0.0165 | [−0.060, +0.028] |
| exudation | 0.7493 | 0.7710 | +0.0217 | [−0.026, +0.069] |
| SVP | 0.8321 | 0.8146 | −0.0175 | [−0.071, +0.032] |
| blood_color | 0.6904 | 0.7119 | +0.0215 | [−0.039, +0.083] |
| rbc_aggregation | 0.4933 | 0.5000 | +0.0067 | [0.000, +0.017]（两臂均塌成单类） |
| papilla | 0.4149 | 0.4163 | +0.0014 | [−0.066, +0.063] |

**主终点上容量增益从 +0.0714 变成 −0.0269，方向反转，7 个字段的 CI 全部含 0。** 分档案看（`capacity_recheck_by_archive.csv`），malformation_ratio 三个档案全部为负（−0.018 / −0.019 / −0.051）。

因此：**执行记录 1 §1.4 的 `adopted_by_numbers` 不成立**，DINOv2-L 在部署配置下没有可采用的增益。两次运行的差别有两个可能来源（读出方式：单 cls 对 5 池化集成；C：折内选择对固定 0.03），本轮**未做隔离实验**，所以我不能说是哪一个，只能说这条增益**不稳健于配置**。按 §H「结果跨配置不稳定时标记 inconclusive，不扩展搜索」，DINOv2-L 状态改为 `inconclusive`。

这也顺带说明执行记录 1 的一个方法学问题：折内选 C 会放大候选之间的差异，和固定配置的部署口径不可直接比较。后续补齐字段矩阵时应统一用固定配置。

## 执行记录 4：完整字段矩阵（2026-09-22，训练前，待确认）

按用户要求，补齐前先产出字段矩阵，**本节不含任何模型结果**。来源为标签文件本身：`server_code_audit/locked_evaluation_v1_reviewed.csv`，development 186 例，locked-47 未读。

复现：

```bash
PYTHONIOENCODING=utf-8 python scripts/build_field_matrix.py
```

产物：`field_matrix.json`、`field_matrix.csv`。

### 4.1 15 个非固定字段

`n` 为映射后可用病例数；`最小类`为**映射后目标**的最小类计数（不是原始取值表，原始表里有一次性脏字符串会伪造出极小类）。

| 字段 | n | 类数 | 最小类 | 众数占比 | 首轮已测 | 观测单位 | 静态图可观测 | 允许的目标 |
|---|---|---|---|---|---|---|---|---|
| clarity | 185 | 2 | 91 | 0.5081 | 是 | 静态图像外观 | 是 | 二分类 清晰 vs 不清/模糊 |
| exudation | 183 | 2 | 90 | 0.5082 | 是 | 静态图像外观 | 是 | 二分类 无 vs +/++/+++ |
| blood_color | 181 | 2 | 81 | 0.5525 | 是 | 静态图像外观 | 是 | 二分类 浅红/淡红 vs 暗红/暗紫 |
| subpapillary_venous_plexus | 184 | 2 | 79 | 0.5707 | 是 | 静态图像外观 | 是 | 二分类 不见 vs 可见 |
| papilla | 185 | 3 | 42 | 0.4162 | 是 | 静态图像形态 | 是 | 三分类 波纹状/浅波纹状/平坦 |
| rbc_aggregation | 181 | 2 | 31 | 0.8287 | 是 | 静态外观，但医生按流动情境分级 | 是 | 二分类 无 vs 轻/中/重 |
| malformation_ratio | 162 | 2 | 70 | 0.5679 | 是 | 静态形态，视野内血管比例 | 是 | 二分类 <=10% vs >10% |
| capillary_count | 182 | 3 | 15 | 0.6868 | **否** | 单位长度条数，需标定与分母定义 | 部分 | **标签本身已是印出的分档** >=7 / 5--6 / <=4 |
| afferent_diameter | 165 | 3 | 34 | 0.4545 | **否** | 微米，需设备标定 | 部分 | 按报告印出的 `正常值 [9-13]` 分三档 |
| efferent_diameter | 164 | 3 | 23 | 0.6037 | **否** | 微米，需设备标定 | 部分 | 按 `[11-17]` 分三档 |
| apex_diameter | 169 | 3 | 25 | 0.4260 | **否** | 微米，需设备标定 | 部分 | 按 `[12-18]` 分三档 |
| loop_length | 176 | 3 | 32 | 0.5057 | **否** | 微米，需设备标定 | 部分 | 按 `[150-250]` 分三档 |
| crossing_ratio | 177 | 2 | 65 | 0.6328 | **否** | 静态形态，视野内血管比例 | 是 | 二分类 <=30% vs >30% |
| flow_state | 179 | 2 | 41 | 0.7709 | **否** | **时间基**（流态等级） | 否 | 二分类 线流/线粒流 vs 粒线流及更差 |
| microthrombus | 182 | 2 | 74 | 0.5934 | **否** | **时间基**（个/min） | 否 | 二分类 无 vs >=1 |

**结论：15 个字段全部有标签、全部有一个可以评估的目标，没有任何字段因「无标签」而不可评估。** 8 个字段在首轮**完全未测试**（上表「否」），这是范围缺口，不是它们没有信号。

### 4.2 必须写明的限制（按字段）

- **四个测量字段**（afferent/efferent/apex diameter、loop_length）：`device_calibration_status.json` 为 `UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION`，**绝对微米输出仍然禁止**。只做报告自己印出的三档。另外标签本身分别只有 24 / 25 / 38 / 135 个不同数值，前三个实际上是序数而非连续量。分档定义取报告 `正常值` 列，**两端都含**，不是我们发明的阈值。
- **capillary_count**：标签在文件里就是分档字符串（`>=7` / `5--6` / `3--4` / `<1`），所以**不需要预测任何 条/mm 数值**；真要输出每毫米条数仍然需要分母定义和设备标定，这条禁令不变。最小类只有 15 例，per-class CI 会很宽。
- **flow_state 与 microthrombus**：观测单位是时间基（流态等级、个/min），**单张静态图不携带这个单位**。本轮仍然评估它们，因为不评估等于把它们判成已关闭，这正是要避免的。但任何出现的信号只能读作「与之相关的静态外观」，**不得称为观察到了流速或血栓事件**。
- **rbc_aggregation**：众数占比 0.8287，是全部字段里最不平衡的；首轮四臂全部塌成单类。最小类 31 例。
- **efferent_diameter / apex_diameter / capillary_count**：最小类 23 / 25 / 15，低于 30，per-class recall 的置信区间会宽到难以据此下结论；这必须在报告时写出来，不能只报均值。

### 4.3 排除字段（按既定规则，不作为本轮成功标准）

| 字段 | 排除依据 | 标签文件实际情况 |
|---|---|---|
| vasomotion | 固定 `not_modelled`，需时间基 | 见 `field_matrix.json` |
| wbc_count | 固定 `not_modelled` | 同上 |
| sweat_duct | 固定 `not_modelled`，需完整视野 | 同上 |
| hemorrhage | 固定 `not_modelled`，需完整视野 | 同上 |
| flow_speed_um_s | 空列 | 同上 |
| output_input_ratio | 派生量，不单独建模 | 同上 |

### 4.4 补齐计划（待用户确认后执行，尚未运行）

对 15 个字段中每一个，跑同一套配置、同一病例折：

- 臂：`anchor_dinov2b_deployed`（ViT-B，部署几何）、`dinov2l_deployed_geometry`（ViT-L，部署几何）、`biomedclip_medical`（已有特征，注意分辨率混淆）；
- 读出与超参：**统一用部署固定配置**（5 池化、C 固定 0.03、PCA 64、seed 20260917），不做折内选择——执行记录 3 已显示折内选 C 会造出不稳健的增益；
- 每字段报告：accuracy、delta（对单一众数答案）、BA、AUROC（二分类）、每类召回、95% CI、按档案分层；
- 每字段标注：是否静态可观测、以及是否因标签/标定/时间/完整视野不足而不能交付。

纪律：不按字段事后挑最好模型（三个臂对全部 15 字段跑同一配置）；不把未测试字段写成无信号；某字段若确实无法合理评估，写明原因和证据。

---

## 执行记录 5：15 字段 × 3 臂补齐（2026-09-22，已执行）

复现命令（全部本地 GPU，服务器无卡模式未参与）：

```bash
PYTHONIOENCODING=utf-8 python scripts/probe_medical_encoder_access.py
PYTHONIOENCODING=utf-8 python scripts/run_field_matrix_three_arms.py
PYTHONIOENCODING=utf-8 python scripts/build_field_verdict_and_attribution.py
```

`locked_cases_seen = 0`（脚本在加载前断言 47 个 locked 病例不在任何一臂的 index 内，
并断言开发集恰为 186 例；另断言 `rep*` 报告扫描件没有进入编码器输入）。
**本记录全部数字是 development OOF，不是上线能力。**

### 5.1 配置（三臂完全相同，无任何逐字段选择）

| 项 | 值 |
|---|---|
| 池化 | `mean, topk_mean, max, cls, std`（固定五池化） |
| C | 固定 0.03（**不做折内选 C**，不使用首轮 in-fold C selection 结果） |
| PCA | 64 |
| seed / bootstrap | 20260917 / 2000 |
| 聚合 | 池化概率均值；多分类为池化多数投票（与交付配置逐字一致） |
| 划分 | `development_fold`，病例级 OOF，三臂同折同评估代码 |
| 逐字段挑选 | 无：不挑 C、不挑模型、不挑池化、不挑阈值 |

多分类的 AUROC 取自池化概率矩阵均值的 macro one-vs-rest；分类指标仍来自交付配置
的多数投票（投票本身不产生分数，所以 AUROC 另取，但**没有改变判定标签的方式**）。

### 5.2 A 组：通用视觉容量对照 / B 组：医学预训练

| 组 | 臂 | 权重 | 输入 | patch | 维度 | 归一化 | 几何 | 状态 |
|---|---|---|---|---|---|---|---|---|
| A | `anchor_dinov2b_deployed` | DINOv2 ViT-B/14 LVD-142M | 518×686 | 14 | 768 | ImageNet | 直接 resize（部署 native） | 已运行 |
| A | `dinov2l_deployed_geometry` | DINOv2 ViT-L/14 LVD-142M | 518×686 | 14 | 1024 | ImageNet | 同上 | 已运行 |
| B | `biomedclip_medical` | BiomedCLIP PMC-15M ViT-B/16 | 224×224 | 16 | 768 | CLIP | 保比 resize + 中性 pad | 已运行（读出用 CLS 与四个 patch 池化，**未使用文本零样本**） |
| B | RETFound_dinov2_meh | — | — | — | — | — | — | **blocked_access** |
| B | RETFound_dinov2_shanghai | — | — | — | — | — | — | **blocked_access** |
| B | RETFound_mae_natureCFP | — | — | — | — | — | — | **blocked_access** |
| B | MedSigLIP-448 | — | — | — | — | — | — | **blocked_access** |

### 5.3 医学候选访问状态（带对照，证据见 `medical_candidate_access.json`）

四个候选全部判定 `blocked_access`，判据不是「下载失败」，而是三项对照：

1. **公共仓对照**：同一客户端、同一端点下 `timm/vit_base_patch14_dinov2.lvd142m`
   与 `vit_large` 均返回 **200**。所以失败不是传输层问题。
2. **带/不带令牌对照**：在 `huggingface.co` 唯一一次可达的窗口里，四个 gated 仓
   **带令牌 403、不带令牌 401** —— 凭据被接受、授权被拒绝。该端点其余时间（含公共仓
   对照）连接超时，所以这一窗口是最干净的一次读数。镜像 `hf-mirror.com` 带/不带令牌
   都是 403，因为镜像不转发凭据。
3. **主机自己给出的理由**（403 响应体原文）：
   `Access to model google/medsiglip-448 is restricted and you are not in the authorized list.`
   四个仓文本一致，只有仓名不同。

结论只能写成：**医学预训练候选因访问受阻，尚未验证**。

- **不得写「医学视觉预训练无效」**：RETFound 与 MedSigLIP 一次都没有跑过，它们是
  `blocked_access`，不是 `rejected`；`rejected` 只保留给已经合法跑完公平实验并且输了的候选。
- **不得写「医学视觉预训练已验证」**：唯一跑起来的 BiomedCLIP 在分辨率（224 vs 518×686）、
  patch（16 vs 14）、编码器体系与预训练目标、归一化、几何处理五项上同时与 anchor 不同，
  它既不能证明也不能否证医学预训练。
- 本轮**没有代用户接受任何许可证，也没有绕过 Hugging Face gate**：探针只读 HTTP 状态码，
  没有取过任何权重文件。要继续这条路，需要用户本人在
  `huggingface.co/YukunZhou/RETFound_dinov2_meh`、`/RETFound_dinov2_shanghai`、
  `/RETFound_mae_natureCFP`、`/google/medsiglip-448` 四个页面各自申请并接受条款
  （MedSigLIP 走 HAI-DEF），此外 `huggingface.co` 直连在本机不稳定，取权重时可能需要
  镜像或代理配合。

### 5.4 主结果：15 字段 × anchor（DINOv2-B，部署配置）

尺子是 **accuracy − 单一常数答案基线（众数）**，即 delta；BA 并列报告，**不替代 delta**。
多分类 AUROC 为 macro one-vs-rest。「档案 BA」按 recovered_archive1/2/3 顺序。

| 字段 | n | k | 类别分布 | 众数基线 | accuracy | **delta [95% CI]** | BA | AUROC | 每类召回 | 档案 BA |
|---|---|---|---|---|---|---|---|---|---|---|
| clarity | 185 | 2 | 91/94 | 0.508 | 0.838 | **+0.330** [+0.238,+0.416] | 0.838 | 0.890 | 0.82/0.85 | 0.833/0.805/0.908 |
| exudation | 183 | 2 | 93/90 | 0.508 | 0.749 | **+0.240** [+0.137,+0.344] | 0.749 | 0.842 | 0.71/0.79 | 0.720/0.754/0.781 |
| subpapillary_venous_plexus | 184 | 2 | 79/105 | 0.571 | 0.837 | **+0.266** [+0.179,+0.353] | 0.832 | 0.897 | 0.80/0.87 | 0.842/0.796/0.859 |
| blood_color | 181 | 2 | 100/81 | 0.552 | 0.685 | **+0.133** [+0.028,+0.238] | 0.690 | 0.789 | 0.64/0.74 | 0.720/0.723/0.604 |
| loop_length | 176 | 3 | 32/55/89 | 0.506 | 0.608 | **+0.102** [+0.028,+0.176] | 0.510 | 0.712 | 0.25/0.44/0.84 | 0.464/0.523/0.523 |
| malformation_ratio | 162 | 2 | 92/70 | 0.568 | 0.648 | **+0.080** [-0.012,+0.173] | 0.630 | 0.707 | 0.76/0.50 | 0.650/0.655/0.570 |
| microthrombus | 182 | 2 | 108/74 | 0.593 | 0.670 | **+0.077** [-0.017,+0.159] | 0.654 | 0.761 | 0.74/0.57 | 0.703/0.695/0.536 |
| papilla | 185 | 3 | 66/77/42 | 0.416 | 0.443 | **+0.027** [-0.065,+0.119] | 0.415 | 0.645 | 0.50/0.51/0.24 | 0.436/0.396/0.400 |
| efferent_diameter | 164 | 3 | 23/99/42 | 0.604 | 0.628 | **+0.024** [-0.030,+0.079] | 0.406 | 0.611 | 0.00/0.91/0.31 | 0.420/0.422/0.355 |
| apex_diameter | 169 | 3 | 25/72/72 | 0.426 | 0.450 | **+0.024** [-0.089,+0.130] | 0.352 | 0.625 | 0.00/0.43/0.62 | 0.312/0.362/0.379 |
| afferent_diameter | 165 | 3 | 56/75/34 | 0.455 | 0.467 | **+0.012** [-0.073,+0.103] | 0.411 | 0.619 | 0.38/0.65/0.21 | 0.337/0.416/0.493 |
| capillary_count | 182 | 3 | 125/42/15 | 0.687 | 0.698 | **+0.011** [-0.033,+0.050] | 0.452 | 0.687 | 0.95/0.07/0.33 | 0.369/0.514/0.481 |
| rbc_aggregation | 181 | 2 | 31/150 | 0.829 | 0.818 | **-0.011** [-0.028,+0.000] | 0.493 | 0.684 | 0.00/0.99 | 0.488/0.492/0.500 |
| crossing_ratio | 177 | 2 | 112/65 | 0.633 | 0.622 | **-0.011** [-0.068,+0.045] | 0.530 | 0.628 | 0.88/0.18 | 0.522/0.509/0.562 |
| flow_state | 179 | 2 | 41/138 | 0.771 | 0.760 | **-0.011** [-0.028,+0.000] | 0.493 | 0.509 | 0.00/0.99 | 0.500/0.491/0.487 |

档案样本量：archive1 50–58、archive2 69–77、archive3 41–50（逐字段见
`field_matrix_three_arms_by_archive.csv`）。

**没有一个字段的最小类小于 10**，所以没有字段因为少数类过小而无法给出 per-class recall；
但 capillary_count（15）、efferent_diameter（23）、apex_diameter（25）的最小类都在 25 及以下，
这三个字段的少数类召回是在极少正例上估的，区间会很宽，**不能只看均值**。

### 5.5 三臂对照：容量 与 医学预训练（配对，同病例同折，无重训）

配对量是 BA 增益（候选 − anchor）及其 bootstrap CI；「折向」是 5 折里增益为正/为负的折数。

| 字段 | 容量 L−B | CI | 折向 | 医学 Bio−B | CI | 折向 |
|---|---|---|---|---|---|---|
| clarity | −0.0165 | [−0.060,+0.025] | +1/−4 | **−0.0542** | [−0.103,**−0.006**] | +0/−5 |
| exudation | +0.0217 | [−0.024,+0.071] | +3/−1 | +0.0163 | [−0.033,+0.070] | +3/−2 |
| blood_color | +0.0215 | [−0.041,+0.083] | +3/−1 | +0.0377 | [−0.032,+0.106] | +4/−0 |
| subpapillary_venous_plexus | −0.0175 | [−0.068,+0.032] | +2/−3 | **−0.1235** | [−0.184,**−0.061**] | +0/−5 |
| papilla | +0.0014 | [−0.064,+0.068] | +1/−4 | −0.0512 | [−0.130,+0.028] | +1/−4 |
| rbc_aggregation | +0.0067 | [+0.000,+0.017] | +1/−0 | +0.0033 | [−0.007,+0.016] | +1/−1 |
| malformation_ratio | −0.0269 | [−0.083,+0.029] | +2/−2 | +0.0449 | [−0.021,+0.112] | +4/−0 |
| capillary_count | −0.0117 | [−0.073,+0.043] | +4/−1 | −0.0550 | [−0.141,+0.041] | +2/−3 |
| afferent_diameter | +0.0291 | [−0.033,+0.091] | +4/−1 | −0.0354 | [−0.110,+0.034] | +1/−4 |
| efferent_diameter | −0.0287 | [−0.081,+0.030] | +1/−4 | +0.0293 | [−0.016,+0.077] | +4/−1 |
| apex_diameter | **+0.0637** | [**+0.009**,+0.122] | +4/−1 | **+0.0915** | [**+0.024**,+0.162] | +4/−1 |
| loop_length | −0.0162 | [−0.066,+0.033] | +2/−2 | −0.0506 | [−0.118,+0.015] | +2/−3 |
| crossing_ratio | −0.0320 | [−0.096,+0.033] | +0/−3 | +0.0442 | [−0.015,+0.104] | +2/−2 |
| flow_state | +0.0000 | [−0.011,+0.011] | +1/−1 | **+0.0316** | [**+0.004**,+0.068] | +4/−0 |
| microthrombus | −0.0110 | [−0.062,+0.039] | +2/−1 | +0.0569 | [−0.002,+0.113] | +4/−1 |

按用户给的读法逐条落地：

- **容量（DINOv2-L）**：15 个字段里只有 **1 个**（apex_diameter +0.064）CI 排除 0，其余 14 个
  区间全部跨 0，且方向在折间来回。执行记录 3 的结论在全字段范围内被确认：**把通用编码器从
  B 放大到 L，在部署配置下几乎什么都不改变**。这不是「只帮了部分字段的容量效应」，而是
  连部分都很勉强——唯一显著的那个字段本身 BA 只有 0.416，仍远在可交付之外。
- **医学预训练（BiomedCLIP）**：4 个字段 CI 排除 0，其中 **2 正 2 负**：apex_diameter +0.092、
  flow_state +0.032 为正；clarity −0.054、subpapillary_venous_plexus −0.124 为负，而这两个
  恰好是本项目最强的两个字段。按用户的读法，「只有 BiomedCLIP 只帮到部分字段，才算存在局部
  医学预训练增益」——形式上确实是部分字段，**但它同时明确损害了最强字段，而且它与 anchor 在
  分辨率/patch/体系/归一化/几何五项上同时不同，所以这 2 正 2 负都不能归因到「医学预训练」**。
  BiomedCLIP 在 224×224、patch16 下工作，它的损失更可能是分辨率损失：clarity 与 SVP 都依赖
  细纹理与排列，降到 224 正是最该掉的两个。
- **测量三分类是否解决了问题**：没有。四个测量字段的 delta 分别 +0.012 / +0.024 / +0.024 /
  +0.102，其中三个 CI 跨 0；BA 分别 0.411 / 0.406 / 0.352 / 0.510，**全部低于三分类的
  chance 0.333 之上一点点甚至接近**（0.352 只比 chance 高 0.019）。**改成三档没有解决测量问题。**
- **flow_state / microthrombus**：见 5.7，两者的任何增益只能读作静态外观相关。
- **malformation_ratio 与 papilla 是否都不稳**：malformation_ratio delta +0.080 CI 含 0、
  档案 3 掉到 0.570；papilla delta +0.027 CI 含 0、BA 0.415 低于三分类可用水平、平坦类召回 0.24。
  **两者都不稳**。按用户给的读法，这意味着**眼底迁移与医学编码器这条路应当停下**——而且 5.3
  显示真正的眼底模型（RETFound）根本没能跑，所以停下的理由是「这条路当前既无信号也无入口」，
  不是「已证明无效」。

### 5.6 四个测量字段：原始数值分布、分档与跨档错误

**三分类结果不是微米测量能力。** 分档取报告自己印出的 `正常值` 区间，绝对微米输出仍然禁止
（`device_calibration_status.json` = `UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION`）。

| 字段 | 印出正常区间 | n | 不同取值 | min/中位/max | 三档大小(低/正常/高) | 近阈值占比 | 严重跨档错误(anchor/L/Bio) |
|---|---|---|---|---|---|---|---|
| afferent_diameter | 9–13 | 165 | 24 | 4 / 10 / 325 | 56/75/34 | 0.182 | 0.036 / 0.024 / 0.030 |
| efferent_diameter | 11–17 | 164 | 25 | 7 / 14 / 89 | 23/99/42 | 0.195 | 0.006 / 0.006 / 0.006 |
| apex_diameter | 12–18 | 169 | 38 | 6 / 18 / 113 | 25/72/72 | 0.106 | 0.035 / 0.047 / 0.047 |
| loop_length | 150–250 | 176 | 135 | 38 / 255.5 / 648 | 32/55/89 | 0.102 | 0.057 / 0.057 / 0.051 |

「近阈值」定义为落在任一印出边界 ±10% 档宽内（即 ±0.4 / ±0.6 / ±0.6 / ±10 μm）。
「严重跨档错误」为预测与真值相差两档（低判成高或高判成低）。

三点必须同时读：

1. **严重跨档错误率很低（0.6%–5.7%）看着好，但这是分档本身给的**，不是模型的能力：
   三档里中间档占比最大（45%–60%），只要模型偏向中间档，两档之差的错误就自然稀少。
   同一批预测的 delta 只有 +0.012 ~ +0.102，BA 0.35–0.51，**低错误率和低能力并不矛盾**。
2. **标签的数值网格比档宽还粗**：afferent/efferent/apex 的标签基本落在整数微米上，
   中位步长 1 μm，而档宽只有 4 / 6 / 6 μm——**一格读数误差就是 25% / 17% / 17% 的档宽**。
   afferent 有 18.2%、efferent 有 19.5% 的样本落在阈值 ±10% 档宽内，这些样本的档归属
   由不到一格的读数差决定。这与 `nailfold-label-granularity-ceiling` 的结论一致：
   瓶颈在标签粒度，换编码器动不了它。
3. **分布有极端值**（afferent 最大 325、efferent 89、apex 113、loop_length 648），
   远超正常区间上界，说明标签里混有量纲或读数异常；本轮没有剔除它们（剔除就是一次
   按结果做的选择），但它们全部落进「高」档，会让高档显得更容易。

**capillary_count**：标签在文件里本来就是印出的分档字符串，本轮直接用该分档，
**没有输出任何 条/mm 数值**；严重跨档错误 0.077 / 0.055 / 0.055。但 delta 只有 +0.011
[−0.033,+0.050]，中间档召回 0.07、最小档 15 例，**报告结果但不判定可交付**。

### 5.7 flow_state 与 microthrombus：静态外观相关，不是流态/事件观测

**标题即约束：这一节的任何数字都是静态外观相关，不是对流速或血栓事件的观测。**
两个字段的观测单位是时间基（流态等级由运动判读；microthrombus 印为 个/min），
单张静态图不携带这个单位。

- **flow_state**：类别分布 41（线流/线粒流，正常）/ 138（粒线流及更差，异常），
  **众数是异常类**，基线 0.771。anchor delta −0.011 [−0.028,+0.000]，BA 0.493，
  AUROC 0.509，**正常类召回 0.00**——在 anchor 与 L 上把所有病例都判成异常，完全塌到众数。
  BiomedCLIP 的 BA 增益 +0.032 CI 排除 0，accuracy 0.760→0.782，但正常类召回只有 0.0488
  （41 例正常里认出 2 例），AUROC 反而降到 0.4897。
  **即使 CI 排除 0，这也不能说明静态图能检出流态事件**：它认出的是「大多数病例都异常」。
- **microthrombus**：anchor delta +0.077 [−0.017,+0.159]，BA 0.654，AUROC 0.761，
  两类召回 0.74/0.57；**CI 含 0**，档案 3 的 BA 掉到 0.536（archive1/2 为 0.703/0.695）。
  BiomedCLIP 在这个字段上 delta +0.132 [+0.044,+0.220]，**在 delta 尺子上 CI 排除 0**
  （它的配对 BA 增益 +0.057 CI [−0.002,+0.113] 则含 0，两把尺子在这里不一致；
  它是 §5.8 列出的四个「只有医学臂越过基线」字段之一）。
  **但这仍然只是静态外观相关**：不得据此声称可以从静态图数出每分钟血栓个数，
  而且它是单臂单字段结果、与 5.5 的负向字段同出一臂、且该臂五项混淆未解，
  按 §5.8 的门槛不构成采用理由。另外 `nailfold-field-variance-ranking` 已记录用户指出
  「数据里没有血栓」，**该字段的临床含义本身需要用户复核**，在复核前它的任何分数都不应进入产品讨论。

### 5.8 一个必须写出来的反向发现：delta 尺子下 BiomedCLIP 多出四个字段

5.5 的配对 BA 对照给出「2 正 2 负」，但换成本项目的主尺子 delta，图景不同。
下面四个字段是 **anchor 的 delta CI 含 0、而 BiomedCLIP 的 delta CI 排除 0**：

| 字段 | anchor delta [CI] | BiomedCLIP delta [CI] |
|---|---|---|
| malformation_ratio | +0.080 [−0.012,+0.173] | **+0.123 [+0.037,+0.210]** |
| microthrombus | +0.077 [−0.017,+0.159] | **+0.132 [+0.044,+0.220]** |
| apex_diameter | +0.024 [−0.089,+0.130] | **+0.118 [+0.006,+0.219]** |
| efferent_diameter | +0.024 [−0.031,+0.079] | **+0.061 [+0.006,+0.116]** |

三点必须一起说，否则这张表会被读成「医学预训练有效」：

1. **这不是配对显著。** 上表是各臂各自对基线的 CI；两个区间都含真值时，「一个排除 0、
   另一个含 0」本身不构成两臂有差别。这四个字段的**配对** BA 增益 CI 只有 apex_diameter
   排除 0（见 5.5），malformation_ratio +0.045、microthrombus +0.057、efferent +0.029
   的配对 CI 都含 0。
2. **同一臂在最强的两个字段上显著变差**（clarity −0.054、SVP −0.124，配对 CI 均排除 0、
   5/5 折同向）。一个臂同时「弱字段涨、强字段跌」，更像读出特性改变——224 低分辨率丢细纹理、
   patch16 网格不同、CLIP 归一化、pad 几何——而不是获得了医学知识。
3. **不能按字段挑臂。** 若只取上表四行就说「microthrombus 用 BiomedCLIP」，那正是用户禁止的
   逐字段挑模型；`nailfold-perfield-selection-refuted` 已实测逐字段选模型是负收益。
   **本轮不采用任何新臂。**

这四行仍然记录在案，因为把它们藏起来同样是修饰结果。它们的正确身份是一个
**待验证假设：低分辨率 + 不同 patch 网格对弱字段可能有利**。检验它需要一个只改分辨率/patch
的通用编码器对照（把 DINOv2-B 降到 224 跑一遍），**不需要也不能用医学预训练来解释**。
该对照未运行，不在本轮授权范围内。

### 5.9 字段判定（15 个字段，仅按 anchor 的固定配置，development OOF）

判定只有三档，**没有「已关闭」这一档**：

- `usable_on_development`：delta CI 排除 0、没有类别被放弃（最差每类召回 ≥ 0.30）、
  三个档案的 BA 全部高于该字段的 chance（1/k）。
- `signal_only`：有可读的数，但不满足上面任一条——**不是无信号，也不是已关闭**。
- `no_signal_at_this_n`：delta CI 整体在 0 及以下。**「at this n」是定语**：
  它说的是在 186 例、当前标签粒度下测不出来，不是这个字段永远没有信号。

| 判定 | 字段 | 关键限制 |
|---|---|---|
| **usable_on_development（4）** | clarity +0.330 | 三档案 BA 0.833/0.805/0.908 |
| | subpapillary_venous_plexus +0.266 | 0.842/0.796/0.859 |
| | exudation +0.240 | 0.720/0.754/0.781 |
| | blood_color +0.133 | 档案 3 掉到 0.604；CI 下界仅 +0.028 |
| **signal_only（9）** | loop_length +0.102 | CI 排除 0，**但** BA 0.510、最低档召回 0.25、需标定禁令 |
| | malformation_ratio +0.080 | CI 含 0；档案 3 0.570 |
| | microthrombus +0.077 | CI 含 0；时间基单位；档案 3 0.536 |
| | papilla +0.027 | CI 含 0；BA 0.415；平坦类召回 0.24 |
| | efferent_diameter +0.024 | CI 含 0；低档召回 0.00；最小类 23 |
| | apex_diameter +0.024 | CI 含 0；低档召回 0.00；最小类 25 |
| | afferent_diameter +0.012 | CI 含 0；BA 0.411 |
| | capillary_count +0.011 | CI 含 0；中档召回 0.07；最小类 15 |
| | crossing_ratio −0.011 | CI 含 0；少数类召回 0.18 |
| **no_signal_at_this_n（2）** | rbc_aggregation −0.011 | 众数 0.829，塌到单类，少数类召回 0.00 |
| | flow_state −0.011 | 众数 0.771，塌到单类；时间基单位 |

与既有账本的关系：**clarity 与 exudation 仍然是唯一两个反复站得住的字段**
（`nailfold-all-fields-audit`、`nailfold-leave-one-archive-out`）。本轮 SVP 与 blood_color
在 development 上同样越过基线，但 `nailfold-leave-one-archive-out` 已实测这两个字段是
**档案依赖的**，三向留一档案外测只有 clarity + exudation 全过。**本轮没有重跑留一档案外测**，
所以 5.9 的「4 个」不推翻那条记录，两者口径不同：这里是三档案分层，那里是留一档案外测。
`nailfold-locked-delivered-result` 记录的「locked 严格尺子下 0 个字段通过」也未被本轮触及
（本轮没有读 locked）。

### 5.10 因子归因（不重训，只重配已存的 OOF 预测）

| 因子 | 构造 | 结果 | 能不能据此下结论 |
|---|---|---|---|
| `capacity_only` | ViT-L vs ViT-B，同为通用预训练、同部署几何、同 loader | 15 字段里 1 个配对 CI 排除 0（apex_diameter +0.064），其余 14 个跨 0、折向摇摆 | 能。**放大通用编码器在部署配置下基本无效**，且唯一显著的字段 BA 仅 0.416 |
| `medical_pretraining_only` | BiomedCLIP vs ViT-B | 4 个配对 CI 排除 0：2 正（apex +0.092、flow_state +0.032）、2 负（clarity −0.054、SVP −0.124） | **不能。** 该对照同时混入分辨率 224 vs 518×686、patch 16 vs 14、编码器体系与预训练目标、CLIP vs ImageNet 归一化、pad vs 直接 resize 五项 |
| `loader_geometry_only` | 首轮 loader control | **不可用**：该臂抽出的特征与 anchor 逐字节相同（`np.array_equal` = True），不携带信息 | 不能。几何差异现在整体落在 BiomedCLIP 对照里，与其他四项无法分离 |
| `target_encoding` | 各字段众数基线本身 | rbc_aggregation 0.829、flow_state 0.771 两个字段的 accuracy 主要由重编码给出 | 能。**这两个字段的高 accuracy 与图像无关**，delta 是唯一可读的数 |
| `static_input_mismatch` | 观测单位 | 时间基：flow_state、microthrombus。需标定：四个直径/长度 + capillary_count | 能。这 7 个字段的单位不是静态单帧能承载的，**与模型无关** |

一句话归因：**本轮观察到的全部差异，没有一项可以归给「医学预训练」**——能归因的只有
「放大容量几乎无效」和「重编码本身贡献了部分 accuracy」，剩下的都压在一个五项混淆的臂上。

### 5.11 停止点与纪律确认

按授权第 7 条，**在此停止**：不做 LoRA、不做 adapter、不做视频模型、不做模型选择。

| 纪律 | 本轮执行情况 |
|---|---|
| 三臂同一交付固定配置 | 是：5 池化、C 固定 0.03、PCA 64、seed 20260917 |
| 不按字段挑 C/模型/池化/阈值 | 是：45 行结果出自同一配置，未采用任何字段的最优臂 |
| 不使用首轮 in-fold C selection 结果 | 是：本轮不调用 `eval_medical_encoders.py` 的产物 |
| 15 个字段全部产生结果 | 是：15/15，无一字段被跳过或写成「已关闭」 |
| 三分类不称为微米测量能力 | 是：5.6 明确标注，且未输出任何微米值 |
| capillary_count 不输出 条/mm | 是：直接用报告印出的分档 |
| flow_state/microthrombus 标题为静态外观相关 | 是：5.7 标题即约束 |
| locked-47 | `locked_cases_seen = 0`，本轮未读取；病例级断言 |
| 医学候选 | 4 个 `blocked_access`，无一个 `rejected`；未代接受许可证、未绕 gate |

产物（全部新写，未改动任何既有 artifacts）：

```
artifacts/experiments/medical_encoder_transfer_20260921/
  medical_candidate_access.json              # 访问状态 + 对照 + 主机原文
  field_matrix_three_arms.json               # 45 行完整指标（15 字段 × 3 臂）
  field_matrix_three_arms.csv                # 扁平表
  field_matrix_three_arms_by_archive.csv     # 档案分层
  field_matrix_three_arms_predictions.csv    # 病例级 OOF 预测，可复核
  field_verdict_and_attribution.json         # 判定 + 因子归因
  field_verdict.csv
scripts/
  probe_medical_encoder_access.py
  run_field_matrix_three_arms.py
  build_field_verdict_and_attribution.py
```

局限（必须与上面的数一起被引用）：

1. **全部是 development OOF，不是上线能力。** locked-47 预算已超支（7 次消耗，3 次在其上做过
   模型选择），任何上线材料必须披露这一点。
2. **本轮没有做留一档案外测。** 5.9 的 4 个 `usable_on_development` 是三档案分层口径；
   `nailfold-leave-one-archive-out` 的三向外测口径下只有 clarity + exudation 全过。
3. **医学预训练仍未验证。** RETFound × 3 与 MedSigLIP 全部 `blocked_access`，需要用户本人
   接受条款；BiomedCLIP 不能代表它们。
4. **测量字段的瓶颈是标签粒度，不是编码器**：标签网格 1 μm 对 4–6 μm 的档宽，
   18%–20% 的样本落在阈值 ±10% 档宽内。
5. **microthrombus 的临床含义待用户复核**（用户曾指出数据里没有血栓）。
6. 上市阻塞项未被本轮触动：独立确诊患者 0 例、外部验证 0 项、5% 患病率下 PPV 0.12–0.15。
