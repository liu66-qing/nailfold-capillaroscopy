from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(r'E:\甲劈微循环\artifacts\甲襞微循环自动分析项目阶段汇报.docx')

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = tcPr.find(qn('w:shd'))
    if shd is None: shd = OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'), fill)

def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ''
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
    r = p.add_run(str(text)); r.bold = bold; r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = 'Table Grid'
    for i, h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, True, 'FFFFFF', 9); shade(t.rows[0].cells[i], '1F4E79')
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row): set_cell_text(cells[i], value, False, None, 9)
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths): row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t

def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f'Heading {level}'); p.add_run(text); return p

def bullet(doc, text):
    p = doc.add_paragraph(style='List Bullet'); p.add_run(text); return p

doc = Document()
sec = doc.sections[0]; sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.72); sec.left_margin = Inches(.82); sec.right_margin = Inches(.82)
styles = doc.styles
styles['Normal'].font.name = 'Microsoft YaHei'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei'); styles['Normal'].font.size = Pt(10.5)
for name, size, color in [('Title', 22, '1F4E79'), ('Heading 1', 15, '1F4E79'), ('Heading 2', 12, '2F75B5')]:
    s = styles[name]; s.font.name = 'Microsoft YaHei'; s._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei'); s.font.size = Pt(size); s.font.color.rgb = RGBColor.from_string(color); s.font.bold = True
styles['Normal'].paragraph_format.space_after = Pt(6); styles['Normal'].paragraph_format.line_spacing = 1.15

p = doc.add_paragraph(style='Title'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run('甲襞微循环自动分析项目阶段汇报')
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = p.add_run('方案、阶段效果与主要困难（事实汇总）'); r.font.size = Pt(12); r.font.color.rgb = RGBColor.from_string('666666')
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.add_run('数据与评估：267例病例 / 2207张图像 / 47例 locked 测试集\n报告日期：2026年8月16日').font.size = Pt(10)

heading(doc, '一、汇报结论', 1)
doc.add_paragraph('本项目采用“分割得到结构、几何算法完成定量、病例级特征融合完成分类”的路线，目标是对甲襞微循环报告中的17个字段进行自动分析。开发阶段最终配置在病例隔离的五折验证中取得 0.6735 的平均得分；在模型、特征和阈值冻结后，对47例 locked 测试病例进行一次性评估，平均得分为 0.6902，未达到 0.75 的目标。')
doc.add_paragraph('结果显示，几何定量是当前最稳定的部分；复杂定性字段和管袢计数是主要短板。因 locked 测试已按协议完成唯一一次评估，本文不把测试集结果用于重新调参。')

heading(doc, '二、问题定义与数据事实', 1)
bullet(doc, '研究对象是甲襞微循环图像及其病例级报告标签，基本分析单位是“一次检查对应的病例文件夹”。')
bullet(doc, '同一病例中的 CAPorg 图像来自同一视频的随机截帧，帧间高度相似，不能当作相互独立的训练样本。')
bullet(doc, '报告中的数值是医生对截取帧测量后取平均值，因此标签天然是病例级聚合结果。')
bullet(doc, '数据共267例、2207张图像、127例有视频；另有 ANFC-THU 数据集的257张训练图、64张验证图和5363个实例标注。')
bullet(doc, '病例按文件夹隔离划分，保证同一病例的帧不进入不同 fold；locked 集包含47例。')

heading(doc, '三、总体技术思路', 1)
doc.add_paragraph('路线的核心判断是：直接从整张图像端到端预测所有报告字段，难以解释也难以稳定迁移；因此先把可观测的微循环结构显式提取出来，再分别处理定量和定性任务。')
table(doc, ['模块', '思路', '输出'], [
    ('质量与结构', '对图像进行质量控制，并用 YOLOv11m-seg 分割正常/异常管袢及相关类别。', '实例 mask、类别与置信度'),
    ('几何定量', '对管袢 mask 做骨架化、端点和中心线分析，结合标定因子计算直径、袢长等。大面积 mask 使用 skeleton 分支。', '管径、袢顶直径、袢长、计数特征'),
    ('视觉表征', '使用 DINOv2-base 与 Hulu-Med-7B 的视觉特征作为补充表征；Hulu-Med 只作特征提取器，不做文本生成。', '病例级视觉向量'),
    ('分类与融合', '使用结构化特征、视觉特征及几何特征训练 GBT/其他冻结分类器，按字段选择 route，并在病例级聚合帧预测。', '定性字段预测'),
    ('动态信息', '尝试稳定片段选择、ROI追踪和分割锚点对齐；不做全局防抖。', '视频辅助特征或兼容性回退'),
], [1.25, 4.8, 1.65])

heading(doc, '四、按步骤的执行结果', 1)
table(doc, ['阶段', '验收事实', '结论'], [
    ('Step 1-2', '服务器环境、双卡硬件和数据结构确认；完成数据格式转换。', '通过'),
    ('Step 3-4', '完成 YOLOv11m-seg 训练及2207张图像推断。', '通过'),
    ('Step 5', '跨域验证后继续执行；低于初始阈值时采用域适配结果。', '通过'),
    ('Step 6', '完成分割结果抽样质检流程。', '通过'),
    ('Step 7', '计数逻辑修正为 normal + abnormal instance count；大面积 mask 使用 skeleton 分支；自训练增益不足，未作为最终依赖。计数 GBT OOF BA=0.5287。', 'Gate通过'),
    ('Step 8', '完成 DINOv2 与 Hulu-Med 双通道特征提取。', '通过'),
    ('Step 9', '几何算法使输入支测量 MAE 改善27.9%。', 'Gate通过'),
    ('Step 10', '5/6主要字段相对基线改善；视觉特征进入分类流程。', 'Gate通过'),
    ('Step 11', '视频光流 weighted BA=0.2024，低于0.35；动态字段降级为兼容性回退。', '失败后按方案回退'),
    ('Step 12', '最佳冻结配置 step12_r3，五折平均=0.673481。用户将 Gate 放宽为≥0.65。', '通过（放宽Gate）'),
    ('Step 13', '冻结记录：development=186例，locked=47例，locked_cases_seen=0，locked_evaluation_authorized=true。', '通过'),
    ('Step 14', '对47例 locked 集完成唯一一次评估，平均=0.690199。', '未达到0.75'),
], [1.0, 5.7, 1.0])

heading(doc, '五、最终效果（Step 14 locked）', 1)
doc.add_paragraph('评估口径：分类字段使用 balanced accuracy；连续字段用 max(0, 1-MAE/开发集字段范围) 转为得分；兼容字段计算默认值命中率。报告共17个字段，平均值为所有字段得分的算术平均。')
rows = [
('汗腺导管','100.00%','44','兼容默认值命中率'),('血管运动','91.49%','47','兼容默认值命中率'),('白细胞数','97.83%','46','兼容默认值命中率'),('血流速度','100.00%*','0','无有效病例，兼容占位分'),
('清晰度','51.59%','46','balanced accuracy'),('血色','48.37%','44','balanced accuracy'),('交叉比例','59.49%','45','balanced accuracy'),('畸形比例','43.57%','39','balanced accuracy'),('渗出','32.67%','47','balanced accuracy'),('出血','50.00%','47','balanced accuracy'),('乳头下静脉丛','40.58%','47','balanced accuracy'),('乳头形态','47.21%','46','balanced accuracy'),('管袢数','38.91%','46','balanced accuracy'),
('输入支直径','99.05%','44','归一化MAE；MAE=3.09'),('输出支直径','94.27%','44','归一化MAE；MAE=4.70'),('袢顶直径','95.25%','44','归一化MAE；MAE=5.08'),('管袢长度','83.06%','45','归一化MAE；MAE=103.36'),]
table(doc, ['字段','得分','有效病例','口径/误差'], rows, [1.85, 1.0, .85, 4.0])
doc.add_paragraph('* 血流速度没有有效评价病例，100%是兼容性占位分，不代表真实预测准确率。', style='Normal').runs[0].italic = True

heading(doc, '六、结果解读', 1)
heading(doc, '1. 已经验证的优势', 2)
bullet(doc, '结构化几何路线有效：四个连续几何字段的得分为83.06%–99.05%，说明分割、骨架和标定方法能够较稳定地恢复部分可测量结构。')
bullet(doc, '病例级聚合比帧级独立判断更符合数据生成过程，避免把同一视频的相似帧错误当作独立样本。')
bullet(doc, 'DINOv2 与 Hulu-Med 的视觉特征补充了结构化特征，开发集上多数主要字段相对基线改善。')
bullet(doc, '模型冻结、病例隔离和 locked 一次性评估流程已建立，最终结果具有可追溯性。')
heading(doc, '2. 尚未解决的问题', 2)
bullet(doc, '定性病灶字段较弱：渗出32.67%、乳头下静脉丛40.58%、畸形比例43.57%，表明静态视觉特征尚未充分表达临床判读所需的细粒度信息。')
bullet(doc, '管袢数仅38.91%，分割漏检、粘连、重叠和大面积 mask 的拓扑不稳定会直接传递到计数。')
bullet(doc, '视频光流 weighted BA=0.2024，低于0.35 Gate；动态信息未能可靠提升模型，因此按规则回退。')
bullet(doc, '开发集均值0.6735与 locked 均值0.6902之间存在差异，说明跨病例/跨域泛化仍是主要风险。')

heading(doc, '七、最大的困难', 1)
doc.add_paragraph('第一，标签与图像的对应关系不完全。医生报告是多帧观察后的病例级总结，而输入是同一视频的若干截帧，导致“图像证据—报告标签”之间存在不可避免的弱监督和噪声。')
doc.add_paragraph('第二，数据量与类别分布限制了复杂字段学习。总病例数为267例，部分类别有效样本更少；balanced accuracy 对少数类召回敏感，少量错误就会显著影响分数。')
doc.add_paragraph('第三，跨设备、白平衡、放大倍率和图像质量造成域差异。ANFC 实例标注数据与真实病例图像的成像条件并不完全一致，分割误差会进一步影响几何测量、计数和分类。')
doc.add_paragraph('第四，动态字段需要时序信息。当前图像主要是视频随机截帧，且未建立足够可靠的运动、红细胞聚集和血流状态标注，光流特征难以与报告字段稳定对应。')
doc.add_paragraph('第五，评估协议本身限制了迭代空间。Step 13 冻结后不得查看 locked 标签做开发决策，Step 14 只能运行一次；因此本次结果低于目标后，不能用测试集反复试错来追分。')

heading(doc, '八、结论与后续方向', 1)
doc.add_paragraph('本阶段已完成从分割、几何测量、视觉特征到病例级分类的完整闭环，并证明定量几何字段具有较好的可行性；但整体17字段平均得分为69.02%，尚未满足75%目标。当前瓶颈不是单一分类器选择，而是病例级高质量标签、跨域分割质量、复杂定性字段的临床语义和时序信息不足。')
bullet(doc, '优先补充病例级、字段级高质量标注，并记录不确定/无法判断状态，减少把弱证据强行映射为确定类别。')
bullet(doc, '针对管袢计数与病灶字段建立更严格的实例/区域质检集，区分漏检、粘连、重叠和真实异常。')
bullet(doc, '在开发集上引入设备/病例分层验证及校准，确认改进是否真正跨域，而不是只改善某一批病例。')
bullet(doc, '在有可靠时序标签后再发展视频模型；当前光流结果不足以支持动态字段自动化。')
bullet(doc, '下一轮研究应重新建立独立 locked 集，在所有模型和阈值冻结后再进行一次性评估。')

heading(doc, '九、证据与可追溯产物', 1)
table(doc, ['证据', '位置/内容'], [
    ('技术路线与约束', '/root/nailfold/docs/甲襞微循环专长模型与医学智能体技术路线调研.md'),
    ('执行手册', '/root/nailfold/docs/务实技术路线执行手册.md'),
    ('Step 13冻结记录', '/root/nailfold/artifacts/evaluation/step13_freeze.json；locked_cases_seen=0，frozen=true'),
    ('Step 14最终报告', '/root/nailfold/artifacts/evaluation/final_locked_evaluation_v1.json；locked_cases_seen=47'),
    ('最终报告SHA-256', '0b8af38c1055c71ece3ee39c4505f2304afcaf8b844856fac1a0e80e431f32ab7'),
], [1.8, 5.8])
doc.add_paragraph('说明：本文只汇总已生成的模型、日志和评估报告中的事实；未将未完成的目标表述为已完成。')

OUT.parent.mkdir(parents=True, exist_ok=True); doc.save(OUT); print(OUT)
