import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
from skimage.morphology import skeletonize

ROOT = Path('/root/autodl-tmp/nailfold')
OUT = ROOT / 'artifacts' / 'report_visuals'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/root/autodl-tmp/nailfold/artifacts/report_visuals/fonts/simhei.ttf'
BOLD = '/root/autodl-tmp/nailfold/artifacts/report_visuals/fonts/simhei.ttf'

def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, n)

def numeric(value):
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace('μm', '').strip())
    except ValueError:
        return None

roles = pd.read_csv(ROOT / 'artifacts/manifest/locked_evaluation_v1.csv')
dev = set(roles.loc[roles.evaluation_role.eq('development'), 'exam_case_id'])
labels = pd.read_csv(ROOT / 'artifacts/labels/multisource_confidence_v3.csv').set_index('exam_case_id')
geo = pd.read_csv(ROOT / 'artifacts/geometry_exact/per_case_measurements.csv').set_index('case_id')
factors = json.load(open(ROOT / 'artifacts/geometry_exact/calibration_factors.json'))
scale = {key: float(np.median([item['um_per_pixel'] for item in values])) for key, values in factors.items()}
fields = [('afferent_diameter', 'afferent_px'), ('apex_diameter', 'apex_px'), ('loop_length', 'length_px')]
ranked = []
for case_id in sorted(dev & set(labels.index) & set(geo.index)):
    errors = []
    for field, pixel in fields:
        truth = numeric(labels.loc[case_id, field])
        pred = numeric(geo.loc[case_id, pixel])
        if truth is None or pred is None:
            errors = []
            break
        errors.append(abs(pred * scale[field] - truth) / (abs(truth) + 1))
    if errors:
        ranked.append((sum(errors), case_id))
chosen = [case_id for _, case_id in sorted(ranked)[:3]]

predictions = json.load(open(ROOT / 'artifacts/pseudo_labels/round0_multiclass_domain_adapted.json'))
by_case = defaultdict(list)
for image_path, item in predictions.items():
    if item['case_id'] in chosen:
        by_case[item['case_id']].append((image_path, item))
colors = {'normal': (40, 190, 80, 130), 'abnormal': (225, 55, 55, 150), 'hemo': (245, 195, 35, 170)}

def best_image(case_id):
    candidates = sorted(by_case[case_id], key=lambda pair: -len(pair[1].get('instances', [])))
    for relpath, item in candidates:
        path = ROOT / 'data' / relpath
        if path.exists():
            return path, item
    raise FileNotFoundError(case_id)

def draw_case(case_id, index):
    path, item = best_image(case_id)
    source = Image.open(path).convert('RGB')
    sw, sh = source.size
    source.thumbnail((455, 620), Image.Resampling.LANCZOS)
    w, h = source.size
    sx, sy = w / sw, h / sh
    canvas = Image.new('RGB', (1920, 760), '#f4f7fa')
    draw = ImageDraw.Draw(canvas)
    for x, title in [(20, '1  ORIGINAL FRAME'), (500, '2  INSTANCE MASKS'), (980, '3  SKELETON + MEASUREMENTS'), (1460, '4  CASE COMPARISON')]:
        draw.text((x, 24), title, font=font(23, True), fill='#17324d')
    draw.line((20, 57, 1900, 57), fill='#c9d6e2', width=2)
    canvas.paste(source, (20, 72))
    overlay = source.convert('RGBA')
    layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    mask = np.zeros((h, w), np.uint8)
    for inst in item.get('instances', []):
        name = inst.get('class_name', '').lower()
        if name not in colors or float(inst.get('confidence', 0)) < 0.12:
            continue
        points = [(int(x * sx), int(y * sy)) for x, y in inst.get('polygon', [])]
        if len(points) >= 3:
            ld.polygon(points, fill=colors[name], outline=colors[name][:3] + (230,))
            if name in ('normal', 'abnormal'):
                cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
    overlay = Image.alpha_composite(overlay, layer).convert('RGB')
    canvas.paste(overlay, (500, 72))
    skeleton_panel = source.copy()
    skeleton = skeletonize(mask > 0)
    array = np.asarray(skeleton_panel).copy()
    array[skeleton] = (0, 255, 255)
    skeleton_panel = Image.fromarray(array)
    sd = ImageDraw.Draw(skeleton_panel)
    ys, xs = np.where(skeleton)
    if len(xs):
        marks = [('afferent', (int(xs.min()), int(ys.min())), '#00d8ff'), ('apex', (int(xs.mean()), int(ys.mean())), '#ff55dd'), ('efferent', (int(xs.max()), int(ys.max())), '#ffb000')]
        for name, point, color in marks:
            sd.ellipse((point[0]-7, point[1]-7, point[0]+7, point[1]+7), fill=color)
            sd.text((point[0]+9, point[1]-14), name, font=font(14, True), fill=color)
    canvas.paste(skeleton_panel, (980, 72))
    draw.text((1480, 92), f'case: {case_id}', font=font(16, True), fill='#17324d')
    draw.text((1480, 128), 'MODEL', font=font(17, True), fill='#198754')
    draw.text((1700, 128), 'REPORT', font=font(17, True), fill='#b42318')
    panel_fields = [('Afferent diameter', 'afferent_diameter', 'afferent_px'), ('Apex diameter', 'apex_diameter', 'apex_px'), ('Loop length', 'loop_length', 'length_px')]
    y = 150
    for title, field, pixel in panel_fields:
        pixel_value = float(geo.loc[case_id, pixel])
        truth = numeric(labels.loc[case_id, field])
        predicted_um = pixel_value * scale[field]
        draw.text((1480, y), title.upper(), font=font(14, True), fill='#244e70')
        draw.text((1480, y+30), f'{predicted_um:.1f} um', font=font(16), fill='#198754')
        draw.text((1480, y+56), f'({pixel_value:.1f} px)', font=font(14), fill='#198754')
        draw.text((1700, y+30), f'{truth:.1f} um', font=font(16), fill='#b42318')
        y += 112
    draw.text((1480, 620), 'Green: normal   Red: abnormal', font=font(14), fill='#4a5568')
    draw.text((1480, 646), 'Yellow: hemo   Cyan: skeleton', font=font(14), fill='#4a5568')
    canvas.save(OUT / f'case_vis_{index}.png', dpi=(150, 150))

for i, case_id in enumerate(chosen, 1):
    draw_case(case_id, i)

# Architecture diagram.
W, H = 2400, 1200
image = Image.new('RGB', (W, H), '#f7fafc')
draw = ImageDraw.Draw(image)
draw.text((70, 35), '甲襞微循环智能分析技术架构', font=font(44, True), fill='#17324d')
draw.text((70, 92), '病例级分析：结构定量 + 视觉分类 + 管袢计数', font=font(23), fill='#52677a')

def box(x, y, w, h, title, lines, fill, accent):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=16, fill=fill, outline=accent, width=4)
    draw.rectangle((x, y, x+10, y+h), fill=accent)
    draw.text((x+28, y+22), title, font=font(25, True), fill='#17324d')
    yy = y + 68
    for line in lines:
        draw.text((x+28, yy), line, font=font(19), fill='#30465a')
        yy += 32

def arrow(x1, y1, x2, y2, label=None):
    draw.line((x1, y1, x2, y2), fill='#315a78', width=6)
    draw.polygon([(x2, y2), (x2-18, y2-12), (x2-18, y2+12)], fill='#315a78')
    if label:
        draw.text(((x1+x2)//2-38, y1-34), label, font=font(18, True), fill='#315a78')

box(70, 455, 300, 220, '原始病例文件夹', ['一次检查 = 一个病例', '按病例隔离划分', '共267例病例'], '#e8f1f8', '#2b6f9f')
arrow(370, 565, 450, 565)
box(465, 455, 275, 220, '图片帧', ['CAPorg*.jpg', '帧级输入', '帧 → 病例聚合'], '#edf7ee', '#388e5c')
arrow(740, 565, 825, 565)
box(840, 385, 390, 360, 'YOLOv11l-seg 分割', ['6类实例 mask', '正常 / 异常', '出血 / 聚集', '模糊 / 管袢', '置信度筛选'], '#fff5df', '#c98516')
arrow(1230, 455, 1340, 250, 'A')
arrow(1230, 565, 1340, 565, 'B')
arrow(1230, 675, 1340, 880, 'C')
box(1360, 120, 480, 260, '路线A：几何定量', ['骨架 + 距离变换', '端点 / 中心线定位', '管径 + 袢长', '4个连续测量字段', '开发集得分：83%-99%'], '#eaf7f1', '#218c67')
arrow(1840, 250, 1990, 565)
box(1360, 435, 480, 260, '路线B：视觉分类', ['DINOv2视觉特征', '分割统计特征', 'GBT冻结分类器', '9个分级字段'], '#edf2fb', '#4c69a8')
arrow(1840, 565, 1990, 565)
box(1360, 750, 480, 220, '路线C：管袢计数', ['正常 + 异常实例', '多阈值 + skeleton分支', '输出管袢数'], '#fff0f0', '#b44b4b')
arrow(1840, 860, 1990, 600)
box(2010, 435, 330, 260, '病例级输出', ['帧 → 病例聚合', '17个结构化字段', 'JSON + 审计记录', '兼容字段回退'], '#e9eef2', '#536878')
draw.rounded_rectangle((70, 1020, 2340, 1135), radius=14, fill='#17324d')
draw.text((100, 1050), '验证结果', font=font(20, True), fill='#ffffff')
draw.text((430, 1050), '几何字段：83%-99%', font=font(19), fill='#d9edf7')
draw.text((940, 1050), '开发集五折：0.6735', font=font(19), fill='#d9edf7')
draw.text((1360, 1050), 'Locked：47例 | 0.6902', font=font(19), fill='#d9edf7')
image.save(OUT / 'technical_architecture.png', dpi=(150, 150))
print(json.dumps({'chosen_cases': chosen, 'outputs': sorted(p.name for p in OUT.glob('*.png'))}, ensure_ascii=False, indent=2))
