from pathlib import Path
import csv, xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
IMG=ROOT/'data/血管数据集/分类数据集/扩充之前/images'
XML=ROOT/'artifacts/derived/vascular_dataset_governance_20260830/xml_repaired_normalized'
OUT=ROOT/'artifacts/derived/vascular_dataset_governance_20260830/manual_duplicate_review'

def boxes(stem):
    root=ET.parse(XML/f'{stem}.xml').getroot(); out=[]
    for i,o in enumerate(root.findall('object'),1):
        b=o.find('bndbox'); vals=[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        out.append({'idx':i,'label':o.findtext('name') or '','bbox':vals})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups={'D340':('354','530'),'D486':('522','69'),'D494':('364','505')}
    guide=[]
    for gid,(a,b) in groups.items():
        src=IMG/f'{a}.jpg'; im=Image.open(src).convert('RGB'); ow,oh=im.size; scale=min(900/ow,650/oh); nw,nh=round(ow*scale),round(oh*scale)
        canvas=Image.new('RGB',(nw*2+40,nh+100),'white'); font=ImageFont.load_default()
        for col,stem,color,title in [(0,a,(220,40,40),f'A = {a}.xml'),(1,b,(20,150,40),f'B = {b}.xml')]:
            p=im.resize((nw,nh)); d=ImageDraw.Draw(p)
            for bx in boxes(stem):
                x1,y1,x2,y2=bx['bbox']; box=(x1*scale,y1*scale,x2*scale,y2*scale); d.rectangle(box,outline=color,width=4); d.text((box[0]+2,box[1]+2),f"{bx['idx']}:{bx['label']}",fill=color,stroke_width=1,stroke_fill='black')
            x=20+col*(nw+20); canvas.paste(p,(x,55)); ImageDraw.Draw(canvas).text((x,15),title,fill=color,font=font)
        d=ImageDraw.Draw(canvas); d.text((20,nh+65),f'{gid}: 红=A，绿=B；编号=XML object 序号，文字=类别',fill='black',font=font)
        canvas.save(OUT/f'{gid}_A_B_overlay.png')
        sa,sb=boxes(a),boxes(b)
        guide.append({'group_id':gid,'member_a':a,'member_b':b,'a_box_count':len(sa),'b_box_count':len(sb),'a_labels':';'.join(x['label'] for x in sa),'b_labels':';'.join(x['label'] for x in sb),'review_question':'逐框判断哪一版更贴合可见血管；若两版均有独特正确框，再考虑 MERGE；若无法确认则 HOLD','decision_options':'A / B / MERGE / HOLD','overlay_path':str(OUT/f'{gid}_A_B_overlay.png')})
    with (OUT/'manual_review_guide.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(guide[0]));w.writeheader();w.writerows(guide)
    (OUT/'README.md').write_text('''# 三组重复 XML 人工语义裁决\n\n请分别打开 `D340_A_B_overlay.png`、`D486_A_B_overlay.png`、`D494_A_B_overlay.png`。每张图左右是同一张原图：红框=A，绿框=B；框内 `序号:类别` 对应 XML object。\n\n重点不是比较框数量，而是判断：\n\n1. 框是否覆盖真实可见血管；\n2. `vessel/malformed_vessel/cross_vessel` 类别是否合理；\n3. 两边是否各有独特且正确的框；\n4. 合并是否会重复框或制造错误标签。\n\n裁决填写在 `artifacts/audits/vascular_dataset_governance_20260830/duplicate_group_decisions.csv` 的 `decision` 列：`A`、`B`、`MERGE` 或 `HOLD`。逐组说明见 `manual_review_guide.csv`。\n''',encoding='utf-8')

if __name__=='__main__':main()
