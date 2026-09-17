from pathlib import Path
import csv, json, xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
DER=ROOT/'artifacts/derived/vascular_dataset_governance_20260830'
SRC=DER/'xml_repaired_normalized'; OUT=DER/'xml_adjudicated_final'; OUT.mkdir(parents=True,exist_ok=True)

DEC={'D340':('B','530','B真实可见血管，但新增框较模糊；保留并标记低质量风险'),'D486':('A','522','交叉血管语义确认，采用 A'),'D494':('MERGE','364+505','两份框互补，合并并按 IoU 去重')}

def objs(stem):
 r=ET.parse(SRC/f'{stem}.xml').getroot(); return r,[o for o in r.findall('object')]
def box(o):
 b=o.find('bndbox'); return [float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1]); return inter/(ua+ub-inter) if ua+ub-inter else 0
def main():
 rows=[]
 for gid,(decision,source,reason) in DEC.items():
  stems=source.split('+'); root, oa=objs(stems[0]); selected=list(oa); removed=0
  if decision=='MERGE':
   _, ob=objs(stems[1]);
   for o in ob:
    if not any(iou(box(o),box(x))>=0.5 and (o.findtext('name') or '')==(x.findtext('name') or '') for x in selected): selected.append(o)
    else: removed+=1
  out_root=ET.fromstring(ET.tostring(root,encoding='unicode'))
  for x in list(out_root.findall('object')): out_root.remove(x)
  for o in selected: out_root.append(ET.fromstring(ET.tostring(o,encoding='unicode')))
  dst=OUT/f'{gid}.xml'; ET.ElementTree(out_root).write(dst,encoding='utf-8',xml_declaration=True)
  rows.append({'duplicate_group_id':gid,'decision':decision,'source_xmls':';'.join(stems),'derived_xml':str(dst),'source_box_count':sum(len(objs(s)[1]) for s in stems),'derived_box_count':len(selected),'merge_removed_overlap_boxes':removed,'reason':reason,'status':'FINAL_ADJUDICATION_DERIVED','raw_overwritten':'no'})
 with (DER/'duplicate_adjudicated_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # update decision table rows
 p=GOV/'duplicate_group_decisions.csv'; old=list(csv.DictReader(p.open(encoding='utf-8-sig'))); by={r['duplicate_group_id']:r for r in rows}
 for r in old:
  if r['duplicate_group_id'] in by:
   z=by[r['duplicate_group_id']]; r['decision']=z['decision']; r['ai_recommended_final']=z['decision']; r['decision_reason']=z['reason']; r['reviewer']='user+AI_visual_adjudication'; r['review_time']='2026-08-30'
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
 fs=json.loads((GOV/'final_status.json').read_text(encoding='utf-8')); fs['duplicate_adjudication']={'A':['D009','D058','D463','D469','D540','D486'],'B':['D356','D401','D340'],'HOLD':[],'MERGE':['D494'],'locked_selected_members_excluded':['D540:A=13']}; fs['still_hold']['duplicate_annotation_conflict_groups']=0; (GOV/'final_status.json').write_text(json.dumps(fs,ensure_ascii=False,indent=2),encoding='utf-8')
 (GOV/'duplicate_adjudication_report.md').write_text('# 重复组最终裁决（2026-08-30）\n\n用户完成 D340、D486、D494 语义裁决后，10 组均已形成最终派生标签：A=6 组、B=3 组、MERGE=1 组、HOLD=0。D340 采用 530.xml 并保留模糊框风险；D486 采用 522.xml，交叉血管语义确认；D494 合并 364/505 并按同类框 IoU>=0.5 去重。D540 的 A=13.jpg 属于 locked，仍永久排除。\n',encoding='utf-8')
if __name__=='__main__':main()
