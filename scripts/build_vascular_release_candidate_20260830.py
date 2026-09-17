from __future__ import annotations
import csv,hashlib,json,random,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
DER=ROOT/'artifacts/derived/vascular_dataset_governance_20260830'
TRAIN=DER/'train_ready'; REL=DER/'release_v20260830'

def md5(p):
 h=hashlib.md5();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0]) if rows else ['status']
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def source_gate():
 rows=list(csv.DictReader((GOV/'source_case_mapping.csv').open(encoding='utf-8-sig')))
 out=[]
 for r in rows:
  s=r['source_mapping_status'];
  if s=='EXACT_RECOVERED_LOCKED': st='EXCLUDE_LOCKED'
  elif s=='EXACT_RECOVERED_DEVELOPMENT': st='PASS_AUXILIARY_USER_ASSERTED_PROVENANCE'
  elif s=='EXACT_RECOVERED_UNLISTED': st='HOLD_UNLISTED_CASE'
  else: st='HOLD_UNKNOWN_CASE'
  out.append({'asset_id':r['original_id'],'image_path':r['image_path'],'linked_recovered_cases':r['linked_recovered_cases'],'development': 'yes' if s=='EXACT_RECOVERED_DEVELOPMENT' else 'no','locked': 'yes' if s=='EXACT_RECOVERED_LOCKED' else 'no','provenance_status':st,'permission_status':'USER_ASSERTED_SAME_DATASET_PERMISSION' if st.startswith('PASS_') else 'UNKNOWN','split_status':'DEVELOPMENT_ONLY_CASE_SPLIT_REQUIRED' if st.startswith('PASS_') else 'HOLD'})
 return out

def pairing_gate():
 rows=list(csv.DictReader((GOV/'segmentation_pair_reconstruction.csv').open(encoding='utf-8-sig'))); out=[]
 for r in rows:
  st='PASS_PAIR' if r['pair_status']=='AUTO_CONFIRMED_PAIR' else 'HOLD_PAIRING'
  out.append({'json_path':r['json_path'],'image_path':r['reconstructed_image'],'mask_path':r['reconstructed_mask'],'pair_status':st,'image_evidence':r['image_evidence'],'mask_evidence':r['mask_evidence'],'mask_polygon_iou':r['mask_polygon_iou'],'pairing_rule':'embedded pixel match or deterministic sequence-offset+dimension+mask geometry' if st=='PASS_PAIR' else 'insufficient evidence; no guessing'})
 return out

def ai_sample_report(source_rows,pair_rows):
 random.seed(20260830)
 cls=[r for r in source_rows if r['provenance_status']!='EXCLUDE_LOCKED']
 cls=sorted(cls,key=lambda r:int(r['asset_id']))
 cls_sample=cls[:100] if len(cls)>=100 else cls
 cls_out=[]
 for r in cls_sample:
  p=Path(r['image_path']); im=Image.open(p).convert('L'); a=np.asarray(im); cls_out.append({'asset_id':r['asset_id'],'image_path':str(p),'width':im.width,'height':im.height,'mean_intensity':round(float(a.mean()),3),'std_intensity':round(float(a.std()),3),'ai_quality':'REVIEW_REQUIRED','reason':'pixel statistics do not replace semantic/clinical review'})
 seg=[r for r in pair_rows if r['pair_status']=='PASS_PAIR']
 # stratified deterministic sample: lowest IoU, sequence-offset and regular records
 seg=sorted(seg,key=lambda r:(float(r['mask_polygon_iou'] or 0),r['json_path']))
 seg_sample=(seg[:34]+seg[len(seg)//2-33:len(seg)//2+33]+seg[-33:])[:100] if len(seg)>=100 else seg
 seg_out=[]
 for r in seg_sample:
  ip,mp=Path(r['image_path']),Path(r['mask_path']); ok=ip.exists() and mp.exists(); ms=np.asarray(Image.open(mp).convert('L')) if ok else np.zeros((1,1)); seg_out.append({'json_path':r['json_path'],'image_path':str(ip),'mask_path':str(mp),'mask_iou':r['mask_polygon_iou'],'mask_foreground_fraction':round(float((ms>0).mean()),6) if ok else '','ai_quality':'REVIEW_REQUIRED','reason':'AI pairing/geometry check only; visual completeness and clinical acceptability require reviewer'})
 return cls_out,seg_out

def main():
 REL.mkdir(parents=True,exist_ok=True)
 source=source_gate(); pairs=pairing_gate(); cls_s,seg_s=ai_sample_report(source,pairs)
 write_csv(REL/'final_governance_manifest.csv',source)
 write_csv(REL/'final_pairing_table.csv',pairs)
 write_csv(REL/'classification_ai_qc_sample_100.csv',cls_s)
 write_csv(REL/'segmentation_ai_qc_sample_100.csv',seg_s)
 # Hash every release artifact and all files inside train_ready.
 hash_rows=[]
 for p in sorted(list(REL.glob('*'))+[x for x in TRAIN.rglob('*') if x.is_file()]):
  hash_rows.append({'path':str(p),'md5':md5(p),'size_bytes':p.stat().st_size,'role':'release_artifact' if p.parent==REL else 'train_ready_asset'})
 write_csv(REL/'release_hash_manifest.csv',hash_rows)
 label_doc=GOV/'semantic_definition_evidence.json'; cal=GOV/'device_calibration_status.json'
 (REL/'label_definitions.md').write_text('''# 发布候选标签定义\n\n- `vessel`：普通可见血管实例（工程候选定义）。\n- `malformed_vessel`：形态异常血管实例（工程候选定义）。\n- `cross_vessel`：交叉/吻合区域或事件框（当前工作定义）。\n- `corss_vessel`：仅做拼写归一化，不改变语义。\n\n弱字段：SVP、红细胞聚集、乳头、汗腺导管和出血均为图片级辅助标签，结构完整但需临床最终认可。外部/同学标注不称为临床金标准。\n''',encoding='utf-8')
 (REL/'calibration_statement.md').write_text('''# 设备与标定声明\n\n用户确认同源数据集大概率使用同一设备、倍率和标尺，但当前没有型号、倍率或像素标尺原始证据。因此本发布候选只允许像素域和相对几何用途；禁止绝对微米测量和临床几何声明。\n''',encoding='utf-8')
 locked=list(csv.DictReader((GOV/'locked_exclusion_manifest.csv').open(encoding='utf-8-sig')))
 summary={'release_version':'vascular_dataset_release_candidate_v20260830','status':'RELEASE_CANDIDATE_HOLD','created_at':'2026-08-30','raw_data_modified':False,'v1_modified':False,'gpu_used':False,'classification_source_rows':len(source),'classification_pass_auxiliary':sum(r['provenance_status'].startswith('PASS_') for r in source),'classification_locked_excluded':sum(r['provenance_status']=='EXCLUDE_LOCKED' for r in source),'segmentation_pair_rows':len(pairs),'segmentation_pass_pairs':sum(r['pair_status']=='PASS_PAIR' for r in pairs),'segmentation_hold_pairs':sum(r['pair_status']=='HOLD_PAIRING' for r in pairs),'classification_ai_qc_sample':len(cls_s),'segmentation_ai_qc_sample':len(seg_s),'locked_exclusion_assets':len(locked),'remaining_release_blockers':['user/clinical approval of provenance and permission for unknown/unlisted source','clinical acceptance of label definitions','visual QC sign-off for 100 classification and 100 segmentation samples','829 unresolved/low-evidence pairing rows are not released','device calibration evidence for micron geometry'],'allowed_use':'conditional auxiliary pretraining only after split and human sign-off'}
 (REL/'release_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (REL/'RELEASE_README.md').write_text('''# 血管数据集发布候选版本\n\n版本：`vascular_dataset_release_candidate_v20260830`。\n\n这是可追溯、可回退的发布候选包，不是无条件放行版本。所有原始文件只读保留；训练资产来自独立 `artifacts/derived`。locked 文件、增强独立样本和未经证实的分割配对均未进入可训练目录。\n\n正式发布前必须完成 `release_summary.json` 中的剩余闸门。\n''',encoding='utf-8')

if __name__=='__main__':main()
