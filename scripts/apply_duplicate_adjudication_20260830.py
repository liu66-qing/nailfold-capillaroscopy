from pathlib import Path
import csv, json

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
DEC={
 'D009':('A','A has one additional vessel box and visually broader coverage.'),
 'D058':('A','A has one additional cross-vessel box and broader coverage.'),
 'D340':('HOLD','Different vessel boxes; merge risks duplicate supervision.'),
 'D356':('B','B covers substantially more visible loops; A is under-annotated.'),
 'D401':('B','B adds a visible right-side vessel and retains the cross-vessel box.'),
 'D463':('A','A covers one additional visible vessel; both use vessel class.'),
 'D469':('A','A has broader vessel coverage; B has fewer boxes.'),
 'D486':('HOLD','A/B assign different semantics to central structures.'),
 'D494':('HOLD','Same count but different coordinates; no safe merge policy.'),
 'D540':('A','A covers substantially more visible structures; B appears under-annotated.'),
}
def main():
 src=list(csv.DictReader((GOV/'duplicate_group_decisions.csv').open(encoding='utf-8-sig'))); rows=[]
 for r in src:
  d,reason=DEC[r['duplicate_group_id']]; r['ai_recommended_final']=d; r['decision']=d; r['decision_reason']=reason; r['reviewer']='AI_visual_adjudication_v1'; r['review_time']='2026-08-30'; rows.append(r)
 with (GOV/'duplicate_group_decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 fs=json.loads((GOV/'final_status.json').read_text(encoding='utf-8'))
 fs['duplicate_adjudication']={'A':['D009','D058','D463','D469','D540'],'B':['D356','D401'],'HOLD':['D340','D486','D494'],'MERGE':[],'locked_selected_members_excluded':['D540:A=13']}
 fs['still_hold']['duplicate_annotation_conflict_groups']=3
 (GOV/'final_status.json').write_text(json.dumps(fs,ensure_ascii=False,indent=2),encoding='utf-8')
 (GOV/'duplicate_adjudication_report.md').write_text('# 重复组最终 AI 裁决（2026-08-30）\n\n已对 10 组完全相同原图生成框叠加图并逐组比较覆盖范围、类别差异和潜在重复框。规则：框明显更完整且无独特反证时选 A/B；类别语义冲突无法由图像覆盖判断时 HOLD；没有去重政策时不自动 MERGE。\n\n最终裁决：A=5 组（D009、D058、D463、D469、D540）；B=2 组（D356、D401）；HOLD=3 组（D340、D486、D494）；MERGE=0。HOLD 组不能进入检测训练；A/B 组仍需经过 locked、病例级 split 和类别语义闸门。\n',encoding='utf-8')
if __name__=='__main__': main()
