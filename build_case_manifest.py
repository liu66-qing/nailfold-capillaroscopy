import json, csv
fields=['clarity','capillary_count','crossing_ratio','malformation_ratio','blood_color','exudation','hemorrhage','subpapillary_venous_plexus','papilla','sweat_duct']
rows={}
for model,method,path in [('Qwen3-VL-8B','zero_shot','qwen_zero_pred.jsonl'),('Qwen3-VL-8B','few_shot','qwen_few_pred.jsonl'),('MedGemma-4B','zero_shot','med_zero_pred.jsonl'),('MedGemma-4B','few_shot','med_few_pred.jsonl'),('InternVL3-8B','zero_shot','intern_zero_pred.jsonl'),('InternVL3-8B','few_shot','intern_few_pred.jsonl')]:
    try: data=[json.loads(x) for x in open(path,encoding='utf-8')]
    except FileNotFoundError: continue
    for r in data:
        key=(r['exam_case_id'],model,method); out={'exam_case_id':r['exam_case_id'],'model':model,'method':method,'fold':r['fold'],'json_valid':r['json_valid']}
        for f in fields: out[f+'_true']=r['target'].get(f); out[f+'_pred']=r['prediction'].get(f)
        rows[key]=out
cols=['exam_case_id','model','method','fold','json_valid']+[x for f in fields for x in (f+'_true',f+'_pred')]
with open('甲襞病例级模型预测清单.csv','w',newline='',encoding='utf-8-sig') as h:
    w=csv.DictWriter(h,fieldnames=cols); w.writeheader(); w.writerows(rows.values())
