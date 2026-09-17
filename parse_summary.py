import json
d=json.load(open(r'E:\甲劈微循环\summary_remote.json',encoding='utf-8'))
for k,v in d['results'].items(): print(k, round(v['balanced_accuracy'],3), round(v['delta'],3), v['delta_ci95'])
