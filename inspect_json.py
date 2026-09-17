import json
p='/root/autodl-tmp/nailfold/artifacts/pseudo_labels/round1_multiclass_domain_adapted.json'; d=json.load(open(p)); k=next(iter(d)); i=d[k]['instances'][0]; print('cases',len(d),'case',k,'instance_keys',list(i),'top_keys',list(d[k]))
