import json,collections
p='/root/autodl-tmp/nailfold/artifacts/pseudo_labels/round1_multiclass_domain_adapted.json';d=json.load(open(p)); c=collections.Counter(i.get('class_name') for v in d.values() for i in v.get('instances',[])); print(len(d),sum(c.values()),c)
