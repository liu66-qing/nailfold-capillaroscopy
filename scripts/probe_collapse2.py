# -*- coding: utf-8 -*-
"""崩塌诊断第二步：geometry locked 的 43 例究竟是谁？混淆矩阵长什么样？"""
import json, pandas as pd, numpy as np, collections, sys

GEO = '/root/nailfold/geo_metrics_v3_locked.json'
MAN = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'

geo = json.load(open(GEO, encoding='utf-8'))['results']
man = pd.read_csv(MAN)
idcol = 'exam_case_id'
split = dict(zip(man[idcol], man['split'].astype(str)))

for f in ['exudation', 'subpapillary_venous_plexus', 'clarity', 'blood_color']:
    v = geo[f]
    recs = v['records']
    ids = [r[idcol] for r in recs]
    sp = collections.Counter(split.get(i, 'NOT_IN_MANIFEST') for i in ids)
    print(f"=== {f}  n={v['n']}  BA={v['balanced_accuracy']:.3f}")
    print('  这些例子的 split 归属:', dict(sp))

    t = [r['truth'] for r in recs]; p = [r['prediction'] for r in recs]
    cats = sorted(set(t) | set(p), key=str)
    print(f"  混淆矩阵 (行=真值, 列=预测)  类别 {cats}")
    for a in cats:
        row = [sum(1 for x, y in zip(t, p) if x == a and y == b) for b in cats]
        n_a = sum(1 for x in t if x == a)
        rec = row[cats.index(a)] / n_a if n_a else float('nan')
        print(f"    {str(a)[:14]:14s}{str(row):28s} n={n_a:3d} recall={rec:.3f}")
    # 预测分布 vs 真值分布
    print('  真值分布', dict(collections.Counter(t)))
    print('  预测分布', dict(collections.Counter(p)))
    acc = np.mean([x == y for x, y in zip(t, p)])
    # 众数基线
    mode = collections.Counter(t).most_common(1)[0][0]
    macc = np.mean([x == mode for x in t])
    mba = 1.0 / len(set(t))
    print(f'  accuracy={acc:.3f}  众数基线 acc={macc:.3f}  众数基线 BA={mba:.3f}')
    print()
