# -*- coding: utf-8 -*-
"""查清 dev BA 0.799 -> locked BA 0.214 的崩塌原因。
第一步：对比 dev / locked 的类别分布、n、置信度。分布不同构就能直接解释一部分。"""
import pandas as pd, numpy as np, json, sys

MAN = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
FIELDS = ['exudation','subpapillary_venous_plexus','clarity','blood_color','papilla',
          'crossing_ratio','hemorrhage','malformation_ratio','capillary_count']

df = pd.read_csv(MAN)
print('manifest rows', len(df))
sc = [c for c in df.columns if 'split' in c.lower()]
print('split-like cols:', sc)
for c in sc:
    print(' ', c, df[c].value_counts(dropna=False).to_dict())
print()

if not sc:
    sys.exit('no split column')
SP = sc[0]
is_lock = df[SP].astype(str).str.lower().str.contains('test|lock')
dev, lock = df[~is_lock], df[is_lock]
print(f'dev n={len(dev)}  locked n={len(lock)}\n')

for f in FIELDS:
    if f not in df.columns:
        print(f'{f}: MISSING'); continue
    dv = dev[f].dropna(); lv = lock[f].dropna()
    dvc = dv.value_counts(normalize=True)
    lvc = lv.value_counts(normalize=True)
    cats = sorted(set(dvc.index) | set(lvc.index), key=str)
    print(f'=== {f}  dev_n={len(dv)} locked_n={len(lv)}')
    print(f"  {'value':22s}{'dev%':>8s}{'lock%':>8s}{'diff':>8s}")
    for c in cats:
        d, l = dvc.get(c, 0.0), lvc.get(c, 0.0)
        print(f'  {str(c)[:22]:22s}{d*100:8.1f}{l*100:8.1f}{(l-d)*100:+8.1f}')
    # 众数基线：locked 上"永远预测 dev 众数"的 BA
    if len(dvc) and len(lv):
        mode = dvc.index[0]
        classes = sorted(set(lv.astype(str)))
        rec = [1.0 if str(mode) == c else 0.0 for c in classes]
        print(f'  dev众数={mode}  -> locked 上常数预测 BA={np.mean(rec):.3f} (类别数 {len(classes)})')
    # 类别是否在 locked 出现了 dev 没有的值
    new = set(lvc.index) - set(dvc.index)
    if new: print(f'  !! locked 出现 dev 未见取值: {new}')
    conf = f + '__confidence'
    if conf in df.columns:
        print(f"  confidence dev={pd.to_numeric(dev[conf],errors='coerce').mean():.3f} "
              f"locked={pd.to_numeric(lock[conf],errors='coerce').mean():.3f}")
    print()
