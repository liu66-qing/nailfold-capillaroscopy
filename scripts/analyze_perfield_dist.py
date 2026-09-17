"""Analyze value distributions + score mappings for the 7 observable fields.
Development rows only, conflicted cases excluded. Read-only analysis.
"""
import json
import pandas as pd

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
IMGIDX = '/root/nailfold/artifacts/features/image_index.csv'
RULES = '/root/autodl-tmp/nailfold/artifacts/labels/score_rules_v3.json'
EXCLUDE = ['recovered_archive2/180', 'recovered_archive3/263']

FIELDS = ['capillary_count', 'microthrombus', 'exudation', 'papilla',
          'subpapillary_venous_plexus', 'blood_color', 'clarity']

rules = json.load(open(RULES))
fr = rules['field_rules']

df = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
print('manifest rows', len(df))
print('evaluation_role values', df['evaluation_role'].value_counts().to_dict()
      if 'evaluation_role' in df.columns else 'MISSING COLUMN')
print('has development_fold', 'development_fold' in df.columns)

dev = df[df['evaluation_role'] == 'development'].copy()
dev = dev[~dev['exam_case_id'].isin(EXCLUDE)]
print('dev rows after exclude', len(dev))
if 'development_fold' in dev.columns:
    print('fold counts', dev['development_fold'].value_counts().sort_index().to_dict())
    print('fold nulls', int(dev['development_fold'].isna().sum()))

idx = pd.read_csv(IMGIDX, dtype={'exam_case_id': str})
fpc = idx.groupby('exam_case_id').size()
devids = set(dev['exam_case_id'])
fpc_dev = fpc[fpc.index.isin(devids)]
print('\nFRAMES: cases with frames', len(fpc_dev), 'total frames', int(fpc_dev.sum()))
print('frames/case min/mean/median/max',
      int(fpc_dev.min()), round(float(fpc_dev.mean()), 2),
      float(fpc_dev.median()), int(fpc_dev.max()))
print('cases with NO frames', len(devids - set(fpc.index)))
print('frame count histogram', fpc_dev.value_counts().sort_index().to_dict())

print('\n' + '=' * 70)
for f in FIELDS:
    rule = fr.get(f, {})
    print('\n### FIELD', f, '| rule type', rule.get('type'), '| rule cases', rule.get('cases'))
    mapping = rule.get('mapping')
    print('mapping:', json.dumps(mapping, ensure_ascii=False) if mapping else rule.get('tree'))
    vc = dev[f].value_counts(dropna=False)
    print('dev value counts (n=%d):' % len(dev))
    total_scored = 0
    for v, c in vc.items():
        sc = mapping.get(v) if (mapping and isinstance(v, str)) else None
        print('   %-14r n=%-4d score=%s' % (v, c, sc))
        if pd.notna(v):
            total_scored += c
    print('   non-null: %d  null: %d' % (total_scored, int(dev[f].isna().sum())))
    if mapping:
        sv = dev[f].map(mapping)
        print('   score contribution: min=%s max=%s range=%s mean=%.3f' % (
            sv.min(), sv.max(), (sv.max() - sv.min()) if sv.notna().any() else None,
            sv.mean() if sv.notna().any() else float('nan')))
        # MAE of constant-median predictor on this field alone
        med = sv.median()
        print('   const-median field MAE=%.4f (median=%s)' % ((sv - med).abs().mean(), med))
        mode_v = dev[f].mode()
        if len(mode_v):
            ms = mapping.get(mode_v.iloc[0])
            print('   mode=%r score=%s  mode-rate=%.3f  mode-pred MAE=%.4f' % (
                mode_v.iloc[0], ms, (dev[f] == mode_v.iloc[0]).mean(),
                (sv - ms).abs().mean() if ms is not None else float('nan')))
    # status/confidence
    st = f + '__status'
    if st in dev.columns:
        print('   status:', dev[st].value_counts(dropna=False).to_dict())

print('\n' + '=' * 70)
print('total_score column present:', 'total_score' in df.columns)
for c in ['total_score', 'morphology_score', 'flow_score', 'periloop_score']:
    if c in dev.columns:
        print(c, 'nonnull', int(dev[c].notna().sum()), 'mean %.3f' % dev[c].mean(),
              'sd %.3f' % dev[c].std(), 'median', dev[c].median())
