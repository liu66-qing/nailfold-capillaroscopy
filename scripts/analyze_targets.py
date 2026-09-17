"""Target arithmetic: what per-field score MAE is needed for total_score MAE ~2.2.
Also compares exp1 model sMAE against the mode-fill baseline per field.
Read-only.
"""
import json
import numpy as np
import pandas as pd

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
RULES = '/root/autodl-tmp/nailfold/artifacts/labels/score_rules_v3.json'
EXP1 = '/root/nailfold/artifacts/experiments/exp1_multiclass/exp1_results.json'
EXCLUDE = ['recovered_archive2/180', 'recovered_archive3/263']
OBS = ['microthrombus', 'exudation', 'papilla', 'capillary_count',
       'subpapillary_venous_plexus', 'blood_color', 'clarity']

rules = json.load(open(RULES))
fr = rules['field_rules']
ALL_FIELDS = [f for g in rules['groups'].values() for f in g]

df = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
dev = df[(df['evaluation_role'] == 'development') & (~df['exam_case_id'].isin(EXCLUDE))]
dev = dev[dev['total_score'].notna()].copy()
truth = dev['total_score'].values
n = len(dev)


def fscore(frame, f):
    r = fr.get(f, {})
    if r.get('type') == 'categorical_lookup':
        return frame[f].map(r['mapping'])
    if r.get('type') == 'numeric_tree':
        t = r['tree']

        def ev(x):
            if pd.isna(x):
                return np.nan
            nd = t
            while 'value' not in nd:
                nd = nd['left'] if x <= nd['threshold'] else nd['right']
            return nd['value']
        return pd.to_numeric(frame[f], errors='coerce').map(ev)
    return pd.Series(np.nan, index=frame.index)


S = pd.DataFrame({f: fscore(dev, f) for f in ALL_FIELDS}, index=dev.index)
mode_s = {f: (S[f].mode().iloc[0] if S[f].notna().any() else 0.0) for f in ALL_FIELDS}
Sf = S.copy()
for f in ALL_FIELDS:
    Sf[f] = Sf[f].fillna(mode_s[f])

print('=== exp1 model sMAE vs mode-fill baseline (the honest comparison) ===')
e1 = json.load(open(EXP1))['per_field']
print('%-28s %8s %8s %9s %8s' % ('field', 'modeMAE', 'exp1MAE', 'gain', 'BA'))
tot_gain = 0.0
for f in OBS:
    mm = (Sf[f] - mode_s[f]).abs().mean()
    em = e1[f]['smae']
    print('%-28s %8.4f %8.4f %+9.4f %8.3f' % (f, mm, em, mm - em, e1[f]['ba']))
    tot_gain += mm - em
print('sum of per-field gains: %+.4f' % tot_gain)

NONOBS = [f for f in ALL_FIELDS if f not in OBS]
fixed = sum(mode_s[f] for f in NONOBS)
print('\nnon-observable mode sum = %.3f' % fixed)
print('perfect-observable ceiling MAE = %.4f' % np.abs(
    Sf[OBS].sum(axis=1).values + fixed - truth).mean())

print('\n=== sweep: recover fraction alpha of deviation, all 7 observable ===')
print('%5s %8s %8s %8s | %s' % ('alpha', 'totMAE', 'spear', 'sdratio', 'per-field MAE'))
res = []
for alpha in np.arange(0, 1.01, 0.05):
    pred = np.full(n, fixed)
    pf = {}
    for f in OBS:
        sv = Sf[f].values
        p = mode_s[f] + alpha * (sv - mode_s[f])
        pf[f] = np.abs(p - sv).mean()
        pred = pred + p
    mae = np.abs(pred - truth).mean()
    res.append((alpha, mae, pf))
    if abs(alpha * 20 - round(alpha * 20)) < 1e-9 and round(alpha * 20) % 2 == 0:
        print('%5.2f %8.4f %8.3f %8.3f | %s' % (
            alpha, mae,
            pd.Series(pred).corr(pd.Series(truth), method='spearman'),
            pred.std() / truth.std(),
            ' '.join('%s=%.3f' % (f[:6], pf[f]) for f in OBS[:4])))

target = 2.2
best = min(res, key=lambda r: abs(r[1] - target))
print('\n=== ALPHA NEEDED FOR total MAE ~= %.2f -> alpha=%.2f (MAE %.4f) ===' % (
    target, best[0], best[1]))
print('%-28s %10s %10s %10s' % ('field', 'modeMAE', 'TARGET', 'exp1now'))
for f in OBS:
    mm = (Sf[f] - mode_s[f]).abs().mean()
    print('%-28s %10.4f %10.4f %10.4f' % (f, mm, best[2][f], e1[f]['smae']))

print('\n=== if ONLY top-3-by-headroom improve, rest stay at mode ===')
for T in [['microthrombus'], ['microthrombus', 'exudation'],
          ['microthrombus', 'exudation', 'papilla'],
          ['microthrombus', 'exudation', 'papilla', 'capillary_count']]:
    out = []
    for alpha in [0.5, 0.7, 0.9, 1.0]:
        pred = np.full(n, fixed)
        for f in OBS:
            sv = Sf[f].values
            a = alpha if f in T else 0.0
            pred = pred + mode_s[f] + a * (sv - mode_s[f])
        out.append('a=%.1f:%.3f' % (alpha, np.abs(pred - truth).mean()))
    print('%-60s %s' % ('+'.join(x[:9] for x in T), '  '.join(out)))
