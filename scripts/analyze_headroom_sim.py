"""Decompose total_score and simulate how per-field score MAE composes into total MAE.
Read-only. Answers: which fields actually carry headroom, and what per-field MAE
targets are needed for total MAE ~2.2.
"""
import json
import itertools
import numpy as np
import pandas as pd

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
RULES = '/root/autodl-tmp/nailfold/artifacts/labels/score_rules_v3.json'
EXCLUDE = ['recovered_archive2/180', 'recovered_archive3/263']
OBS = ['capillary_count', 'microthrombus', 'exudation', 'papilla',
       'subpapillary_venous_plexus', 'blood_color', 'clarity']

rules = json.load(open(RULES))
fr = rules['field_rules']
groups = rules['groups']
ALL_FIELDS = [f for g in groups.values() for f in g]
print('all fields in groups:', len(ALL_FIELDS))

df = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
dev = df[df['evaluation_role'] == 'development'].copy()
dev = dev[~dev['exam_case_id'].isin(EXCLUDE)]
dev = dev[dev['total_score'].notna()].copy()
print('dev cases with total_score:', len(dev))


def field_score(frame, f):
    """Map raw field value -> score using score_rules_v3."""
    rule = fr.get(f, {})
    t = rule.get('type')
    if t == 'categorical_lookup':
        return frame[f].map(rule['mapping'])
    if t == 'numeric_tree':
        tree = rule['tree']

        def ev(x):
            if pd.isna(x):
                return np.nan
            n = tree
            while 'value' not in n:
                n = n['left'] if x <= n['threshold'] else n['right']
            return n['value']
        return pd.to_numeric(frame[f], errors='coerce').map(ev)
    return pd.Series(np.nan, index=frame.index)


S = pd.DataFrame({f: field_score(dev, f) for f in ALL_FIELDS}, index=dev.index)
recon = S.sum(axis=1, min_count=1)
print('\n--- Does sum(field scores) == total_score? ---')
d = (recon - dev['total_score']).abs()
print('mean abs diff %.4f  median %.4f  max %.4f  frac<0.01 %.3f' % (
    d.mean(), d.median(), d.max(), (d < 0.01).mean()))

print('\n--- Per-observable-field headroom (mode-fill MAE = max MAE a perfect model removes) ---')
rows = []
for f in OBS:
    sv = S[f]
    rng = sv.max() - sv.min()
    mode_raw = dev[f].mode()
    ms = fr[f]['mapping'].get(mode_raw.iloc[0]) if len(mode_raw) else np.nan
    mode_mae = (sv - ms).abs().mean()
    med_mae = (sv - sv.median()).abs().mean()
    rows.append(dict(field=f, dyn_range=rng, sd=sv.std(),
                     mode_rate=(dev[f] == mode_raw.iloc[0]).mean() if len(mode_raw) else np.nan,
                     mode_fill_MAE=mode_mae, median_fill_MAE=med_mae))
H = pd.DataFrame(rows).sort_values('mode_fill_MAE', ascending=False)
print(H.to_string(index=False, float_format=lambda x: '%.4f' % x))
print('\nsum of mode_fill_MAE over 7 observable fields: %.4f' % H.mode_fill_MAE.sum())
print('RANK BY DYN RANGE :', list(H.sort_values("dyn_range", ascending=False).field))
print('RANK BY REAL HEADROOM:', list(H.field))

# --- ceiling reproduction: perfect on observable, mode elsewhere ---
NONOBS = [f for f in ALL_FIELDS if f not in OBS]
base_mode = {}
for f in NONOBS:
    sv = S[f]
    base_mode[f] = sv.mode().iloc[0] if sv.notna().any() else 0.0
fixed = sum(base_mode[f] for f in NONOBS)
truth = dev['total_score'].values

perfect_obs = S[OBS].fillna(0.0).sum(axis=1).values + fixed
print('\n--- ceiling checks (vs true total_score) ---')
print('perfect-observable + mode-elsewhere : MAE %.4f  spearman %.4f  sd_ratio %.4f' % (
    np.abs(perfect_obs - truth).mean(),
    pd.Series(perfect_obs).corr(pd.Series(truth), method='spearman'),
    perfect_obs.std() / truth.std()))
const = np.median(truth)
print('constant-median predictor          : MAE %.4f' % np.abs(const - truth).mean())

# all-mode baseline (mode for every field incl observable)
allmode = np.full(len(dev), 0.0)
tot_mode = 0.0
for f in ALL_FIELDS:
    sv = S[f]
    tot_mode += sv.mode().iloc[0] if sv.notna().any() else 0.0
print('all-field mode predictor           : MAE %.4f (pred=%.2f)' % (
    np.abs(tot_mode - truth).mean(), tot_mode))

# --- SIMULATION: what per-field MAE is needed for total MAE ~2.2 ---
# Model: for a chosen subset of fields, the model recovers fraction alpha of the
# error relative to mode-fill; other fields stay at mode. Errors are simulated by
# shrinking each field's residual toward truth.
print('\n--- simulation: shrink residuals on target fields ---')
print('target fields = microthrombus, exudation, capillary_count, papilla')
TARGETS = ['microthrombus', 'exudation', 'capillary_count', 'papilla']
mode_scores = {}
for f in ALL_FIELDS:
    sv = S[f]
    mode_scores[f] = sv.mode().iloc[0] if sv.notna().any() else 0.0

for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    pred = np.zeros(len(dev))
    per_field_mae = {}
    for f in ALL_FIELDS:
        sv = S[f].fillna(mode_scores[f]).values
        m = mode_scores[f]
        if f in TARGETS:
            p = m + alpha * (sv - m)   # recovers alpha of the deviation
        else:
            p = np.full(len(dev), m)
        per_field_mae[f] = np.abs(p - sv).mean()
        pred += p
    mae = np.abs(pred - truth).mean()
    sp = pd.Series(pred).corr(pd.Series(truth), method='spearman')
    print('alpha=%.1f total MAE %.4f spearman %.3f sd_ratio %.3f | per-field MAE: %s' % (
        alpha, mae, sp, pred.std() / truth.std(),
        ' '.join('%s=%.3f' % (f[:7], per_field_mae[f]) for f in TARGETS)))
