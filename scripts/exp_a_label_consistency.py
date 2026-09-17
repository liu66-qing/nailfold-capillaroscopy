"""EXP-A: label consistency audit + EXP-B feasibility (video coverage).

No GPU. Establishes, before any training:
  1. total_score vs report overall_assessment conflicts under both cut rules.
  2. Which cases are conflicted (to be EXCLUDED / flagged, not relabelled).
  3. Video coverage of the dev set, and whether flow-field label variance
     survives inside the video subset (feasibility gate for EXP-B).
  4. Date-group structure for the leakage sensitivity analysis (EXP-D).
"""
import glob, json, os, re, collections
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
DATA = '/root/nailfold/data'
OUT = Path('/root/nailfold/artifacts/experiments/exp_a_label_audit')
OUT.mkdir(parents=True, exist_ok=True)

SEV = ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"]
CUTS = [1.0, 2.0, 4.0, 8.0]


def cut(s, inclusive):
    for t, lab in zip(CUTS, SEV[:-1]):
        if (s <= t) if inclusive else (s < t):
            return lab
    return SEV[-1]


L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
L['ts'] = pd.to_numeric(L.total_score, errors='coerce')

print("=" * 72)
print("PART 1 - label consistency: report level vs score-derived level")
print("=" * 72)

rows = []
for role in ['development', 'locked_test']:
    d = L[L.evaluation_role == role].dropna(subset=['ts', 'overall_assessment']).copy()
    d['r_strict'] = d.ts.map(lambda s: cut(s, False))
    d['r_incl'] = d.ts.map(lambda s: cut(s, True))
    n = len(d)
    for name, col in [('strict <', 'r_strict'), ('inclusive <=', 'r_incl')]:
        ok = (d[col] == d.overall_assessment).sum()
        print(f"[{role:11s}] n={n:3d}  {name:13s} agree={ok/n:.4f}  conflicts={n-ok}")
    conf = d[d.r_incl != d.overall_assessment]
    for _, r in conf.iterrows():
        rows.append({'case': r.exam_case_id, 'role': role, 'total_score': r.ts,
                     'report_level': r.overall_assessment, 'incl_rule_level': r.r_incl,
                     'strict_rule_level': r.r_strict})

conf_df = pd.DataFrame(rows)
conf_df.to_csv(OUT / 'conflicted_cases.csv', index=False)
print(f"\nirreducible conflicts under the inclusive rule: {len(conf_df)}")
print(conf_df.to_string(index=False) if len(conf_df) else "  (none)")

dev = L[L.evaluation_role == 'development'].dropna(subset=['ts', 'overall_assessment']).copy()
dev['r_incl'] = dev.ts.map(lambda s: cut(s, True))
print("\ndev level counts - REPORT label (authoritative) vs inclusive rule:")
cmp = pd.DataFrame({'report': dev.overall_assessment.value_counts(),
                    'incl_rule': dev.r_incl.value_counts()}).reindex(SEV).fillna(0).astype(int)
print(cmp.to_string())

print("\n" + "=" * 72)
print("PART 2 - video coverage (EXP-B feasibility gate)")
print("=" * 72)

vid = collections.defaultdict(list)
for f in glob.glob(f'{DATA}/recovered_archive*/*/*.avi'):
    vid['/'.join(f.split('/')[-3:-1])].append(f)

L['n_video'] = L.exam_case_id.map(lambda c: len(vid.get(c, [])))
for role in ['development', 'locked_test']:
    d = L[L.evaluation_role == role]
    hv = (d.n_video > 0).sum()
    print(f"[{role:11s}] n={len(d):3d}  with video={hv:3d} ({hv/len(d):.1%})  without={len(d)-hv}")

devv = dev.merge(L[['exam_case_id', 'n_video']], on='exam_case_id', how='left')
sub = devv[devv.n_video > 0]
print(f"\ndev cases with video: {len(sub)}")
print("\nflow-related label distribution, video subset vs full dev:")
for f in ['flow_state', 'rbc_aggregation', 'microthrombus', 'blood_color']:
    if f not in L.columns:
        continue
    full = dev[f].value_counts(dropna=False)
    vsub = sub[f].value_counts(dropna=False)
    print(f"\n  {f}:")
    for k in full.index:
        print(f"    {str(k)[:22]:24s} full={full.get(k,0):4d}  video={vsub.get(k,0):4d}")
    nz = (vsub[vsub.index.notna()] >= 5).sum()
    print(f"    -> classes with >=5 cases in video subset: {nz}")

print("\ndev level distribution in video subset (is it still spread?):")
print(sub.overall_assessment.value_counts().reindex(SEV).fillna(0).astype(int).to_string())

print("\n" + "=" * 72)
print("PART 3 - date groups (EXP-D leakage sensitivity)")
print("=" * 72)

cd = collections.defaultdict(set)
for f in glob.glob(f'{DATA}/recovered_archive*/*/rep_[0-9]*.jpg.jpg'):
    m = re.match(r'rep_(\d{6})', os.path.basename(f))
    if m:
        cd['/'.join(f.split('/')[-3:-1])].add(m.group(1))

lat = collections.defaultdict(set)
for f in glob.glob(f'{DATA}/recovered_archive*/*/rep_*手*.jpg.jpg'):
    b = os.path.basename(f)
    if not re.search(r'�', b):
        lat['/'.join(f.split('/')[-3:-1])].add(b.replace('rep_', '').replace('.jpg.jpg', '').strip())

recs = []
for c in L.exam_case_id:
    ds = sorted(cd.get(c, []))
    ls = sorted(lat.get(c, []))
    recs.append({'case': c, 'dates': '|'.join(ds), 'n_dates': len(ds),
                 'laterality': '|'.join(ls),
                 'group_key': (ds[0] if ds else 'nodate') + '#' + (ls[0] if ls else 'nolat')})
g = pd.DataFrame(recs).merge(
    L[['exam_case_id', 'evaluation_role', 'development_fold', 'ts', 'overall_assessment']],
    left_on='case', right_on='exam_case_id').drop(columns='exam_case_id')
g.to_csv(OUT / 'case_date_groups.csv', index=False)

print(f"cases with a usable date: {(g.n_dates > 0).sum()}/{len(g)}")
print(f"cases with >1 date (possible repeat visit): {(g.n_dates > 1).sum()}")
gs = g.groupby('group_key').size()
print(f"\ndate#laterality groups: {len(gs)}  (vs {len(g)} cases)")
print("group size distribution:")
print(gs.value_counts().sort_index().to_string())
multi = gs[gs > 1].index
straddle = sum(g[g.group_key == k].evaluation_role.nunique() > 1 for k in multi)
print(f"\ngroups with >1 case: {len(multi)}")
print(f"  ...of which straddle development/locked_test: {straddle}")

summary = {
    'conflicts_inclusive': int(len(conf_df)),
    'conflicted_cases': conf_df.case.tolist(),
    'dev_with_video': int((devv.n_video > 0).sum()),
    'dev_total': int(len(devv)),
    'date_groups': int(len(gs)),
    'groups_multi': int(len(multi)),
    'groups_straddling_split': int(straddle),
}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(f"\nsaved -> {OUT}")
