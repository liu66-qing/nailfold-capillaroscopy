"""Map each case to its exam date (from rep_<YYMMDD><idx>.jpg filenames) and
check whether dates recur across cases / straddle dev vs locked_test folds."""
import re, glob, os, collections
import pandas as pd

D = '/root/nailfold/data'

case_dates = collections.defaultdict(set)
for f in glob.glob(f'{D}/recovered_archive*/*/rep_[0-9]*.jpg.jpg'):
    case = '/'.join(f.split('/')[-3:-1])
    m = re.match(r'rep_(\d{6})(\d*)', os.path.basename(f))
    if m:
        case_dates[case].add(m.group(1))

print(f"cases with a date stem: {len(case_dates)}")
multi = {c: d for c, d in case_dates.items() if len(d) > 1}
print(f"cases with >1 distinct date stem: {len(multi)}")
for c, d in list(multi.items())[:5]:
    print(f"  {c}: {sorted(d)}")

# invert: date -> cases
date_cases = collections.defaultdict(set)
for c, ds in case_dates.items():
    for d in ds:
        date_cases[d].add(c)

print(f"\ndistinct dates: {len(date_cases)}")
shared = {d: cs for d, cs in date_cases.items() if len(cs) > 1}
print(f"dates covering >1 case: {len(shared)} (covering {sum(len(v) for v in shared.values())} cases)")

L = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                dtype={'exam_case_id': str})
role = dict(zip(L.exam_case_id, L.evaluation_role))
fold = dict(zip(L.exam_case_id, L.development_fold))
ts = dict(zip(L.exam_case_id, pd.to_numeric(L.total_score, errors='coerce')))
oa = dict(zip(L.exam_case_id, L.overall_assessment))

straddle_role, straddle_fold = [], []
for d, cs in sorted(shared.items()):
    cs = sorted(c for c in cs if c in role)
    if len(cs) < 2:
        continue
    roles = {role[c] for c in cs}
    folds = {fold[c] for c in cs if pd.notna(fold[c])}
    if len(roles) > 1:
        straddle_role.append((d, cs))
    if len(folds) > 1:
        straddle_fold.append((d, cs))

print(f"\n*** shared dates straddling development vs locked_test: {len(straddle_role)}")
for d, cs in straddle_role[:15]:
    print(f"  {d}: " + ", ".join(f"{c}[{role[c]}/ts={ts.get(c)}/{oa.get(c)}]" for c in cs))

print(f"\n*** shared dates straddling >1 development_fold: {len(straddle_fold)}")
for d, cs in straddle_fold[:15]:
    print(f"  {d}: " + ", ".join(f"{c}[f{fold.get(c)}/ts={ts.get(c)}]" for c in cs))

print("\nall shared dates (date -> cases, labels):")
for d, cs in sorted(shared.items())[:25]:
    cs = sorted(c for c in cs if c in role)
    print(f"  {d}: " + ", ".join(f"{c}(ts={ts.get(c)})" for c in cs))
