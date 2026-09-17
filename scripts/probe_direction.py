"""EXP-F follow-up 3: for the discrete periloop residuals, which side is wrong --
the extracted FIELD VALUE or the extracted SUBSCORE?

Tests:
 A. Direction. If residual = +1.5 with exudation='无', the report's periloop_score
    contains an exudation contribution the field value denies. Tally the implied
    misread direction (field read as too MILD vs too SEVERE).
 B. Confidence. Is the implicated field's own __confidence lower on residual
    cases than on clean ones? If yes, the field value is the unreliable side.
 C. Rule-tree approximation. Quantify how much of the morphology MAE is just the
    regression-tree leaf imprecision (apex 0.022034, loop 0.219048/0.093069)
    rather than any extraction disagreement: snap those leaves to 1dp and re-measure.
 D. Final additive budget for the 0.600.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RULES = '/root/nailfold/artifacts/labels/score_rules_v3.json'
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
OUT = Path('/root/nailfold/artifacts/experiments/exp_f_label_noise')
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263"}


def score(rule, raw, snap=False):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
        return None
    s = str(raw).strip()
    if rule.get('type') == 'categorical_lookup':
        return rule['mapping'].get(s)
    if rule.get('type') == 'numeric_tree':
        try:
            x = float(s)
        except ValueError:
            return None
        nd = rule['tree']
        while 'threshold' in nd:
            nd = nd['left'] if x < nd['threshold'] else nd['right']
        v = nd.get('value')
        return round(v, 1) if (snap and v is not None) else v
    return None


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float('nan')
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def main():
    R = json.load(open(RULES))
    rules, groups = R['field_rules'], R['groups']
    allf = [f for g in groups.values() for f in g]
    peri = groups['periloop_score']
    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    for c in ['total_score', 'morphology_score', 'flow_score', 'periloop_score']:
        L[c] = pd.to_numeric(L[c], errors='coerce')
    d = L[(L.evaluation_role == 'development') & L.total_score.notna()]
    d = d[~d.exam_case_id.isin(CONFLICTED)].reset_index(drop=True)
    n = len(d)
    out = {'n': n}

    sc = pd.DataFrame({f: [score(rules.get(f, {'type': 'missing'}), v) for v in d[f]]
                       for f in allf if f in d.columns})
    peri_recon = sc[peri].fillna(0.0).sum(axis=1)
    presid = (d.periloop_score - peri_recon).round(4)

    # ---- A/B: attribute each non-zero periloop residual to a field + direction
    rows = []
    for i in range(n):
        r = float(presid.iloc[i]) if not pd.isna(presid.iloc[i]) else np.nan
        if np.isnan(r) or abs(r) <= 0.05:
            continue
        for f in peri:
            cur = sc[f].iloc[i]
            cur = 0.0 if pd.isna(cur) else float(cur)
            mp = rules[f].get('mapping', {})
            for val, v in mp.items():
                if abs((float(v) - cur) - r) <= 0.05:
                    rows.append({'i': i, 'exam_case_id': d.exam_case_id.iloc[i],
                                 'field': f, 'residual': r,
                                 'extracted_value': str(d[f].iloc[i]).strip(),
                                 'implied_value': val,
                                 'extracted_score': cur, 'implied_score': float(v),
                                 'direction': 'field_read_too_mild' if v > cur else 'field_read_too_severe',
                                 'field_conf': pd.to_numeric(
                                     pd.Series([d.get(f + '__confidence', pd.Series([np.nan]*n)).iloc[i]]),
                                     errors='coerce').iloc[0],
                                 'field_status': str(d.get(f + '__status', pd.Series(['']*n)).iloc[i])})
                    break
    A = pd.DataFrame(rows)
    out['direction_tally'] = dict(Counter(A.direction)) if len(A) else {}
    out['field_tally'] = dict(Counter(A.field)) if len(A) else {}
    out['implied_misreads_top'] = (
        A.groupby(['field', 'extracted_value', 'implied_value']).size()
         .sort_values(ascending=False).head(15).to_dict() if len(A) else {})
    out['implied_misreads_top'] = {' -> '.join(map(str, k)): int(v)
                                   for k, v in out['implied_misreads_top'].items()}

    # B: confidence of implicated fields on residual cases vs all cases
    cb = {}
    for f in peri:
        col = f + '__confidence'
        if col not in d.columns:
            continue
        cvals = pd.to_numeric(d[col], errors='coerce')
        flagged = A[A.field == f].i.tolist() if len(A) else []
        mask = d.index.isin(flagged)
        cb[f] = {'n_flagged': int(mask.sum()),
                 'mean_conf_flagged': float(cvals[mask].mean()) if mask.any() else None,
                 'mean_conf_clean': float(cvals[~mask].mean()),
                 'cliffs_delta': cliffs(cvals[mask], cvals[~mask]),
                 'status_counts_flagged': dict(Counter(
                     d.get(f + '__status', pd.Series(['']*n))[mask].astype(str)))}
    out['implicated_field_confidence'] = cb

    # ---- C: tree-approximation share of the morphology error
    sc_snap = pd.DataFrame({f: [score(rules.get(f, {'type': 'missing'}), v, snap=True)
                                for v in d[f]] for f in allf if f in d.columns})
    mres = (d.morphology_score - sc[groups['morphology_score']].fillna(0.0).sum(axis=1)).abs()
    mres_snap = (d.morphology_score - sc_snap[groups['morphology_score']].fillna(0.0).sum(axis=1)).abs()
    out['tree_approximation'] = {
        'morph_mae_raw_leaves': float(mres.mean()),
        'morph_mae_snapped_to_1dp': float(mres_snap.mean()),
        'recovered_by_snapping': float(mres.mean() - mres_snap.mean()),
        'morph_pct_err_below_0.25': float((mres <= 0.25).mean()),
        'morph_mae_from_cases_below_0.25': float(mres[mres <= 0.25].sum() / n),
        'morph_mae_from_cases_above_0.25': float(mres[mres > 0.25].sum() / n),
    }
    tot_raw = (sc.fillna(0.0).sum(axis=1) - d.total_score).abs()
    tot_snap = (sc_snap.fillna(0.0).sum(axis=1) - d.total_score).abs()
    out['total_mae_snap_leaves'] = {'before': float(tot_raw.mean()),
                                   'after': float(tot_snap.mean()),
                                   'recovered': float(tot_raw.mean() - tot_snap.mean())}

    # ---- D: additive budget. Replace each group's reconstruction by the report's
    # own subscore, one group at a time, and see how much total MAE drops.
    gr = {g: sc[groups[g]].fillna(0.0).sum(axis=1) for g in groups}
    budget = {}
    for g in groups:
        alt = sum(d[g2] if g2 == g else gr[g2] for g2 in groups)
        e = (alt - d.total_score).abs()
        budget[g] = {'total_mae_if_group_perfect': float(e.mean()),
                     'recovered': float(tot_raw.mean() - e.mean())}
    out['group_repair_budget'] = budget

    # combined: fix periloop only, using snapped leaves too
    alt = d.periloop_score + sc_snap[groups['morphology_score']].fillna(0.0).sum(axis=1) \
        + sc_snap[groups['flow_score']].fillna(0.0).sum(axis=1)
    e = (alt - d.total_score).abs()
    out['combined_periloop_perfect_plus_snap'] = {
        'mae': float(e.mean()), 'recovered': float(tot_raw.mean() - e.mean()),
        'exact_rate': float((e <= 0.05).mean())}

    A.to_csv(OUT / 'periloop_implicated_fields.csv', index=False, encoding='utf-8')
    (OUT / 'direction.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
