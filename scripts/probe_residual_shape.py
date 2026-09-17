"""EXP-F follow-up 2: the periloop residual is not a rule bug. What shape is it?

If the residual takes a few discrete values matching individual field score
deltas, it is a mis-EXTRACTION of that field (systematic, fixable at the OCR
step). If it is continuous / unstructured, it is irreducible label noise.

Also tests whether the residual is per-case constant across the three groups
(a whole-row shift) and whether total_score or the subscores are the more
trustworthy anchor.
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


def score(rule, raw):
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
        return nd.get('value')
    return None


def main():
    R = json.load(open(RULES))
    rules, groups = R['field_rules'], R['groups']
    allf = [f for g in groups.values() for f in g]
    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    for c in ['total_score', 'morphology_score', 'flow_score', 'periloop_score']:
        L[c] = pd.to_numeric(L[c], errors='coerce')
    d = L[(L.evaluation_role == 'development') & L.total_score.notna()]
    d = d[~d.exam_case_id.isin(CONFLICTED)].reset_index(drop=True)
    n = len(d)
    out = {'n': n}

    sc = pd.DataFrame({f: [score(rules.get(f, {'type': 'missing'}), v) for v in d[f]]
                       for f in allf if f in d.columns})
    gr = {g: sc[[f for f in m if f in sc.columns]].fillna(0.0).sum(axis=1) for g, m in groups.items()}
    resid = {g: (d[g] - gr[g]).round(4) for g in groups}
    tot_resid = (d.total_score - sc.fillna(0.0).sum(axis=1)).round(4)

    # 1. Discreteness of the periloop residual.
    for g in groups:
        r = resid[g]
        out[f'residual_hist_{g}'] = {str(k): int(v) for k, v in
                                     sorted(Counter(r).items(), key=lambda kv: -kv[1])[:25]}
        out[f'residual_stats_{g}'] = {
            'mean': float(r.mean()), 'sd': float(r.std()),
            'n_distinct': int(r.nunique()),
            'pct_zero': float((r.abs() <= 0.05).mean()),
            'skew': float(r.skew())}

    # 2. Is the periloop residual equal to a single field's score (i.e. a field
    #    value was dropped/added by the extractor)?
    peri = groups['periloop_score']
    matchable = Counter()
    for i in range(n):
        r = float(resid['periloop_score'].iloc[i])
        if abs(r) <= 0.05:
            matchable['exact_ok'] += 1
            continue
        hits = []
        for f in peri:
            cur = sc[f].iloc[i]
            cur = 0.0 if pd.isna(cur) else float(cur)
            vals = list(rules[f].get('mapping', {}).values())
            for v in vals:
                if abs((float(v) - cur) - r) <= 0.05:
                    hits.append(f)
                    break
        matchable['|'.join(sorted(set(hits))) or 'NO_FIELD_EXPLAINS'] += 1
    out['periloop_residual_single_field_match'] = dict(matchable.most_common(20))

    # 3. Is the residual a whole-row shift (correlated across groups)?
    dfres = pd.DataFrame(resid)
    out['residual_corr_between_groups'] = {
        f'{a}~{b}': float(dfres[a].corr(dfres[b]))
        for a, b in [('morphology_score', 'flow_score'),
                     ('morphology_score', 'periloop_score'),
                     ('flow_score', 'periloop_score')]}

    # 4. Which anchor is more self-consistent: total_score or the subscores?
    #    (a) subscore additivity was 97.3% exact -> subscores agree with total.
    #    (b) compare recon-vs-total MAE against recon-vs-(sum of subscores).
    subsum = d[['morphology_score', 'flow_score', 'periloop_score']].sum(axis=1, min_count=3)
    reconsum = sc.fillna(0.0).sum(axis=1)
    out['anchor_comparison'] = {
        'mae_recon_vs_total': float((reconsum - d.total_score).abs().mean()),
        'mae_recon_vs_subscore_sum': float((reconsum - subsum).abs().dropna().mean()),
        'mae_total_vs_subscore_sum': float((d.total_score - subsum).abs().dropna().mean()),
    }

    # 5. Rounding hypothesis: are the reports rounding each subscore to 1 decimal
    #    while the true sum has more precision? Test whether residuals cluster at
    #    multiples of 0.1 and whether error correlates with the count of fields
    #    whose rule value is non-round (apex 0.022034, loop 0.219048/0.093069).
    nonround_fields = []
    for f in allf:
        r = rules.get(f, {})
        vals = []
        if r.get('type') == 'numeric_tree':
            st = [r['tree']]
            while st:
                nd = st.pop()
                if 'threshold' in nd:
                    st += [nd['left'], nd['right']]
                else:
                    vals.append(float(nd['value']))
        if any(abs(v * 10 - round(v * 10)) > 1e-6 for v in vals):
            nonround_fields.append(f)
    out['fields_with_nonround_leaf_values'] = nonround_fields
    nr = sc[[f for f in nonround_fields if f in sc.columns]]
    is_nonround = (nr.apply(lambda c: c.map(
        lambda v: (not pd.isna(v)) and abs(v * 10 - round(v * 10)) > 1e-6))).sum(axis=1)
    e = tot_resid.abs()
    out['nonround_effect'] = {
        'mean_nonround_fields_per_case': float(is_nonround.mean()),
        'mae_when_any_nonround': float(e[is_nonround > 0].mean()) if (is_nonround > 0).any() else None,
        'mae_when_none': float(e[is_nonround == 0].mean()) if (is_nonround == 0).any() else None,
    }
    # residual mod 0.1
    m = (tot_resid.abs() * 100).round() % 10
    out['total_residual_mod_0.1_hist'] = {str(int(k)): int(v) for k, v in sorted(Counter(m).items())}

    # 6. Decompose the 0.600 total MAE into group contributions on the SAME cases,
    #    signed, so contributions add up.
    out['signed_group_contribution_to_total_error'] = {
        g: {'mean_signed': float(resid[g].mean()),
            'mean_abs': float(resid[g].abs().mean()),
            'share_of_total_abs_err': float(resid[g].abs().sum() / tot_resid.abs().sum())}
        for g in groups}
    out['total_abs_err_mean'] = float(tot_resid.abs().mean())

    # 7. Would rounding each group's reconstruction to 1dp fix it?
    r1 = sum(gr[g].round(1) for g in groups)
    out['round_each_group_to_1dp'] = {
        'mae': float((r1 - d.total_score).abs().mean()),
        'exact_rate': float(((r1 - d.total_score).abs() <= 0.05).mean())}

    (OUT / 'residual_shape.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
