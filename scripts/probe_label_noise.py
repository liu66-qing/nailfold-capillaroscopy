"""EXP-F: attribute the 0.600 total_score reconstruction MAE to causes.

Read-only forensic analysis. Trains nothing, writes nothing outside OUT.

Causes tested:
  1 group localisation  -- do subscores sum to total? does each subscore equal
                           the sum of its own group's field scores? Per-group MAE
                           isolates which group's rules are broken, and tells us
                           whether the two "missing"-rule fields
                           (output_input_ratio, flow_speed_um_s) carry any weight
                           in the real reports at all.
  2 rule incompleteness -- per field, which raw values fail to score, with counts.
  3 OCR extraction error -- error cross-tabbed against __status / __confidence,
                           with effect sizes (Cliff's delta, rank-biserial,
                           Spearman), not just directions.
  4 field attribution   -- for the >1.0 cases, which single field's score change
                           could close the gap.

Also: what assessment_rule actually says about cut points and inclusivity,
and whether total_score is more self-consistent than its own field values
(via range_by_value: the same raw value scoring differently across reports).
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RULES = '/root/nailfold/artifacts/labels/score_rules_v3.json'
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
OUT = Path('/root/nailfold/artifacts/experiments/exp_f_label_noise')
# Excluded by EXP-E as label-conflicted; kept out so numbers stay comparable.
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263"}
TOL = 0.05


def field_score(rule, raw):
    """Score one field value. Returns (score, reason) where score is None on failure."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
        return None, 'empty'
    s = str(raw).strip()
    t = rule.get('type')
    if t == 'missing':
        return None, 'no_rule'
    if t == 'categorical_lookup':
        v = rule['mapping'].get(s)
        return (v, 'ok') if v is not None else (None, 'unmapped')
    if t == 'numeric_tree':
        try:
            x = float(s)
        except ValueError:
            return None, 'non_numeric'
        node = rule['tree']
        while 'threshold' in node:
            node = node['left'] if x < node['threshold'] else node['right']
        return node.get('value'), 'ok'
    return None, 'unknown_rule_type'


def cliffs_delta(a, b):
    """P(a>b) - P(a<b). Non-parametric effect size, range [-1,1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    gt = lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return float((gt - lt) / (len(a) * len(b)))


def spearman(x, y):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return float('nan')
    return float(x[m].rank().corr(y[m].rank()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.load(open(RULES))
    rules, groups = R['field_rules'], R['groups']
    all_fields = [f for g in groups.values() for f in g]
    rep = {}

    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    for c in ['total_score', 'morphology_score', 'flow_score', 'periloop_score',
              'mean_label_confidence', 'low_confidence_field_count']:
        L[c] = pd.to_numeric(L[c], errors='coerce')
    d = L[(L.evaluation_role == 'development') & L.total_score.notna()].copy()
    d = d[~d.exam_case_id.isin(CONFLICTED)].reset_index(drop=True)
    n = len(d)
    rep['n_dev_cases'] = n

    # ---- per-field oracle scores + failure reasons ----------------------------
    sc = pd.DataFrame(index=d.index, dtype=float)
    why = {}
    for f in all_fields:
        if f not in d.columns:
            sc[f] = np.nan
            why[f] = ['absent_column'] * n
            continue
        pairs = [field_score(rules.get(f, {'type': 'missing'}), v) for v in d[f]]
        sc[f] = [p[0] for p in pairs]
        why[f] = [p[1] for p in pairs]

    # ================= CAUSE 1: group localisation ============================
    # 1a. Do the three subscores sum to total_score, as recorded by OCR?
    sub = ['morphology_score', 'flow_score', 'periloop_score']
    sub_sum = d[sub].sum(axis=1, min_count=3)
    add_err = (sub_sum - d.total_score).abs()
    rep['additivity_subscores_vs_total'] = {
        'n': int(add_err.notna().sum()),
        'exact_rate': float((add_err <= TOL).mean()),
        'mae': float(add_err.mean()),
        'max': float(add_err.max()),
        'n_off_gt_1': int((add_err > 1.0).sum()),
    }

    # 1b. Does each subscore equal the sum of ITS OWN group's field scores?
    #     Two summation conventions: strict (NaN poisons) vs zero-filled (as EXP-E).
    per_group = {}
    grp_recon_zero = {}
    for g, members in groups.items():
        mem = [f for f in members if f in sc.columns]
        recon_zero = sc[mem].fillna(0.0).sum(axis=1)
        recon_strict = sc[mem].sum(axis=1, min_count=len(mem))
        grp_recon_zero[g] = recon_zero
        ez = (recon_zero - d[g]).abs()
        es = (recon_strict - d[g]).abs()
        n_unscore = int(sc[mem].isna().sum(axis=1).gt(0).sum())
        per_group[g] = {
            'n_fields': len(mem),
            'cases_with_any_unscoreable_field': n_unscore,
            'zerofill': {'mae': float(ez.mean()), 'exact_rate': float((ez <= TOL).mean()),
                         'max': float(ez.max()), 'n_off_gt_1': int((ez > 1.0).sum()),
                         'mean_signed': float((recon_zero - d[g]).mean())},
            'strict_complete_only': {'n': int(es.notna().sum()),
                                     'mae': float(es[es.notna()].mean()) if es.notna().any() else None,
                                     'exact_rate': float((es[es.notna()] <= TOL).mean()) if es.notna().any() else None},
        }
    rep['per_group_reconstruction'] = per_group

    # 1c. Do the two no-rule fields carry weight? Compare morphology/flow group
    #     residual on cases where the field is present vs blank in the report.
    noruleinfo = {}
    for f, g in [('output_input_ratio', 'morphology_score'), ('flow_speed_um_s', 'flow_score')]:
        if f not in d.columns:
            continue
        present = d[f].notna() & (d[f].astype(str).str.strip() != '')
        resid = (d[g] - grp_recon_zero[g])  # positive = OCR subscore exceeds recon
        noruleinfo[f] = {
            'n_present_in_report': int(present.sum()),
            'n_blank': int((~present).sum()),
            'group': g,
            'group_residual_mean_when_present': float(resid[present].mean()) if present.any() else None,
            'group_residual_mean_when_blank': float(resid[~present].mean()) if (~present).any() else None,
            'cliffs_delta_present_vs_blank': cliffs_delta(resid[present], resid[~present]),
            'distinct_raw_values': int(d.loc[present, f].astype(str).str.strip().nunique()),
            'top_raw_values': dict(Counter(d.loc[present, f].astype(str).str.strip()).most_common(8)),
        }
    rep['no_rule_fields'] = noruleinfo

    # ---- total reconstruction (reproduce EXP-E, then decompose) ---------------
    recon_total = sc[all_fields].fillna(0.0).sum(axis=1)
    err_signed = recon_total - d.total_score
    err = err_signed.abs()
    rep['total_reconstruction'] = {
        'mae': float(err.mean()), 'exact_rate': float((err <= TOL).mean()),
        'max': float(err.max()), 'n_off_gt_1': int((err > 1.0).sum()),
        'mean_signed_error': float(err_signed.mean()),
        'pct_recon_below_ocr': float((err_signed < -TOL).mean()),
        'pct_recon_above_ocr': float((err_signed > TOL).mean()),
    }
    # Sum of the three per-group MAEs bounds the total MAE (triangle inequality).
    rep['total_reconstruction']['sum_of_group_maes'] = float(
        sum(per_group[g]['zerofill']['mae'] for g in groups))

    # ================= CAUSE 2: rule incompleteness ===========================
    incomplete = {}
    for f in all_fields:
        reasons = Counter(why[f])
        bad_idx = [i for i, r in enumerate(why[f]) if r in ('unmapped', 'non_numeric')]
        badvals = Counter(str(d[f].iloc[i]).strip() for i in bad_idx) if f in d.columns else Counter()
        incomplete[f] = {
            'rule_type': rules.get(f, {}).get('type', 'ABSENT'),
            'coverage': float(sc[f].notna().mean()),
            'reason_counts': dict(reasons),
            'n_distinct_failing_values': len(badvals),
            'failing_values': dict(badvals.most_common(30)),
            'n_cases_failing_recoverably': len(bad_idx),
        }
    rep['rule_incompleteness'] = incomplete
    rep['rule_incompleteness_summary'] = {
        'n_fields_zero_coverage': sum(1 for f in all_fields if incomplete[f]['coverage'] == 0),
        'total_field_cells': n * len(all_fields),
        'cells_empty': sum(Counter(why[f])['empty'] for f in all_fields),
        'cells_no_rule': sum(Counter(why[f])['no_rule'] for f in all_fields),
        'cells_unmapped': sum(Counter(why[f])['unmapped'] for f in all_fields),
        'cells_non_numeric': sum(Counter(why[f])['non_numeric'] for f in all_fields),
    }

    # How much MAE would vanish if every dirty value were cleaned to its field's
    # mode score (an upper bound on what value-normalisation can buy)?
    mode_sc = {f: (float(sc[f].dropna().mode().iloc[0]) if sc[f].notna().any() else 0.0)
               for f in all_fields}
    patched = sc.copy()
    for f in all_fields:
        bad = [i for i, r in enumerate(why[f]) if r in ('unmapped', 'non_numeric')]
        if bad:
            patched.loc[patched.index[bad], f] = mode_sc[f]
    err_patched = (patched[all_fields].fillna(0.0).sum(axis=1) - d.total_score).abs()
    rep['counterfactual_clean_dirty_values'] = {
        'mae_after': float(err_patched.mean()),
        'mae_before': float(err.mean()),
        'mae_recovered': float(err.mean() - err_patched.mean()),
    }

    # ================= CAUSE 3: OCR extraction error ==========================
    # 3a. total_score's own status/confidence vs reconstruction error
    st_tot = d.get('total_score__status', pd.Series([''] * n)).astype(str)
    by_status_total = {}
    for s, idx in st_tot.groupby(st_tot).groups.items():
        e = err.loc[idx]
        rest = err.loc[~err.index.isin(idx)]
        by_status_total[s] = {'n': int(len(e)), 'mae': float(e.mean()),
                             'median': float(e.median()),
                             'cliffs_delta_vs_rest': cliffs_delta(e, rest)}
    rep['error_by_total_score_status'] = by_status_total

    # 3b. per-field status: does a case containing an engine_conflict /
    #     single_engine / qwen_only field have larger reconstruction error?
    flags = {}
    for tag in ['engine_conflict', 'single_engine', 'rapid_layout_missing_qwen_only',
                'invalid_non_numeric_removed', 'derived_from_diameters',
                'redundancy_corrected_from', 'high_consensus', 'missing']:
        cnt = np.zeros(n, dtype=int)
        for f in all_fields:
            col = f + '__status'
            if col in d.columns:
                cnt += d[col].astype(str).str.contains(tag, regex=False, na=False).astype(int)
        has = cnt > 0
        e_has, e_not = err[has], err[~has]
        flags[tag] = {
            'n_cases_with_tag': int(has.sum()),
            'mean_fields_flagged': float(cnt.mean()),
            'mae_with': float(e_has.mean()) if has.any() else None,
            'mae_without': float(e_not.mean()) if (~has).any() else None,
            'cliffs_delta': cliffs_delta(e_has, e_not),
            'spearman_count_vs_err': spearman(cnt, err),
        }
    rep['error_by_field_status_flag'] = flags

    # 3c. confidence predictors
    conf = {}
    for c in ['mean_label_confidence', 'low_confidence_field_count']:
        if c in d.columns:
            conf[c] = {'spearman_vs_abs_err': spearman(d[c], err),
                       'spearman_vs_signed_err': spearman(d[c], err_signed),
                       'n': int(d[c].notna().sum())}
    # low-confidence field count computed from the per-field confidence columns
    lowcnt = np.zeros(n, dtype=int)
    for f in all_fields:
        col = f + '__confidence'
        if col in d.columns:
            lowcnt += (pd.to_numeric(d[col], errors='coerce') < 0.9).fillna(True).astype(int)
    conf['recomputed_low_conf_field_count'] = {
        'spearman_vs_abs_err': spearman(lowcnt, err), 'mean': float(lowcnt.mean())}
    rep['error_by_confidence'] = conf

    # ================= CAUSE 4: which field explains the gap ==================
    big = err > 1.0
    rep['n_cases_gap_gt_1'] = int(big.sum())
    # For each big-gap case, which single field could close it? A field can close
    # the gap if some other value in its own rule's value set shifts its score by
    # the residual. Also record the "unscoreable field" explanation separately.
    attain = {}
    for f in all_fields:
        r = rules.get(f, {})
        if r.get('type') == 'categorical_lookup':
            attain[f] = sorted(set(float(v) for v in r['mapping'].values()))
        elif r.get('type') == 'numeric_tree':
            leaves = []
            stack = [r['tree']]
            while stack:
                nd = stack.pop()
                if 'threshold' in nd:
                    stack += [nd['left'], nd['right']]
                else:
                    leaves.append(float(nd['value']))
            attain[f] = sorted(set(leaves))
        else:
            attain[f] = []

    culprit = Counter()
    culprit_unique = Counter()
    unexplained_cases = []
    detail = []
    for i in np.where(big.values)[0]:
        gap = float(d.total_score.iloc[i] - recon_total.iloc[i])  # need +gap from a field
        cands = []
        for f in all_fields:
            cur = sc[f].iloc[i]
            cur = 0.0 if pd.isna(cur) else float(cur)
            for v in attain[f]:
                if abs((v - cur) - gap) <= TOL:
                    cands.append(f)
                    break
        for f in cands:
            culprit[f] += 1
        if len(cands) == 1:
            culprit_unique[cands[0]] += 1
        if not cands:
            unexplained_cases.append(d.exam_case_id.iloc[i])
        detail.append({
            'exam_case_id': d.exam_case_id.iloc[i],
            'ocr_total': float(d.total_score.iloc[i]),
            'recon_total': float(recon_total.iloc[i]),
            'gap_needed': gap,
            'n_single_field_explanations': len(cands),
            'candidates': cands[:12],
            'unscoreable_fields': [f for f in all_fields if pd.isna(sc[f].iloc[i])],
        })
    rep['gap_attribution'] = {
        'culprit_any_candidate_counts': dict(culprit.most_common()),
        'culprit_unique_explanation_counts': dict(culprit_unique.most_common()),
        'n_cases_no_single_field_explanation': len(unexplained_cases),
        'cases_no_single_field_explanation': unexplained_cases,
    }

    # ================= label self-consistency ================================
    # range_by_value: same raw value observed with DIFFERENT scores in reports.
    # This is irreducible noise in the label, independent of our rules.
    incons = {}
    for f in all_fields:
        rbv = rules.get(f, {}).get('range_by_value', {})
        nz = {k: v for k, v in rbv.items() if v and v > 0}
        if nz:
            incons[f] = nz
    rep['label_self_inconsistency_range_by_value'] = incons
    rep['label_self_inconsistency_summary'] = {
        'n_fields_with_nonzero_range': len(incons),
        'max_single_value_range': float(max((max(v.values()) for v in incons.values()), default=0.0)),
        'sum_of_max_ranges': float(sum(max(v.values()) for v in incons.values())),
    }
    # Expected |error| contribution if each case independently hits the observed
    # range for its own value, with the range split evenly (rough magnitude only).
    per_case_range = np.zeros(n)
    for f in all_fields:
        rbv = rules.get(f, {}).get('range_by_value', {})
        if f in d.columns and rbv:
            per_case_range += d[f].astype(str).str.strip().map(rbv).fillna(0.0).values
    rep['label_self_inconsistency_per_case'] = {
        'mean_summed_range': float(per_case_range.mean()),
        'median': float(np.median(per_case_range)),
        'pct_cases_with_any_range': float((per_case_range > 0).mean()),
        'spearman_range_vs_abs_err': spearman(per_case_range, err),
    }

    # ================= assessment_rule verification ==========================
    ar = R['assessment_rule']
    thr, labs = ar['thresholds'], ar['labels']
    oa = d.get('overall_assessment', pd.Series([''] * n)).astype(str).str.strip()

    def apply_cuts(x, inclusive):
        for t, lab in zip(thr, labs):
            if (x <= t) if inclusive else (x < t):
                return lab
        return labs[-1]

    res_ar = {'raw_rule': ar}
    for name, inc in [('inclusive_le', True), ('exclusive_lt', False)]:
        pred = d.total_score.map(lambda x: apply_cuts(x, inc))
        m = oa.isin(labs)
        res_ar[name] = {'agreement_on_labelled': float((pred[m] == oa[m]).mean()),
                        'n': int(m.sum()),
                        'n_disagree': int((pred[m] != oa[m]).sum())}
    res_ar['boundary_cases'] = {}
    for t in thr:
        at = d.total_score.sub(t).abs() <= TOL
        if at.any():
            res_ar['boundary_cases'][str(t)] = {
                'n': int(at.sum()),
                'assessments_seen': dict(Counter(oa[at]))}
    res_ar['overall_assessment_value_counts'] = dict(Counter(oa))
    rep['assessment_rule_check'] = res_ar

    # ---- write -----------------------------------------------------------------
    (OUT / 'label_noise_report.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    pd.DataFrame(detail).to_csv(OUT / 'gap_gt1_cases.csv', index=False, encoding='utf-8')
    per_case = pd.DataFrame({
        'exam_case_id': d.exam_case_id, 'ocr_total': d.total_score,
        'recon_total': recon_total, 'signed_err': err_signed, 'abs_err': err,
        'n_unscoreable_fields': sc[all_fields].isna().sum(axis=1),
        'summed_value_range': per_case_range,
        'mean_label_confidence': d.get('mean_label_confidence'),
        'low_confidence_field_count': d.get('low_confidence_field_count'),
    })
    for g in groups:
        per_case['ocr_' + g] = d[g]
        per_case['recon_' + g] = grp_recon_zero[g]
        per_case['err_' + g] = (grp_recon_zero[g] - d[g]).abs()
    per_case.to_csv(OUT / 'per_case_errors.csv', index=False, encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
