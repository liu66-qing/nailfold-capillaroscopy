"""EXP-F final: is the periloop residual a SYSTEMATIC extraction bug (fixable)
or RANDOM label noise (a floor)? Then consolidate the 0.600 MAE budget.

Systematic hypotheses tested:
 H1 row-shift: the OCR score column is misaligned by one row, so a field's score
    lands on its neighbour. Test whether periloop_score is reproduced by summing
    field scores with the periloop members cyclically shifted by +/-1.
 H2 archive/layout effect: residual concentrated in one archive or report layout.
 H3 predictability: fit the abs residual on all available covariates (statuses,
    confidences, archive, field values). If nothing predicts it, it is random.

Randomness evidence: mean signed residual vs sd, sign balance, and whether the
same raw field value gets inconsistent scores across cases (range_by_value).
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
    presid = (d.periloop_score - sc[peri].fillna(0.0).sum(axis=1)).round(4)
    tresid = (d.total_score - sc.fillna(0.0).sum(axis=1)).round(4)
    base = float(tresid.abs().mean())
    out['baseline_total_mae'] = base

    # ---- H1 row-shift within the periloop group
    sh = {}
    for k in (-2, -1, 1, 2):
        order = peri[k:] + peri[:k]
        shifted = sc[order].fillna(0.0)
        shifted.columns = peri
        e = (d.periloop_score - shifted.sum(axis=1)).abs()
        sh[f'shift_{k}'] = {'mae': float(e.mean()), 'exact_rate': float((e <= 0.05).mean())}
    sh['shift_0'] = {'mae': float(presid.abs().mean()),
                     'exact_rate': float((presid.abs() <= 0.05).mean())}
    out['H1_row_shift'] = sh

    # ---- H2 archive / layout
    h2 = {}
    for key in ['archive_id', 'report_present_field_count', 'report_copy_count']:
        if key in d.columns:
            g = d.assign(e=tresid.abs()).groupby(key).e.agg(['mean', 'count'])
            h2[key] = {str(k): {'mae': float(v['mean']), 'n': int(v['count'])}
                       for k, v in g.iterrows()}
    out['H2_archive_layout'] = h2

    # ---- H3 predictability of |residual| from covariates
    feats = pd.DataFrame(index=d.index)
    for f in allf:
        c = f + '__confidence'
        if c in d.columns:
            feats['conf_' + f] = pd.to_numeric(d[c], errors='coerce')
        s = f + '__status'
        if s in d.columns:
            feats['hc_' + f] = d[s].astype(str).str.contains('high_consensus').astype(int)
        if f in sc.columns:
            feats['sc_' + f] = pd.to_numeric(sc[f], errors='coerce')
    feats['archive'] = pd.Categorical(d.archive_id).codes
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)
    y = tresid.abs().values
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1)
        r2 = cross_val_score(rf, feats.values, y, cv=5, scoring='r2')
        mae_cv = -cross_val_score(rf, feats.values, y, cv=5,
                                  scoring='neg_mean_absolute_error')
        rf.fit(feats.values, y)
        imp = sorted(zip(feats.columns, rf.feature_importances_),
                     key=lambda t: -t[1])[:12]
        out['H3_predictability'] = {
            'cv_r2_mean': float(r2.mean()), 'cv_r2_per_fold': [float(x) for x in r2],
            'cv_mae': float(mae_cv.mean()),
            'baseline_mae_predict_mean': float(np.abs(y - y.mean()).mean()),
            'top_features': [{'f': f, 'imp': float(i)} for f, i in imp]}
    except Exception as e:
        out['H3_predictability'] = {'error': str(e)}

    # ---- randomness signature
    nz = presid[presid.abs() > 0.05]
    out['randomness_signature'] = {
        'periloop_pct_nonzero': float((presid.abs() > 0.05).mean()),
        'periloop_mean_signed': float(presid.mean()),
        'periloop_sd': float(presid.std()),
        'periloop_nonzero_pct_positive': float((nz > 0).mean()),
        'periloop_nonzero_n': int(len(nz)),
        'total_mean_signed': float(tresid.mean()),
        'total_sd': float(tresid.std()),
        'sign_test_p_approx': None,
    }
    # binomial two-sided p for sign balance among non-zero periloop residuals
    try:
        from scipy import stats
        k = int((nz > 0).sum())
        out['randomness_signature']['sign_test_p_approx'] = float(
            stats.binomtest(k, len(nz), 0.5).pvalue)
        # is |residual| independent of mean_label_confidence?
        mc = pd.to_numeric(d.mean_label_confidence, errors='coerce')
        ok = mc.notna()
        out['randomness_signature']['spearman_conf_vs_abserr_p'] = float(
            stats.spearmanr(mc[ok], tresid.abs()[ok]).pvalue)
    except Exception as e:
        out['randomness_signature']['scipy_error'] = str(e)

    # ---- FINAL BUDGET -------------------------------------------------------
    gr = {g: sc[groups[g]].fillna(0.0).sum(axis=1) for g in groups}
    budget = {}
    # (1) unscoreable/no-rule fields: give every unscoreable cell its field mode
    mode_sc = {f: (float(sc[f].dropna().mode().iloc[0]) if sc[f].notna().any() else 0.0)
               for f in allf if f in sc.columns}
    filled = sc.copy()
    for f in filled.columns:
        filled[f] = filled[f].fillna(mode_sc[f])
    budget['c1_impute_unscoreable_with_mode'] = float(
        base - (filled.sum(axis=1) - d.total_score).abs().mean())
    # (1b) the two no-rule fields specifically: best constant per field
    b = base
    best = {}
    for f in ['output_input_ratio', 'flow_speed_um_s']:
        cand = np.arange(0, 1.55, 0.05)
        errs = [float((sc.fillna(0.0).sum(axis=1) + c - d.total_score).abs().mean()) for c in cand]
        best[f] = {'best_constant': float(cand[int(np.argmin(errs))]),
                   'mae_at_best': float(min(errs)), 'recovered': float(b - min(errs))}
    budget['c1b_best_constant_for_no_rule_fields'] = best
    # (2) rule incompleteness: already measured ~0
    # (3) per-group perfect
    for g in groups:
        alt = sum(d[g2] if g2 == g else gr[g2] for g2 in groups)
        budget[f'c3_{g}_perfect'] = float(base - (alt - d.total_score).abs().mean())
    # (4) all groups perfect = subscore sum (residual = pure total_score inconsistency)
    subsum = d[['morphology_score', 'flow_score', 'periloop_score']].sum(axis=1, min_count=3)
    budget['c4_all_subscores_perfect_residual_mae'] = float(
        (subsum - d.total_score).abs().dropna().mean())
    out['budget'] = budget

    (OUT / 'final_budget.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
