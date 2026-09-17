#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only clinical/screening metrics for nailfold total_score predictions.

No training, no GPU. Consumes existing OOF prediction CSVs.
"""
import json
import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
RULES = '/root/nailfold/artifacts/labels/score_rules_v3.json'
OUTDIR = '/root/nailfold/artifacts/experiments/exp_h_clinical_metrics'
ARMS = {
    'A_reg_only_raw': '/root/nailfold/artifacts/experiments/exp_c_v2_g0/A_reg_only_raw_oof.csv',
    'B_reg_only_znorm': '/root/nailfold/artifacts/experiments/exp_c_v2_g0/B_reg_only_znorm_oof.csv',
    'C_multi_balanced': '/root/nailfold/artifacts/experiments/exp_c_v2_g1/C_multi_balanced_oof.csv',
}
LEVELS = ['正常', '大致正常', '轻度异常', '中度异常', '重度异常']
L2I = {l: i for i, l in enumerate(LEVELS)}

OUT = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    OUT.append(s)


def cut_inclusive(score, thr):
    """assessment_rule: ordered_thresholds, INCLUSIVE (<=)."""
    for i, t in enumerate(thr):
        if score <= t:
            return i
    return len(thr)


def binary_stats(y_true_pos, y_pred_pos):
    tp = int(np.sum(y_true_pos & y_pred_pos))
    fp = int(np.sum(~y_true_pos & y_pred_pos))
    fn = int(np.sum(y_true_pos & ~y_pred_pos))
    tn = int(np.sum(~y_true_pos & ~y_pred_pos))
    n = tp + fp + fn + tn
    d = lambda a, b: (a / b) if b else float('nan')
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n': n,
        'prevalence': d(tp + fn, n),
        'sensitivity': d(tp, tp + fn),
        'specificity': d(tn, tn + fp),
        'ppv': d(tp, tp + fp),
        'npv': d(tn, tn + fn),
        'balanced_acc': (d(tp, tp + fn) + d(tn, tn + fp)) / 2.0,
    }


def kappa(a, b, weights):
    """Weighted kappa on 0..4 ordinal labels."""
    k = 5
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O = O / O.sum()
    ra = O.sum(axis=1)
    cb = O.sum(axis=0)
    E = np.outer(ra, cb)
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (abs(i - j) ** 2) / ((k - 1) ** 2) if weights == 'quadratic' \
                else abs(i - j) / (k - 1)
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else float('nan')


def c_index(y, s, max_pairs=None):
    """Concordance on continuous score, ties in y skipped, ties in s = 0.5."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n = len(y)
    conc = 0.0
    tot = 0
    for i in range(n):
        dy = y[i + 1:] - y[i]
        ds = s[i + 1:] - s[i]
        m = dy != 0
        if not m.any():
            continue
        dy = dy[m]
        ds = ds[m]
        tot += len(dy)
        agree = np.sign(dy) == np.sign(ds)
        conc += float(agree.sum()) + 0.5 * float((ds == 0).sum())
    return conc / tot if tot else float('nan')


def quantile_thresholds(pred, true_levels):
    """Re-fit 4 cuts on predicted score dist by matching TRUE level prevalences."""
    counts = np.array([np.sum(true_levels == i) for i in range(5)], float)
    cum = np.cumsum(counts / counts.sum())[:4]
    ps = np.sort(np.asarray(pred, float))
    thr = []
    for q in cum:
        thr.append(float(np.quantile(ps, min(max(q, 0.0), 1.0))))
    # enforce monotone non-decreasing
    for i in range(1, 4):
        if thr[i] < thr[i - 1]:
            thr[i] = thr[i - 1]
    return thr


# ---------------- field-score reconstruction ----------------
def eval_numeric_tree(node, x):
    while 'value' not in node:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        node = node['left'] if x <= node['threshold'] else node['right']
    return float(node['value'])


def field_score(rule, val):
    if rule['type'] == 'categorical_lookup':
        if val is None:
            return None
        return rule['mapping'].get(str(val).strip())
    if rule['type'] == 'numeric_tree':
        try:
            x = float(val)
        except (TypeError, ValueError):
            return None
        if math.isnan(x):
            return None
        return eval_numeric_tree(rule['tree'], x)
    return None


SCREEN_CUTS = [
    ('any_abnormality (>=轻度异常)', 2),
    ('moderate_or_worse (>=中度异常)', 3),
    ('severe (==重度异常)', 4),
]


def full_block(name, true_lv, pred_lv, true_ts, pred_ts, do_ordinal=True):
    """Metrics 1-5 for one (true, pred) level pairing."""
    res = {}
    p('')
    p('#' * 74)
    p('## ' + name)
    p('#' * 74)
    n = len(true_lv)
    p('n = %d' % n)

    # --- 1 screening
    p('')
    p('[1] SCREENING / TRIAGE BINARY CUTS')
    p('%-32s %6s %6s %6s %6s %6s %6s  %s' % (
        'cut', 'prev', 'sens', 'spec', 'PPV', 'NPV', 'BA', '(tp,fp,fn,tn)'))
    res['screening'] = {}
    for label, k in SCREEN_CUTS:
        st = binary_stats(true_lv >= k, pred_lv >= k)
        res['screening'][label] = st
        p('%-32s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  (%d,%d,%d,%d)' % (
            label, st['prevalence'], st['sensitivity'], st['specificity'],
            st['ppv'], st['npv'], st['balanced_acc'],
            st['tp'], st['fp'], st['fn'], st['tn']))

    # --- 2 severe miss
    p('')
    p('[2] SEVERE-MISS RATE (true 重度异常 cases)')
    sev = true_lv == 4
    nsev = int(sev.sum())
    res['severe_miss'] = {'n_true_severe': nsev}
    if nsev:
        hard = int(np.sum(pred_lv[sev] <= 1))
        soft = int(np.sum(pred_lv[sev] <= 2))
        res['severe_miss'].update({
            'sent_away_as_fine_n': hard, 'sent_away_as_fine_rate': hard / nsev,
            'pred_le_mild_n': soft, 'pred_le_mild_rate': soft / nsev,
            'caught_as_severe_n': int(np.sum(pred_lv[sev] == 4)),
        })
        p('  n true 重度异常                    = %d' % nsev)
        p('  predicted 正常/大致正常 (sent away) = %d  (%.1f%%)' % (hard, 100 * hard / nsev))
        p('  predicted <= 轻度异常 (soft miss)   = %d  (%.1f%%)' % (soft, 100 * soft / nsev))
        p('  predicted 重度异常 exactly (caught) = %d  (%.1f%%)' % (
            res['severe_miss']['caught_as_severe_n'],
            100 * res['severe_miss']['caught_as_severe_n'] / nsev))
    else:
        p('  no true severe cases')

    # --- 3 accuracy
    p('')
    p('[3] LEVEL ACCURACY')
    exact = float(np.mean(pred_lv == true_lv))
    w1 = float(np.mean(np.abs(pred_lv - true_lv) <= 1))
    res['exact_level_acc'] = exact
    res['within1_level_acc'] = w1
    p('  exact level accuracy   = %.4f  (%.1f%%)' % (exact, 100 * exact))
    p('  within-+/-1 accuracy   = %.4f  (%.1f%%)' % (w1, 100 * w1))
    p('  mean abs level error   = %.4f' % float(np.mean(np.abs(pred_lv - true_lv))))

    # --- 4 confusion
    p('')
    p('[4] 5x5 CONFUSION (rows=true, cols=pred) + per-true-level recall')
    cm = np.zeros((5, 5), int)
    for t, q in zip(true_lv, pred_lv):
        cm[t, q] += 1
    p('%-10s %s %8s %8s' % ('true\\pred', ''.join('%10s' % l for l in LEVELS),
                            'n', 'recall'))
    rec = {}
    for i in range(5):
        rn = cm[i].sum()
        r = cm[i, i] / rn if rn else float('nan')
        rec[LEVELS[i]] = {'n': int(rn), 'recall': r,
                          'row': [int(v) for v in cm[i]]}
        p('%-10s %s %8d %8s' % (LEVELS[i], ''.join('%10d' % v for v in cm[i]),
                                rn, ('%.3f' % r) if rn else '-'))
    p('%-10s %s %8d' % ('pred n', ''.join('%10d' % v for v in cm.sum(axis=0)),
                        cm.sum()))
    res['confusion'] = [[int(v) for v in row] for row in cm]
    res['per_true_level_recall'] = rec

    # --- 5 ordinal
    if do_ordinal:
        p('')
        p('[5] ORDINAL METRICS')
        qwk = kappa(true_lv, pred_lv, 'quadratic')
        lwk = kappa(true_lv, pred_lv, 'linear')
        ci = c_index(true_ts, pred_ts)
        sp = spearmanr(true_ts, pred_ts).correlation
        mae = float(np.mean(np.abs(np.asarray(true_ts, float) - np.asarray(pred_ts, float))))
        res.update({'qwk': qwk, 'lwk': lwk, 'c_index': ci,
                    'spearman_total_score': float(sp), 'mae_total_score': mae})
        p('  QWK (quadratic)  = %.4f' % qwk)
        p('  LWK (linear)     = %.4f' % lwk)
        p('  C-index (cont.)  = %.4f' % ci)
        p('  Spearman         = %.4f' % sp)
        p('  MAE total_score  = %.4f' % mae)
    return res


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rules = json.load(open(RULES, encoding='utf-8'))
    ar = rules['assessment_rule']
    THR = list(ar['thresholds'])
    assert ar['labels'] == LEVELS, ar['labels']
    p('=' * 74)
    p('CLINICAL / SCREENING METRICS - nailfold total_score  (read-only analysis)')
    p('=' * 74)
    p('assessment_rule thresholds = %s  (INCLUSIVE <=)' % THR)
    p('labels = %s' % LEVELS)
    p('')
    p('FILES USED')
    p('  ground truth : %s' % MANIFEST)
    p('                 columns: exam_case_id (str), overall_assessment (AUTHORITATIVE'
      ' true level), total_score, evaluation_role, 21 field-value columns')
    p('  rules        : %s  (assessment_rule, field_rules)' % RULES)
    for k, v in ARMS.items():
        p('  pred %-17s: %s' % (k, v))
    p('                 columns: case,true_ts,pred_ts,true_level,pred_level_from_score'
      ' (+pred_level_classify for C)')

    man = pd.read_csv(MANIFEST, dtype={'exam_case_id': str}, low_memory=False)
    dev = man[man['evaluation_role'] == 'development'].copy()
    p('')
    p('manifest rows total=%d | development=%d | locked_test=%d (NEVER touched)' % (
        len(man), len(dev), int((man['evaluation_role'] == 'locked_test').sum())))
    gt = dev.set_index('exam_case_id')

    results = {'files': {'manifest': MANIFEST, 'rules': RULES, 'arms': ARMS},
               'assessment_thresholds_inclusive': THR, 'arms': {}}

    # ---- assemble aligned frame from best arm's case list
    base = pd.read_csv(ARMS['B_reg_only_znorm'], dtype={'case': str})
    cases = base['case'].tolist()
    missing = [c for c in cases if c not in gt.index]
    if missing:
        p('WARNING: %d OOF cases absent from development manifest rows: %s'
          % (len(missing), missing[:5]))
    cases = [c for c in cases if c in gt.index]
    sub = gt.loc[cases]

    true_lv = np.array([L2I[str(x).strip()] for x in sub['overall_assessment']])
    true_ts = sub['total_score'].astype(float).values
    p('aligned cases used = %d' % len(cases))

    # sanity: authoritative level vs level re-derived from true total_score
    derived = np.array([cut_inclusive(v, THR) for v in true_ts])
    agree = float(np.mean(derived == true_lv))
    p('sanity: authoritative overall_assessment vs inclusive-cut of TRUE total_score'
      ' agreement = %.4f (%d/%d) -> confirms inclusive cut; authoritative column used'
      ' for truth regardless' % (agree, int(np.sum(derived == true_lv)), len(cases)))
    p('true level distribution: %s' % {LEVELS[i]: int(np.sum(true_lv == i)) for i in range(5)})
    p('true_ts mean=%.3f sd=%.3f median=%.3f' % (true_ts.mean(), true_ts.std(ddof=1),
                                                 float(np.median(true_ts))))
    results['n_cases'] = len(cases)
    results['true_level_counts'] = {LEVELS[i]: int(np.sum(true_lv == i)) for i in range(5)}
    results['inclusive_cut_selfcheck_agreement'] = agree

    # ================= BASELINES =================
    p('')
    p('=' * 74)
    p('BASELINES')
    p('=' * 74)
    med = float(np.median(true_ts))
    const_pred = np.full(len(cases), med)
    const_lv = np.array([cut_inclusive(med, THR)] * len(cases))
    p('constant-median baseline: total_score=%.3f -> level=%s' % (med, LEVELS[const_lv[0]]))
    maj = int(np.bincount(true_lv, minlength=5).argmax())
    maj_lv = np.full(len(cases), maj)
    p('majority-level baseline : level=%s (n=%d)' % (LEVELS[maj], int(np.sum(true_lv == maj))))

    results['baselines'] = {}
    results['baselines']['constant_median'] = full_block(
        'BASELINE constant-median total_score = %.2f' % med,
        true_lv, const_lv, true_ts, const_pred)
    results['baselines']['majority_level'] = full_block(
        'BASELINE majority level = %s' % LEVELS[maj],
        true_lv, maj_lv, true_ts, const_pred, do_ordinal=False)

    # ================= ARMS =================
    for arm, path in ARMS.items():
        df = pd.read_csv(path, dtype={'case': str}).set_index('case')
        df = df.loc[[c for c in cases if c in df.index]]
        if len(df) != len(cases):
            p('WARNING: arm %s has %d of %d aligned cases' % (arm, len(df), len(cases)))
        pts = df['pred_ts'].astype(float).values
        tlv = np.array([L2I[str(x).strip()] for x in gt.loc[df.index, 'overall_assessment']])
        tts = gt.loc[df.index, 'total_score'].astype(float).values
        plv = np.array([cut_inclusive(v, THR) for v in pts])

        p('')
        p('=' * 74)
        p('ARM %s   pred_ts sd=%.3f mean=%.3f (true sd=%.3f) sd_ratio=%.3f' % (
            arm, pts.std(ddof=1), pts.mean(), tts.std(ddof=1),
            pts.std(ddof=1) / tts.std(ddof=1)))
        p('=' * 74)
        entry = {'pred_sd': float(pts.std(ddof=1)), 'pred_mean': float(pts.mean()),
                 'true_sd': float(tts.std(ddof=1)),
                 'sd_ratio': float(pts.std(ddof=1) / tts.std(ddof=1))}
        entry['original_thresholds'] = full_block(
            '%s | ORIGINAL thresholds %s' % (arm, THR), tlv, plv, tts, pts)

        # ---- recalibrated (metric 6)
        rthr = quantile_thresholds(pts, tlv)
        plv_r = np.array([cut_inclusive(v, rthr) for v in pts])
        p('')
        p('[6] RECALIBRATED THRESHOLDS (prevalence/quantile matching on SAME data)')
        p('    OPTIMISTIC: fitted on the very data it is evaluated on. Needs nested CV')
        p('    to claim honestly. Reported to separate RANKING ability from CALIBRATION.')
        p('    refit thresholds = [%s]' % ', '.join('%.4f' % t for t in rthr))
        entry['recalibrated_thresholds_values'] = rthr
        entry['recalibrated_thresholds'] = full_block(
            '%s | RECALIBRATED thresholds (optimistic, in-sample)' % arm,
            tlv, plv_r, tts, pts)
        results['arms'][arm] = entry

    # ================= CEILING (metric 7) =================
    p('')
    p('=' * 74)
    p('[7] CEILING REFERENCE: top-k fields perfect, remaining fields = training mode')
    p('=' * 74)
    fr = rules['field_rules']
    FIELDS = list(fr.keys())
    RANK = ['microthrombus', 'exudation', 'rbc_aggregation', 'loop_length',
            'capillary_count']
    p('variance ranking used (given): microthrombus 41.2%%, exudation 13.1%%,'
      ' rbc_aggregation 7.9%%, loop_length 7.3%%, capillary_count 6.9%%')

    # per-field score matrix on aligned dev cases
    fs = {}
    for f in FIELDS:
        if f not in sub.columns:
            p('  WARNING: field column %s absent from manifest' % f)
            continue
        fs[f] = np.array([field_score(fr[f], v) if pd.notna(v) else None
                          for v in sub[f].values], dtype=object)
    # mode of the FIELD SCORE over cases with a resolvable score
    mode_score = {}
    for f, arr in fs.items():
        vals = [v for v in arr if v is not None]
        mode_score[f] = float(pd.Series(vals).mode().iloc[0]) if vals else 0.0

    def recon(kfields):
        tot = np.zeros(len(sub))
        for f in fs:
            if f in kfields:
                col = np.array([mode_score[f] if v is None else float(v)
                                for v in fs[f]])
            else:
                col = np.full(len(sub), mode_score[f])
            tot += col
        return tot

    # additivity check: all fields perfect
    all_perfect = recon(set(fs.keys()))
    p('')
    p('additivity check (ALL 21 fields perfect, sum of field_rules scores):')
    p('  MAE vs true total_score = %.4f | exact-match rate = %.4f' % (
        float(np.mean(np.abs(all_perfect - true_ts))),
        float(np.mean(np.abs(all_perfect - true_ts) < 1e-6))))
    p('  -> quantifies the label self-inconsistency floor MEASURED here.')
    results['ceiling'] = {'all_fields_perfect_mae':
                          float(np.mean(np.abs(all_perfect - true_ts)))}

    scen = {'k2 (microthrombus, exudation)': RANK[:2],
            'k5 (top-5 variance fields)': RANK[:5],
            'k21 (all fields perfect)': FIELDS}
    for nm, kf in scen.items():
        rec_ts = recon(set(kf))
        rec_lv = np.array([cut_inclusive(v, THR) for v in rec_ts])
        mae = float(np.mean(np.abs(rec_ts - true_ts)))
        p('')
        p('  --- %s : reconstructed MAE = %.4f, sd = %.3f' % (nm, mae,
                                                             rec_ts.std(ddof=1)))
        results['ceiling'][nm] = {'mae': mae, 'pred_sd': float(rec_ts.std(ddof=1))}
        results['ceiling'][nm]['metrics'] = full_block(
            'CEILING %s | original thresholds' % nm, true_lv, rec_lv, true_ts, rec_ts)

    p('')
    p('NOTE on QWK: no repeat annotation exists in this dataset, so the inter-rater')
    p('ceiling for QWK is UNMEASURABLE. A QWK of X cannot be judged against a human')
    p('reproducibility reference here. Interpret QWK only relative to the baselines.')

    with open(os.path.join(OUTDIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    with open(os.path.join(OUTDIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT) + '\n')
    print('\nWROTE %s/summary.json and report.txt' % OUTDIR)


if __name__ == '__main__':
    main()
