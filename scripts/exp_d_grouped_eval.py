"""EXP-D: leakage sensitivity + honest metric suite.

Two independent jobs, neither of which trains anything:

  (1) Metric suite, saved as a reusable module-level function so every future
      experiment reports the same numbers. Report level is authoritative; the
      inclusive '<=' cut is applied ONLY to predicted scores.

  (2) Date#laterality grouped sensitivity analysis. Cases sharing an exam date
      and hand cannot be shown to be different patients, so we re-score every
      existing OOF prediction file under a grouped view: collapse each group to
      one representative and recompute. If metrics drop materially, all prior
      case-level numbers were optimistic.

Usage:
  python exp_d_grouped_eval.py --preds a.csv b.csv ...
  (auto-discovers exp*/oof_predictions.csv and exp_c_*/[ABC]_*_oof.csv if omitted)
"""
import argparse, glob, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

SEV = ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"]
SEV_MAP = {s: i for i, s in enumerate(SEV)}
CUTS = [1.0, 2.0, 4.0, 8.0]
GROUPS_CSV = '/root/nailfold/artifacts/experiments/exp_a_label_audit/case_date_groups.csv'
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263",
              "recovered_archive2/182", "recovered_archive3/244"}


def cut_inclusive(s):
    for t, lab in zip(CUTS, SEV[:-1]):
        if s <= t:
            return lab
    return SEV[-1]


def c_index(true, pred):
    """Fraction of comparable pairs ranked correctly; 0.5 = random."""
    conc = disc = 0
    for i, j in itertools.combinations(range(len(true)), 2):
        if true[i] == true[j]:
            continue
        s = np.sign(true[i] - true[j]) * np.sign(pred[i] - pred[j])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    tot = conc + disc
    return conc / tot if tot else float('nan')


def metric_suite(true_ts, pred_ts, true_lvl, pred_lvl):
    """The canonical metric set. Includes an in-sample constant baseline so no
    number is ever read without its trivial reference point."""
    true_ts = np.asarray(true_ts, float)
    pred_ts = np.asarray(pred_ts, float)
    ti = np.array([SEV_MAP[x] for x in true_lvl])
    pi = np.array([SEV_MAP[x] for x in pred_lvl])

    const = float(np.median(true_ts))
    const_lvl = np.array([SEV_MAP[cut_inclusive(const)]] * len(ti))

    def per_level_mae(p):
        return {SEV[k]: (float(np.mean(np.abs(p[ti == k] - true_ts[ti == k])))
                         if (ti == k).any() else None) for k in range(5)}

    plm = per_level_mae(pred_ts)
    macro = float(np.mean([v for v in plm.values() if v is not None]))

    out = {
        'n': int(len(ti)),
        'mae': float(np.mean(np.abs(pred_ts - true_ts))),
        'macro_mae': macro,
        'spearman': float(spearmanr(true_ts, pred_ts).statistic),
        'c_index': c_index(true_ts, pred_ts),
        'qwk': float(cohen_kappa_score(ti, pi, weights='quadratic', labels=range(5))),
        'lwk': float(cohen_kappa_score(ti, pi, weights='linear', labels=range(5))),
        'balanced_acc': float(balanced_accuracy_score(ti, pi)),
        'acc': float(np.mean(ti == pi)),
        'within_1': float(np.mean(np.abs(ti - pi) <= 1)),
        'severe_underest': float(np.mean((ti - pi) >= 2)),
        'pred_std': float(pred_ts.std()),
        'true_std': float(true_ts.std()),
        'std_ratio': float(pred_ts.std() / true_ts.std()),
        'per_level_mae': plm,
        'per_level_recall': {SEV[k]: (float(np.mean(pi[ti == k] == k))
                                     if (ti == k).any() else None) for k in range(5)},
        'baseline_constant': {
            'value': const,
            'mae': float(np.mean(np.abs(true_ts - const))),
            'macro_mae': float(np.mean([v for v in per_level_mae(
                np.full_like(true_ts, const)).values() if v is not None])),
            'qwk': float(cohen_kappa_score(ti, const_lvl, weights='quadratic', labels=range(5))),
            'acc': float(np.mean(ti == const_lvl)),
            'within_1': float(np.mean(np.abs(ti - const_lvl) <= 1)),
            'severe_underest': float(np.mean((ti - const_lvl) >= 2)),
        },
    }
    out['beats_constant_mae'] = out['mae'] < out['baseline_constant']['mae']
    out['beats_constant_qwk'] = out['qwk'] > out['baseline_constant']['qwk']
    return out


def load_preds(path, labels):
    d = pd.read_csv(path, dtype={'case': str})
    if 'true_ts' not in d.columns:
        return None
    d = d[d.true_ts.notna()].copy()
    # attach authoritative report level; never trust a level column in the file
    d = d.merge(labels[['exam_case_id', 'report_level']],
                left_on='case', right_on='exam_case_id', how='left')
    d = d[d.report_level.notna()]
    d['pred_level'] = d.pred_ts.map(cut_inclusive)
    return d[['case', 'true_ts', 'pred_ts', 'report_level', 'pred_level']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', nargs='*', default=None)
    args = ap.parse_args()

    out = Path('/root/nailfold/artifacts/experiments/exp_d_grouped_eval')
    out.mkdir(parents=True, exist_ok=True)

    L = pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',
                    dtype={'exam_case_id': str})
    L['report_level'] = L.overall_assessment.where(L.overall_assessment.isin(SEV))
    labels = L[['exam_case_id', 'report_level']]

    groups = pd.read_csv(GROUPS_CSV, dtype={'case': str})
    gmap = dict(zip(groups.case, groups.group_key))

    paths = args.preds
    if not paths:
        paths = sorted(glob.glob('/root/nailfold/artifacts/experiments/*/oof_predictions.csv')
                       + glob.glob('/root/nailfold/artifacts/experiments/exp_c_*/*_oof.csv'))
    print(f"scoring {len(paths)} prediction files\n")

    results = {}
    for p in paths:
        name = Path(p).parent.name + '/' + Path(p).stem
        d = load_preds(p, labels)
        if d is None or len(d) < 20:
            print(f"skip {name} (no usable rows)")
            continue
        d = d[~d.case.isin(CONFLICTED)]

        full = metric_suite(d.true_ts, d.pred_ts, d.report_level, d.pred_level)

        # grouped view: one representative per date#laterality group (median pred)
        d = d.copy()
        d['grp'] = d.case.map(gmap).fillna(d.case)
        agg = d.groupby('grp').agg(true_ts=('true_ts', 'median'),
                                   pred_ts=('pred_ts', 'median'),
                                   report_level=('report_level', 'first')).reset_index()
        agg['pred_level'] = agg.pred_ts.map(cut_inclusive)
        grouped = metric_suite(agg.true_ts, agg.pred_ts, agg.report_level, agg.pred_level)

        results[name] = {'case_level': full, 'date_grouped': grouped}
        print(f"{name}")
        for tag, m in [('case', full), ('grouped', grouped)]:
            print(f"  {tag:8s} n={m['n']:3d} MAE={m['mae']:.3f}(const {m['baseline_constant']['mae']:.3f}) "
                  f"macroMAE={m['macro_mae']:.3f} rho={m['spearman']:.3f} C={m['c_index']:.3f} "
                  f"QWK={m['qwk']:.3f} sd_ratio={m['std_ratio']:.3f} "
                  f"beats_const={m['beats_constant_mae']}/{m['beats_constant_qwk']}")
        print()

    (out / 'all_metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

    if results:
        print("=" * 100)
        print(f"{'experiment':42s} {'MAE':>6s} {'const':>6s} {'macro':>6s} {'rho':>6s} "
              f"{'C-idx':>6s} {'QWK':>6s} {'sdR':>5s}")
        for n, r in sorted(results.items(), key=lambda kv: -kv[1]['case_level']['qwk']):
            m = r['case_level']
            print(f"{n[:42]:42s} {m['mae']:6.3f} {m['baseline_constant']['mae']:6.3f} "
                  f"{m['macro_mae']:6.3f} {m['spearman']:6.3f} {m['c_index']:6.3f} "
                  f"{m['qwk']:6.3f} {m['std_ratio']:5.2f}")
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
