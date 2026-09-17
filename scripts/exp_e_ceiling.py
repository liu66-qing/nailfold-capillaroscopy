"""EXP-E: what is the best achievable total_score MAE given the INPUT, not the model?

Decomposes total_score into its 17 field contributions via score_rules_v3.json,
then simulates oracle/mode substitution per observability tier:

  tier OBS   fields with demonstrated image signal   (lift over random >= 0.08)
  tier NONE  fields with no image signal             (lift <= 0.05)
             -> flow_state, rbc_aggregation, crossing_ratio, malformation_ratio,
                hemorrhage; plus every field never modelled at all
                (afferent/efferent/apex diameter, loop_length, ratio,
                 flow_speed, vasomotion, wbc_count, sweat_duct)

Scenarios (all evaluated against the true OCR total_score):
  S0  everything = per-field mode           -> pure prior, no vision at all
  S1  OBS oracle + NONE mode                -> CEILING for a static-image model
  S2  OBS oracle + NONE oracle              -> sanity: should be ~0 (validates
                                               that the rules reproduce the score)
  S3  OBS mode + NONE oracle                -> how much the unobservable half
                                               alone is worth
  S4  current model on OBS + NONE mode      -> where EXP-C actually sits

If S1's MAE is close to the ~2.78 that EXP-C reached, the model is already at
the limit the input permits and further architecture work cannot pay off.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RULES = '/root/nailfold/artifacts/labels/score_rules_v3.json'
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
OUT = Path('/root/nailfold/artifacts/experiments/exp_e_ceiling')

# Measured image signal (balanced-accuracy lift over random) from prior runs.
OBSERVABLE = {'clarity', 'blood_color', 'exudation', 'capillary_count',
              'microthrombus', 'papilla', 'subpapillary_venous_plexus'}
NO_SIGNAL = {'flow_state', 'rbc_aggregation', 'crossing_ratio',
             'malformation_ratio', 'hemorrhage'}
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263"}


def field_score(rule, raw):
    """Score one field value using its rule; None if unscoreable."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == '':
        return None
    if rule['type'] == 'categorical_lookup':
        return rule['mapping'].get(str(raw).strip())
    if rule['type'] == 'numeric_tree':
        try:
            x = float(str(raw).strip())
        except ValueError:
            return None
        node = rule['tree']
        while 'threshold' in node:
            node = node['left'] if x < node['threshold'] else node['right']
        return node.get('value')
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.load(open(RULES))
    rules, groups = R['field_rules'], R['groups']
    all_fields = [f for g in groups.values() for f in g]

    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    L['ts'] = pd.to_numeric(L.total_score, errors='coerce')
    d = L[(L.evaluation_role == 'development') & L.ts.notna()].copy()
    d = d[~d.exam_case_id.isin(CONFLICTED)]
    print(f"cases: {len(d)}   fields in rules: {len(all_fields)}\n")

    # per-case, per-field oracle score
    sc = pd.DataFrame(index=d.index)
    for f in all_fields:
        if f not in d.columns or f not in rules:
            sc[f] = np.nan
            continue
        sc[f] = [field_score(rules[f], v) for v in d[f]]

    cov = sc.notna().mean()
    print("field coverage / dynamic range / tier:")
    spread, tier = {}, {}
    for f in all_fields:
        vals = sc[f].dropna()
        spread[f] = float(vals.max() - vals.min()) if len(vals) else 0.0
        tier[f] = 'OBS' if f in OBSERVABLE else 'NONE'
        print(f"  {f:28s} cov={cov[f]:.2f}  spread={spread[f]:5.2f}  "
              f"mode={vals.mode().iloc[0] if len(vals) else float('nan'):5.2f}  {tier[f]}")

    obs_sp = sum(spread[f] for f in all_fields if tier[f] == 'OBS')
    none_sp = sum(spread[f] for f in all_fields if tier[f] == 'NONE')
    print(f"\ntotal dynamic range: OBS={obs_sp:.2f} ({obs_sp/(obs_sp+none_sp):.1%})  "
          f"NONE={none_sp:.2f} ({none_sp/(obs_sp+none_sp):.1%})")

    # rule-reconstructed total vs OCR total: validates the rules AND bounds label noise
    recon = sc[all_fields].sum(axis=1, min_count=1)
    ok = recon.notna() & d.ts.notna()
    err = (recon[ok] - d.ts[ok]).abs()
    print(f"\nrule-reconstructed total vs OCR total_score (n={ok.sum()}):")
    print(f"  exact (<=0.05): {(err <= 0.05).mean():.1%}   MAE={err.mean():.3f}   "
          f"max={err.max():.2f}   >1.0 off: {(err > 1.0).sum()} cases")

    # Fields with zero coverage contribute nothing and have no mode; treat as 0.
    mode_val = {}
    for f in all_fields:
        v = pd.to_numeric(sc[f], errors='coerce').dropna()
        mode_val[f] = float(v.mode().iloc[0]) if len(v) else 0.0

    def simulate(use_oracle):
        """use_oracle: set of fields taking their true value; rest take the mode."""
        tot = np.zeros(len(d), dtype=float)
        for f in all_fields:
            m = mode_val[f]
            if f in use_oracle:
                col = pd.to_numeric(sc[f], errors='coerce').values.astype(float)
                tot += np.where(np.isnan(col), m, col)
            else:
                tot += m
        return tot

    obs = {f for f in all_fields if tier[f] == 'OBS'}
    none = {f for f in all_fields if tier[f] == 'NONE'}
    true = d.ts.values

    scen = {
        'S0_all_mode': set(),
        'S1_obs_oracle_CEILING': obs,
        'S2_all_oracle': obs | none,
        'S3_none_oracle_only': none,
    }
    print("\n" + "=" * 78)
    print("SCENARIO                     MAE   RMSE   pred_sd  sd_ratio   rho")
    print("=" * 78)
    res = {}
    for name, use in scen.items():
        p = simulate(use)
        mae = float(np.mean(np.abs(p - true)))
        rmse = float(np.sqrt(np.mean((p - true) ** 2)))
        sd = float(p.std())
        rho = float(pd.Series(p).corr(pd.Series(true), method='spearman'))
        res[name] = {'mae': mae, 'rmse': rmse, 'pred_sd': sd,
                     'sd_ratio': sd / float(true.std()), 'spearman': rho}
        print(f"{name:26s} {mae:6.3f} {rmse:6.3f}  {sd:7.3f}  {sd/true.std():8.3f} {rho:6.3f}")

    const = float(np.median(true))
    res['constant_median'] = {'mae': float(np.mean(np.abs(true - const))), 'value': const}
    print(f"{'constant(median)':26s} {res['constant_median']['mae']:6.3f}")
    print(f"\ntrue_sd = {true.std():.3f}")

    ceil = res['S1_obs_oracle_CEILING']['mae']
    print("\n" + "=" * 78)
    print(f"CEILING for a static-image model (perfect on observable fields): MAE = {ceil:.3f}")
    print(f"EXP-C best actual:                                              MAE = 2.775")
    print(f"in-sample constant baseline:                                    MAE = {res['constant_median']['mae']:.3f}")
    gap = 2.775 - ceil
    print(f"\nremaining headroom from modelling alone: {gap:.3f}")
    if gap < 0.4:
        print("=> Model is already near the input-imposed limit. Architecture work,")
        print("   bigger backbones and LLMs cannot close a gap this small; the binding")
        print("   constraint is the unobservable half of the score and label noise.")
    else:
        print("=> Meaningful headroom remains for modelling the observable fields.")
    print("=" * 78)

    (OUT / 'ceiling.json').write_text(json.dumps(
        {'scenarios': res, 'spread': spread, 'tier': tier,
         'obs_spread': obs_sp, 'none_spread': none_sp,
         'recon_exact_rate': float((err <= 0.05).mean()),
         'recon_mae': float(err.mean())}, ensure_ascii=False, indent=2) + '\n')
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
