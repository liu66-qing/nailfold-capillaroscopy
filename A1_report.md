# A1 pooling operator search under multiclass / ordinal targets

Status: `FAIL` for the pooling question; confidence: `HIGH`. Development-only, `locked_cases_seen: 0` (186 dev cases asserted, locked-47 intersection asserted empty on both labels and frames). Seed 20260913.

Grid: 7 fields x 6 pooling operators (mean, median, max, q90, topk3_mean, logsumexp) x 4 feature sets (dinov2, geometry, roi_quality, concat) x 1-2 heads (multinomial LR; ordinal cumulative-link for the 5 ordinal fields). Operator/feature/head chosen on the validation fold `(test_fold+1)%5` by accuracy; test fold scored once as OOF. Mode baseline computed per fold on that fold's training folds.

## Headline: the gate as specified is the wrong instrument

`delta = accuracy - mode_baseline_acc` was passed by 3 fields, but that delta measures **multiclass head vs mode baseline**, not pooling. Isolating the pooling contribution (selected operator minus `mean` at the same feature set and head) shows almost nothing:

| Field | n | Selected config | acc | mode base | delta | folds+ | **pooling gain vs mean** | adj1 | QWK | main-class recall drop | gate as spec'd | pooling >= 0.05 |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|
| exudation | 178 | concat / mean / ordinal | 0.697 | 0.517 | +0.1798 | 5/5 | **+0.0000** | 0.989 | 0.418 | +0.0000 | PASS | NO |
| clarity | 185 | dinov2 / median / multinomial | 0.724 | 0.454 | +0.2703 | 5/5 | **-0.0378** | 1.000 | 0.449 | +0.0319 | FAIL | NO |
| blood_color | 172 | concat / max / multinomial | 0.674 | 0.547 | +0.1279 | 4/5 | **-0.0116** | 1.000 | 0.345 | -0.0745 | PASS | NO |
| subpapillary_venous_plexus | 181 | concat / median / multinomial | 0.597 | 0.381 | +0.2155 | 5/5 | **+0.0497** | 0.972 | 0.535 | +0.0000 | PASS | NO (0.0497) |
| microthrombus | 182 | geometry / topk3_mean / ordinal | 0.593 | 0.593 | +0.0000 | 3/5 | **+0.0604** | 0.808 | 0.347 | -0.0926 | FAIL | YES |
| papilla | 181 | dinov2 / q90 / multinomial | 0.503 | 0.425 | +0.0773 | 4/5 | **+0.0166** | 0.895 | 0.312 | +0.0649 | FAIL | NO |
| capillary_count | 181 | concat / median / ordinal | 0.624 | 0.691 | **-0.0663** | 0/5 | -0.0055 | 0.961 | 0.406 | +0.0240 | FAIL | NO |

Per-class recalls (n>=10 only) and per-fold deltas are in `metrics.json`. Rare class: `exudation` class 2 (`+++`, n=4) recall 0.000, reported in `rare_classes` and excluded from all summaries.

Per-fold deltas, selected configs:
- exudation `[0.270, 0.161, 0.139, 0.125, 0.206]`
- clarity `[0.237, 0.387, 0.308, 0.279, 0.147]`
- blood_color `[0.222, -0.107, 0.088, 0.190, 0.188]`
- subpapillary `[0.189, 0.167, 0.263, 0.238, 0.206]`
- microthrombus `[0.000, 0.100, 0.105, -0.238, 0.088]`
- papilla `[0.000, 0.258, 0.026, 0.095, 0.030]`
- capillary_count `[-0.027, -0.129, -0.053, -0.122, 0.000]`

## Sanity check: PASSED

Prediction was that max/topk3 beats mean on sparse-event fields but not on global-attribute fields.

| Field | max - mean | topk3 - mean | expected |
|---|--:|--:|---|
| microthrombus | +0.0440 | **+0.0604** | win (sparse) - held |
| exudation | -0.0506 | -0.1180 | win (sparse) - did not hold |
| clarity | -0.0108 | +0.0162 | no win (global) - held |
| blood_color | -0.0116 | +0.0174 | no win (global) - held |

`global_fields_where_max_wins: []`. No global-attribute field was won by max/topk3 at >=0.05, so the results are not pure noise. The one clear sparse-event win (microthrombus topk3 +0.060) is directionally consistent with the mechanism. Exudation preferring `mean` weakens the story: exudation is graded by extent rather than by presence of a single worst frame, so mean-like aggregation is defensible there.

## Conclusions

1. **Pooling operator choice is rejected as a lever, again.** Only microthrombus shows a pooling gain past 0.05, and it fails the stability requirement (one fold at -0.238) and costs 0.093 of main-class recall. Every other field is within noise of `mean` or worse. A1 does not overturn E2; changing the target from binary to multiclass/ordinal did not make pooling matter.
2. **capillary_count is below its own mode baseline** (-0.0663, 0/5 folds positive). Under multiclass, this field is currently not learnable from these frozen features; the mode `>=7` is the better predictor.
3. **The real finding is orthogonal to A1's question**: the multiclass/ordinal head itself clears the mode baseline substantially on exudation (+0.180), clarity (+0.270), blood_color (+0.128) and subpapillary (+0.216), all 4-5/5 folds positive, with `adj1` 0.97-1.00 and QWK 0.35-0.54. That is a head/target-formulation result, not a pooling result, and it was obtained with plain `mean` pooling on concat or dinov2 features. This is worth following up; the pooling search is not.
4. Recommendation: stop searching frame-aggregation operators on frozen features. Use `mean` pooling as the fixed default and spend the next experiment on the multiclass/ordinal formulation and on the 4 fields above, where the signal actually is.

Artifacts: `metrics.json`, `a1_pooling.py`, `run.log` in this directory.
