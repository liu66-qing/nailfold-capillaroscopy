# Exudation Frame-Consistency Reflection

## Fact result

- Cohort: 182 labelled development cases, 1,708 development frames.
- Protocol: outer 5-fold OOF; model and pooling selected on the adjacent validation fold only.
- Candidate pooling: mean, median, max positive frame, top 25%, top 50%, and top 75% positive-probability frames.
- Best result: dual DINOv2 + HuluMed frame route, OOF BA **0.6793**.
- Formal case-level `dual_seg` binary baseline: **0.6836**.
- Gate: baseline + 0.02 = 0.7036. Result: **fail**.

## Why this is a failure

1. The selected pooling method is unstable across folds: median, top 50%, and top 75% all win different folds.
2. Top-k positive evidence does not outperform case-level mean features, so the data do not support a stable sparse-lesion frame-selection rule.
3. Training still relies on a case label attached to all case frames. Equal case weighting limits domination by long bags but cannot create lesion-local truth.
4. The result is below the simpler formal baseline that also includes segmentation statistics.

## Rollback decision

- Status: **rejected**.
- Retain the formal binary `dual_seg` exudation route at BA 0.6836.
- Do not deploy top-k frame selection or call it lesion localization.
- No baseline model or locked cohort was modified or evaluated.

## Next gate

Without lesion boxes/masks, another attention/top-k/self-training sweep is not justified. Future exudation work must introduce an independently auditable lesion candidate (for example temporal persistence of a spatial ROI) and must beat the same formal baseline by at least 0.02 BA.
