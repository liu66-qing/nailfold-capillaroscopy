# Spatial ROI Feature Reflection

## Fact result

- Development-only, 186 cases, 1,708 frames, fixed case folds, `locked_cases_seen=0`.
- Features: fixed full/upper/lower/center ROI color, exposure, sharpness and texture statistics.
- Initial seed 20260828: clarity +2.30pp (4/5 fold wins), exudation +4.20pp (4/5), but blood color -0.92pp and SVP -2.32pp.
- Five-seed stability: clarity deltas +2.30, +0.77, +1.54, +2.87, +2.43pp; mean **+1.98pp**. Exudation deltas +4.20, -1.10, +3.60, -0.59, +3.24pp; mean **+1.87pp**.
- Formal gates require >=2pp field gain and >=3/5 fold wins, with no unacceptable shared-field regression.

## Decision

- Shared `dual_seg_roi` route: **rejected**.
- Clarity and exudation signals are promising but do not pass the multi-seed margin; they remain research observations only.
- Blood color regression is consistent enough to prohibit shared ROI concatenation.
- No baseline, geometry route, or locked cohort was changed.

## Interpretation

Spatial quality/color statistics may help selected fields, but the effect is small relative to seed/fold noise. This is not evidence that a color/ROI representation solves weak labels. Background fields must be evaluated independently rather than through a shared route.
