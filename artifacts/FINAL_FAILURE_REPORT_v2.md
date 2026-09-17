# Model upgrade v2 final failure report

Date: 2026-08-31  
Branch: `model-upgrade-remote-20260831`  
Decision: **retain v1; no v2 release**

## Scope and safety

All completed comparisons used the 186-case development cohort and fixed case-level five-fold OOF. The 47 locked cases were not used for training, feature learning, threshold/pooling selection, error-driven tuning, or intermediate evaluation. No final locked evaluation was run because no candidate passed the development gates.

## Frozen baseline

The formal deployment baseline was reproduced exactly: DINOv2 frame mean + HuluMed frame mean + 105 segmentation features + frame count, ExtraTrees (300 trees, leaf=2, balanced), seed `20260828 + test_fold`. The audit matched 903 reference field-case predictions with zero disagreements. Formal global OOF BA values were clarity 0.694933, blood color 0.661888, exudation 0.683626, SVP 0.698672 and papilla 0.564270.

## Experiment disposition

| Route | Result | Reason not released |
|---|---|---|
| E1 quality gate | FAIL | only exudation improved 5/5; other fields failed fold/minority-recall gates |
| E2 pooling | FAIL | no field met BA + macro-F1 + 3/5-fold + minority-recall gate |
| E3 threshold calibration | FAIL | no field met all gates; SVP BA -6.45pp |
| E4 geometry features | FAIL | clarity/SVP/papilla regressed; blood-color recall -10pp |
| E5 sparse-lesion MIL | FAIL | best exudation BA 0.679260 < v1 0.683626 |
| E6 instance/topology | HOLD | no classification OOF for count/crossing/malformation; only exploratory pixel regressions |
| E7 ROI + ordinal | HOLD | no fold-isolated anatomical ROI proposal artifact |
| E8 calibrated measurement | HOLD | no external micrometre calibration |
| E9 LoRA/adapter | FAIL | exudation BA 0.621164 and papilla BA 0.549131 below v1 |

## Adversarial conclusion

The pro hypothesis was that quality gating, robust pooling, topology features, MIL, ROI heads, calibration, or adaptation would address the diagnosed bottlenecks. The opposing evidence is fold instability, minority-recall loss, non-target regression, absent common endpoints, missing calibration, and lower shared-endpoint OOF. Exact baseline reproduction and locked isolation were independently checked. The opposing evidence defeats every release claim with HIGH confidence.

## Release statement

No new checkpoint is clinically or experimentally justified as a v2 replacement. Preserve v1 models, features, labels and reports. The nearest useful observations are the exudation-only E1 gain and exploratory pixel-domain loop-length improvement, but neither satisfies the full release gate. Further work requires new independently adjudicated labels, calibrated acquisition metadata, or a properly frozen ROI/instance classification protocol.
