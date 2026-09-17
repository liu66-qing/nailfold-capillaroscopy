# MODEL v4 overnight experiment report

All experiments used development cases only (186 cases, fixed five folds); locked test cases were never used (`locked_cases_seen=0`). v1 remains frozen.

## Best field routes

| field | BA | route |
|---|---:|---|
| clarity | 0.715986 | v3 YOLO321 detector + v1 features |
| blood_color | 0.710474 | v3 YOLO321 detector + v1 features |
| exudation | 0.710855 | v3 YOLO321 detector + v1 features |
| SVP | 0.698672 | frozen v1 |
| papilla | 0.576955 | v3 YOLO321 detector + v1 features |

## Experiments

- Full 671-image YOLO retraining: downstream BA clarity 0.683439, blood_color 0.698128, exudation 0.687326, SVP 0.687546, papilla 0.597771. More detector labels did not improve the feature distribution.
- U-Net pretraining on 2,332 image/mask pairs: clarity 0.716631, blood_color 0.673437, exudation 0.683626, SVP 0.689390, papilla 0.576955. No field gate pass.
- Detector + U-Net fusion: clarity 0.716631, blood_color 0.675826, exudation 0.699090, SVP 0.693570, papilla 0.581020. No gate pass.
- Detector pooling mean/median/top-k: no field passed the BA, fold-win, and minority-recall gate.
- XGBoost detector fusion: clarity 0.661095, blood_color 0.699721, exudation 0.676289, SVP 0.661114, papilla 0.551093.
- Frame-level MIL: all major fields degraded (clarity 0.415472, blood_color 0.423148, exudation 0.400384, SVP 0.400935, papilla 0.487679).

## Decision

No new v4 route is stable enough to publish. Keep v1 as the formal model and use the v3 detector route only for the four fields that passed its gate (clarity, blood_color, exudation, papilla); keep SVP on v1. The 671 detector labels and 2,332 segmentation pairs are preserved as research/pretraining assets, but current evidence does not support a v4 release or locked-set evaluation.
