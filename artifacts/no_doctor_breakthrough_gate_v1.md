# No-Doctor-Annotation Breakthrough Gate

## Scope

This gate records what can and cannot be improved with the current 186 development cases, existing frames/videos, existing weak case labels, and no new doctor annotations. The 47-case historical locked cohort remains excluded from selection.

## Completed controlled attempts

| Group | Controlled attempt | Result | Decision |
|---|---|---:|---|
| B count/instances | 152 threshold/skeleton/aggregation candidates, validation-only selection | OOF BA 0.3050 vs dedicated count GBT 0.5287 | Reject; aggregation alone is not the bottleneck |
| D exudation | mean/median/max/top-k frame evidence with DINO/Hulu/dual | Best BA 0.6793 vs formal binary baseline 0.6836 | Reject; no stable informative frame subset |
| A/E ROI | fixed full/upper/lower/center quality/background features | clarity mean +1.98pp; exudation +1.87pp; color and SVP regress | Reject shared route; ROI-only SVP -11.4pp, papilla -7.3pp |

## Actual strategy

1. **Freeze high-performing groups.** Keep geometry routes and compatibility-default policies as protected components. Do not interpret normalized MAE scores or default hit rates as classification accuracy.
2. **Keep binary delivery where it is defensible.** Retain the formal `dual_seg` binary route for clarity, blood_color, exudation and SVP; keep papilla exploratory and hemorrhage review-only.
3. **For Group B, change evidence rather than classifier.** Any future count/crossing/malformation improvement must use independently auditable instance masks, skeleton/topology and temporal identity. Threshold-only post-processing is closed.
4. **For Group D, use review/abstention rather than frame selection.** Exudation may remain a binary screening result with uncertainty; hemorrhage should be anomaly-triggered and never majority-class claimed.
5. **For Groups A/E, improve measurement validity.** Color/illumination calibration, evaluability flags and fixed ROI can be product safeguards, but they are not an accuracy breakthrough without field-specific gold evidence.
6. **For Group F, do not train another static classifier.** Dynamic fields require timestamped continuous clips, observation duration and event evidence; current flow-state performance fails the gate.

## Stop condition

Under the no-doctor-annotation constraint, another global encoder, LoRA, TTA, attention pooling, threshold sweep, or stacking search is not justified. A genuine accuracy breakthrough requires new task-specific evidence: at minimum a small independently reviewed set of instances/lesions/background regions or reliable dynamic event labels. Without that evidence, improvements can only be claimed as development-protocol changes, not as medical recognition gains.
