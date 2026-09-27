# loop_length > 250 on locked-47: preregistration (written before scoring)

Authorised by the user on 2026-09-28 ("授权给你").

## What is read
- Features only: `artifacts/experiments/locked_consumed_20260924/features_locked/`.
  These were extracted on 2026-09-24 with the same encoder (weights sha256 55cbb5d8…),
  47 cases and 402 CAPorg images, with no placeholder images. No image is re-read and no image leaves the machine.
- Labels: `loop_length` of the locked cases in `server_code_audit/locked_evaluation_v1_reviewed.csv`.
  Of the 47 locked cases, 45 have a numeric value; cases without a value are dropped before scoring.

## What is frozen (never refitted or changed after reading)
- `artifacts/models/rag_heads_v1/bundle.joblib`, sha256 8c66fbb940217697ae3cb3b1b4473c384edf178428450e33add8d1665e3dae72.
  It was fitted on the 186 dev cases only.
- Field: `loop_length`. The target is > 250. The fitted heads are the five poolings (StandardScaler → PCA64 → LogReg C=0.03).
  - Probability: for each pooling, the mean over the case's images; then the mean over poolings.
  - Decision threshold: 0.5.
  - Abstain margin: 0.1965, from bundle `abstain_margin`.
- Features are used as they are; there is no recentring. This is the same device.
  The batch distance is reported for information only and does not decide anything.

## Metrics
All metrics are computed from one per-case file: full-coverage and selective accuracy, BA, AUROC, per-class recall, confusion matrix, coverage, the correct/wrong/abstained split, and abstention by class.
- Case-bootstrap 95% CI for BA; 2000 resamples.
- 1000 label permutations with the frozen predictions; p = (1 + #≥obs) / 1001.

## Gates (identical to the dev audit)
1. Full coverage: BA ≥ 0.65, and both recalls ≥ 0.50.
2. Permutation p < 0.05. The dev audit used 0.01; n = 45 cannot reach that without an extreme effect, so 0.05 is stated here before reading.
3. Selective: coverage ≥ 0.50, covered BA ≥ 0.75, and both covered recalls ≥ 0.65.
4. The abstention-rate gap between the two classes is ≤ 0.15.

## Interpretation fixed in advance
- Pass: "loop_length > 250 was reproduced on a consumed internal holdout of the same hospital and device". It is still not an external or cross-device validation.
- Fail: loop_length does not enter RAG.
- In either case the number is recorded, and there is no tuning afterwards.
- Budget: locked-47 has already been consumed for other fields. This is the first read for loop_length > 250.
