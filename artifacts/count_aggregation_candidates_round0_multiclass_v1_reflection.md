# Count Aggregation Candidate Reflection

## Fact result

- Protocol: development-only, 5 outer folds; adjacent fold used for candidate selection; outer test fold evaluated once.
- Source: `round0_multiclass_domain_adapted.json`, filtered to development cases; no locked rows were used in scoring.
- Candidate space: confidence threshold 0.05--0.50, skeleton expansion on/off, median/q75/max/mean aggregation.
- OOF balanced accuracy: **0.3050**.
- Majority balanced-accuracy baseline: 0.2500.
- Existing dedicated count GBT baseline: **0.5287** under its own fixed development OOF protocol.

## Why this is a failure

1. The candidate is only a deterministic post-processing layer over noisy instance predictions; it cannot recover the richer frame-level count features used by the dedicated GBT.
2. Validation repeatedly selected the minimum confidence threshold, indicating that threshold selection is not identifying a stable precision/recall operating point.
3. The OOF confusion matrix remains dominated by `>=7` predictions and has weak recovery of `3--4` and `5--6`, so changing aggregation does not solve the class-boundary problem.
4. The result does not support the earlier hypothesis that q75/max aggregation alone fixes undercount. It also does not quantify instance-level recall because no instance gold truth exists.

## Rollback decision

- Status: **rejected**.
- Production/development route: retain the existing dedicated count GBT and current frozen geometry routes.
- Do not apply the selected threshold or aggregation to any locked or deployment data.
- No segmentation model or v1 baseline was modified.

## Next gate

Do not run another threshold-only sweep. A new B-group experiment is allowed only if it adds an independently testable mechanism, such as temporal instance stability or fragment filtering, and reports count plus crossing/malformation and all four protected geometry MAEs together.
