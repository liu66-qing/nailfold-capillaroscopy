# Deep root-cause iteration 2

## Finding 1: the unit of supervision is misaligned

The feature builders aggregate all frames by `exam_case_id` before fitting heads. The labels are also case-level, but the source records do not establish that every frame expresses the case label. This creates an irreducible label-mixture problem: averaging features across frames can erase the small region that determines exudation, hemorrhage, crossing, or papilla state. More images from the same case do not add independent supervision.

**Fix:** retain frame-level features and train a gated case aggregator on development folds only. The aggregator must learn mean/top-k/quality-weighted pooling inside each training fold, with an explicit `unreadable` route; never select pooling on locked data.

## Finding 2: topology features are proxies, not targets

The current count pipeline uses connected components/skeleton-derived proxies. Its own attribution audit records overcount, undercount, and frame instability. Therefore count, crossing ratio, malformation ratio, and loop length are being learned from a noisy proxy whose failure modes are correlated with image quality. A larger classifier cannot correct this systematically.

**Fix:** add instance-level centerline/endpoint/crossing supervision for a small, stratified development subset; use the governed masks only as auxiliary representation learning. Until then, route topology-dependent outputs through an uncertainty/HOLD policy instead of reporting false precision.

## Finding 3: chained heads can amplify upstream errors

`evaluate_classifier_chain.py` appends each previous field's predicted one-hot output to the next field's features. This is a causal modeling assumption that is not supported by the labels and can propagate an early error through all later fields. The independent-head v1 route should remain the baseline; the chain is exploratory unless it wins the fold gate.

**Fix:** compare independent heads, teacher-forced chain, and cross-fitted predicted chain. Only the cross-fitted version is admissible for OOF claims; reject any route whose gains disappear when previous labels are not available at inference.

## Finding 4: apparent high scores are not the target metric

Default compatibility outputs and normalized continuous scores inflate aggregate summaries. The only defensible acceptance contract is field-level BA/macro-F1/minority recall for categorical tasks and MAE/tolerance-in-range for continuous tasks. A 90% aggregate score can coexist with failed clinical fields.

## Minimal high-value experiment sequence

1. CPU-only leakage audit: verify no image path or hash crosses folds and no locked id appears in features.
2. Frame pooling ablation: mean, median, top-quality 25%, learned quality gate; fixed folds, independent heads.
3. Add 2,332 confirmed segmentation-derived statistics as auxiliary features, with a missing/uncertain indicator.
4. Compare independent vs cross-fitted chain; do not use teacher-forced predictions in deployment.
5. Evaluate topology fields separately and keep them HOLD unless both accuracy and stability gates pass.

The deepest current issue is therefore **supervision granularity and proxy-target mismatch**, with chained error propagation as a secondary implementation risk. Additional ordinary images alone will not solve either problem.
