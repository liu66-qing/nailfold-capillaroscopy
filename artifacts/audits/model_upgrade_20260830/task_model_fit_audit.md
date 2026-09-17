# Task-model fit audit (2026-08-30)

## Executive conclusion

The existing system is modular in name, but several modules are still solving the wrong statistical problem. DINOv2/HuluMed embeddings plus tree heads are reasonable small-sample baselines; they are not task-specific measurement models. The low scores are therefore not evidence that vascular data are useless. They show that the inductive bias is mismatched to several endpoints.

## Geometry fields

**Current fit:** weak. A global image embedding and tabular regressor cannot reliably estimate microns or loop length without an explicit scale and centerline. The high normalized diameter scores are partly range normalization, while loop-length tolerance is 15.56%.

**Correct model:** a measurement pipeline: vessel segmentation -> skeleton graph -> endpoint/branch identification -> calibrated pixel-to-length regression, with heteroscedastic uncertainty. Train the segmentation/centerline component separately; use robust Huber/quantile regression for residual correction. Do not predict absolute microns until calibration metadata are established.

## Capillary count

**Current fit:** weak. Counting connected components is not instance counting; fragments and merged vessels cause the recorded overcount/undercount pattern.

**Correct model:** instance-aware detector/segmenter plus graph-based deduplication and a count-specific ordinal head. Supervise instance IDs or at least centers/endpoints on a stratified subset. Report count error and agreement, not only class BA.

## Crossing and malformation ratio

**Current fit:** poor. A case-level tree sees aggregate shape statistics but no explicit crossing event or malformation morphology. `cross_vessel` is an engineering class, not yet a clinical definition.

**Correct model:** detect vessel instances, build a proximity/intersection graph, classify graph nodes/edges (crossing, loop, abnormal), then aggregate ratios with uncertainty. Treat ambiguous crossings as abstentions. A plain classifier on pooled embeddings cannot infer a ratio whose denominator is itself uncertain.

## Clarity and blood color

**Current fit:** partial. These are image-quality/appearance tasks, yet generic semantic embeddings and case-level labels are used. The target is closer to calibrated ordinal quality assessment than generic multiclass recognition.

**Correct model:** frame-level quality network with exposure/blur/color statistics and an ordinal (cumulative-link) head; learn quality-weighted pooling to the case. Use illumination normalization only inside training folds and retain an unreadable class.

## Exudation and hemorrhage

**Current fit:** poor. These are sparse, focal findings; mean pooling dilutes positives and class-weighting cannot create support. Hemorrhage has approximately 12 development positives.

**Correct model:** frame-level multiple-instance learning with top-k positive pooling, focal/asymmetric loss, and an explicit uncertain/absent state. Require more true positive cases before clinical use. Evaluate PR-AUC, sensitivity at a prespecified specificity, BA and minority recall; ordinary accuracy is unsafe.

## SVP and papilla morphology

**Current fit:** poor-to-partial. Both require region localization and ordinal morphology, but current features do not enforce a papilla/SVP ROI.

**Correct model:** ROI detector or landmark crop followed by ordinal classifier, with rater-agreement analysis. Without a stable ROI and definition, adding backbone capacity is unlikely to help.

## Why the named “specialized” models underperform

1. Most specialization is downstream feature selection, not a new observation model; the input representation remains generic and pooled.
2. Tree models are strong for low-dimensional tabular data but brittle when fed noisy, correlated embedding statistics and weak labels.
3. Case-level aggregation is performed before the task head, preventing focal-event fields from using the informative frame.
4. Topology-dependent labels are predicted without an explicit graph/instance intermediate target.
5. Chain heads can propagate upstream errors and appear specialized while reducing robustness.

## Priority redesign

P0: frame-level quality-gated MIL for focal fields; independent heads; fixed development OOF.

P1: segmentation/centerline graph for count, crossing, malformation and length; use the 2,332 confirmed pairs only as auxiliary supervision.

P2: ROI-conditioned ordinal heads for clarity, blood color, SVP and papilla.

P3: calibrated measurement and uncertainty, with abstention/HOLD when visibility or pairing quality is insufficient.

Acceptance remains field-level BA + macro-F1/minority recall for categories, MAE + tolerance-in-range for continuous fields, and >=3/5 positive folds without cross-field regression. No aggregate score may mask a failed field.
