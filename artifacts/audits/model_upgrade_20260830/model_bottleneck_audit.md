# Model upgrade bottleneck audit (2026-08-30)

## Current best and evaluation contract

The protected v1 structured model remains the reference: DINOv2 + HuluMed features, segmentation statistics, and case-level ExtraTrees/GBT heads under the fixed 186-development / 47-locked split and five case-level OOF folds. It must not be overwritten.

Primary metrics are balanced accuracy for categorical fields, MAE and engineering tolerance-in-range for continuous fields, and only a supplementary normalized error score for cross-field summaries. The v1 development OOF mean BA for nine categorical fields is 40.00%; locked retest mean BA is 45.82%. The 67.08% equalized 16-field score is not accuracy. Geometry normalized scores near 93% do not imply clinical accuracy; loop-length tolerance-in-range is 15.56%.

## Bottleneck diagnosis

1. Labels are case-level weak labels while inputs are multi-frame images; frame-to-label assignment is not identifiable.
2. Rare classes are under-supported (hemorrhage positive development support is about 12 cases), so ordinary accuracy is misleading and minority recall collapses.
3. Capillary count, crossing ratio, malformation ratio, and loop length depend on instance/topology truth. Current masks produce overcount, undercount, and frame-instability errors; no instance-level clinical gold standard exists.
4. Existing attention-MIL, geometry concatenation, LoRA, and TTA pilots did not show stable gains. This argues against a backbone-only intervention.
5. Several high apparent scores are defaults or metric artifacts (compatibility defaults and normalized error score).

## Governed auxiliary data

Use only the conditional development assets in `artifacts/derived/vascular_dataset_governance_20260830/train_ready`: 89 detection images and 2,332 automatically confirmed segmentation pairs. Keep 829 unresolved pairs, 454 unknown/unpermissioned images, all locked-overlap assets, and unresolved semantic cases out of training and model selection. Augmentations remain bound to their source group and are not independent samples.

## Experiment gate

Run CPU smoke checks first. Any development experiment must use the frozen case folds, never locked cases, and preserve v1 artifacts. Compare v1 against (a) frozen v1 features plus vascular auxiliary statistics, (b) segmentation-quality gating for weak fields, and (c) topology/count-specific features. Accept a route only if both macro-F1 and balanced accuracy improve, with at least 3/5 positive folds and no material regression in other delivered fields. Report field-level BA, macro-F1, minority recall, MAE, and tolerance-in-range.

## Deliverability conclusion

The current evidence supports starting a development-only auxiliary experiment. It does **not** support a claim that all fields can reach 90% accuracy now. The highest-value path is better frame/instance/topology supervision and clearer weak-field definitions; adding ordinary images alone is unlikely to fix the observed errors.
