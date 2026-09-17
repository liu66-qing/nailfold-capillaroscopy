# v6 Experiment Audit

All experiments used fixed case-level development folds. Locked cases were excluded from training and tuning; the only locked evaluation was run after freezing rank4 checkpoints.

## Accepted evidence

- Rank4 DINOv2 LoRA, 30 epochs, learning rate 1e-4: exudation development BA 0.7719 with 4/5 fold improvements versus rank8/lr1e-4; SVP development BA 0.7612 but insufficient fold gate.
- Frozen rank4 locked evaluation: clarity 0.7125, blood_color 0.7464, exudation 0.6182, SVP 0.8730, papilla 0.5699.

## Failed or held routes

- Attention MIL: below the established routes.
- Hemorrhage DINO/ExtraTrees: BA 0.5000; no reliable improvement.
- Capillary count detector features: below v1.
- Crossing ratio detector proxy: BA 0.1914, Spearman -0.016.
- Malformation ratio detector proxy: exploratory binary result only; insufficient class support for a formal gate.
- Detector, segmentation, geometry, pooling, threshold, and adapter variants did not provide a stable global improvement.

## Release decision

No new route demonstrates stable superiority under the development gate and locked audit. Keep v1 as the formal model. Preserve all v6 scripts, logs, OOF files, and failure evidence for future research.
