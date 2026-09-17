# Current model complete issue register

Date: 2026-08-30

## Confirmed issues

1. **Supervision-unit mismatch.** Inputs are multi-frame, most labels are case-level. Current aggregation can dilute focal findings and cannot establish which frame supports a label.
2. **Topology proxy mismatch.** Count, crossing, malformation and loop length use connected-component/skeleton proxies rather than instance and graph truth. Audits already show overcount, undercount and frame instability.
3. **No defensible absolute calibration.** Device/scale metadata are not verified. Pixel-domain geometry is usable; micron claims are not.
4. **Rare-class collapse.** Hemorrhage has roughly 12 development positives. Class weighting cannot manufacture support; ordinary accuracy hides majority prediction.
5. **Generic representation.** DINOv2/HuluMed embeddings are useful baselines but do not enforce vessel, ROI, endpoint or crossing structure.
6. **Pooling is not task-specific.** A single case-level pool is reused for focal events, quality grades, topology and measurement tasks.
7. **ROI absence.** SVP and papilla are localized anatomical endpoints but the main route has no stable ROI/landmark constraint.
8. **Chain error propagation risk.** The exploratory classifier chain appends previous predictions; upstream errors can contaminate later fields. Teacher-forced results are not deployable evidence.
9. **Metric confusion.** Normalized error scores and default compatibility outputs can inflate cross-field summaries; they are not clinical accuracy.
10. **Locked/data-governance boundary.** Unknown assets, unresolved segmentation pairs, duplicate conflicts and all locked-overlap assets must remain excluded from training and selection.

## High-risk implementation issues requiring verification

1. Fold-local fitting of every scaler, imputer, pooling choice, threshold, class weight and calibrator.
2. Exact case-level disjointness of all frame paths, hashes and augmentation groups across folds.
3. Whether any precomputed feature artifact was generated with locked images or with labels unavailable at training time.
4. Missing-value semantics: distinguish unreadable/absent from zero and from failed extraction.
5. Class vocabulary drift between folds; rare classes may be dropped or remapped silently.
6. Probability calibration and abstention thresholds are not yet part of the formal release contract.
7. Multiple-comparison risk across many exploratory routes and seeds; a single best run is not evidence of improvement.
8. Device, illumination and acquisition-session drift are not represented in the current split audit.
9. Reproducibility of preprocessing/library versions and random seeds on the remote server remains to be recorded.
10. Deployment-time frame availability may differ from training; frame-count and quality distribution shift is unquantified.
11. **Manifest hash completeness failure found in this audit.** `artifacts/manifest/files.csv` contains 244 rows with blank SHA256 values. They are primarily `wmv*.avi` records absent from the local `.transfer/images512` derivative, so no hash could be computed. A total of 1,222 manifest records are absent from the current derivative. Non-blank available files show one duplicate hash group and zero valid train/val/test cross-fold hash leaks; this does not close the missing-video gap.
12. **Video provenance is not auditable locally.** Until the 244 videos are restored or explicitly excluded with a source-level statement, their duplicate/augmentation/same-case status cannot be proven.

## Task-model mismatch register

| Task | Current mismatch | Required intermediate object |
|---|---|---|
| clarity/blood colour | pooled semantic features for ordinal appearance | frame quality + ordinal head |
| exudation/hemorrhage | mean pooling for focal sparse events | frame MIL + top-k + uncertainty |
| capillary count | connected components treated as instances | instance detector/segmenter |
| crossing/malformation | image statistic treated as graph ratio | vessel graph + node/edge labels |
| diameter/length | regression without scale/centerline | pixel measurement + calibration |
| SVP/papilla | global features without anatomy ROI | landmark/ROI + ordinal head |

## Items deliberately not claimed

No current artifact proves clinical gold-standard semantics for external labels, absolute micron calibration, or 90%+ accuracy on all fields. The governed vascular assets are auxiliary supervision, not a replacement for these missing targets.

## Required next verification pass

Run a CPU-only audit that emits: feature provenance hashes, fold leakage intersections, per-fold class vocabularies/support, missingness tables, calibration curves, frame-count/quality shift summaries, and seed/version metadata. Only after this pass should P0 frame-level MIL and P1 graph/instance experiments be compared to v1.
