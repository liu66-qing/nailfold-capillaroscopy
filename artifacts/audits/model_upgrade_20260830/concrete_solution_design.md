# Concrete solution design: from pooled classifiers to identifiable estimators

## 1. Reframe the endpoints

Each endpoint must be mapped to an observation unit before choosing a model:

* **Frame-state endpoints:** clarity, blood colour, visible exudation/hemorrhage. The case label is a noisy aggregation of frame states.
* **Instance endpoints:** vessel count, vessel diameter, malformed-vessel status. These require one prediction per vessel instance.
* **Graph endpoints:** crossing ratio and malformation ratio. These are functions of an instance graph, not image-level classes.
* **Metric endpoints:** loop length and diameters. These are measurements in pixels followed by calibration, not semantic regression.
* **ROI morphology endpoints:** SVP and papilla. These require a localized anatomical crop and ordinal morphology.

This reframing is the solution to the current model mismatch; it is more important than selecting a larger backbone.

## 2. Minimal models that can be implemented now

### A. Frame-state MIL (P0)

Use frozen per-frame DINO/Hulu features plus quality scalars (blur, exposure, saturation, segmentation coverage). For each case and field, train a small gated-attention MIL head with **top-k pooling** (k=1,2,4 selected inside the training fold) and an `unreadable` gate. Use focal loss for sparse hemorrhage/exudation and CORAL/cumulative-link loss for ordered clarity/colour grades. The case probability is the weighted mean of frame probabilities, not the mean embedding. Train and select entirely inside the 5-fold development protocol.

### B. Vessel instance and graph head (P1)

Use the 2,332 confirmed pairs to train/fit a CPU-compatible feature extractor or use existing masks as auxiliary supervision. Extract connected components, skeleton endpoints, junction candidates, width profile and component confidence per frame. Build a graph where nodes are endpoints/junctions and edges are centerline segments. Predict crossing/malformation at nodes/edges, then compute ratios with a denominator confidence. If denominator confidence is below threshold, output `HOLD` rather than a forced class.

### C. Measurement head (P1)

Estimate diameter from robust width quantiles along centerline segments and length from geodesic skeleton distance. Keep values in pixels unless a calibration record exists. Fit only a fold-local residual correction model (Huber or quantile regression) and output prediction intervals. Absolute micron claims remain disabled under the current calibration status.

### D. ROI ordinal head (P2)

Create a deterministic ROI proposal from the vascular mask and image landmarks; reject crops with insufficient visible area. Train a low-capacity ordinal classifier for SVP/papilla on ROI features. Report adjacent-grade accuracy and weighted kappa in addition to BA; unresolved adjacent grades remain `HOLD`.

## 3. What the governed data can and cannot supervise

The 2,332 segmentation pairs can supervise vessel appearance and mask-quality features. The 89 detection images can supervise coarse vessel localization. They do **not** provide clinical truth for hemorrhage, exudation, papilla, SVP, or calibrated micron scale. Those heads must remain auxiliary or exploratory until matching labels exist.

## 4. Leakage-safe training recipe

For every outer test fold: fit preprocessing, top-k choice, quality thresholds, class weights, and residual calibrators using the other four folds only. Keep all frames from a case in one fold. Never use augmentation derivatives as independent cases. Store frame-level predictions, pooling weights, rejected frames, and case-level aggregation in an audit table.

## 5. Acceptance gates

* Frame-state fields: BA and macro-F1 both improve over the independent-head baseline in >=3/5 folds; minority recall must not decrease.
* Instance/graph fields: count MAE and ratio calibration error improve, and graph confidence coverage is reported; otherwise remain HOLD.
* Metric fields: MAE and tolerance-in-range improve in >=3/5 folds; no micron claim without calibration.
* ROI morphology: weighted kappa and BA improve with no adjacent-grade collapse.
* Any route that gains only on aggregate normalized score, default-value agreement, or teacher-forced chain predictions is rejected.

## 6. Priority order

Implement P0 first because it can use current features and directly attacks the frame/case mismatch. Implement P1 next because the governed segmentation data are most relevant to topology. Defer P2/P3 clinical release until ROI labels and calibration metadata are available. This is the shortest path that can produce a defensible improvement without touching v1 or locked data.
