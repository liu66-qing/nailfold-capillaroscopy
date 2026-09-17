# Stage 2: Qwen3-VL Static-Field Baselines

## Scope and boundary

- Model: local `Qwen3-VL-8B-Instruct-MS`, 4-bit NF4, deterministic decoding.
- Input: four `CAPorg` nailfold images per case. No report image or original report text was supplied to the model.
- Development: 186 cases, five group-isolated folds. All 47 `locked_test` cases were excluded from prompts, examples, label schema, and metrics.
- Few-shot: two complete fixed-JSON demonstrations per fold, selected only from that fold's training cases. Demonstration images plus normalized target JSON were used; query labels were never used to choose examples.

## Protocol corrections

The first run was aborted because OCR/layout fragments were admitted as label classes. Do not use `qwen_stage2_cv_v1` for any result. `qwen_stage2_cv_v2/canonical_mapping.json` freezes the corrected medical-only label mapping and retains no report unit fragments.

## Results

Generation produced valid fixed JSON for every development case in both conditions. Five deterministic repeat cases were exact matches for both conditions. The generation-call latency was 2.80 s/case (zero-shot) and 2.86 s/case (few-shot); model load was 4.13 s and peak allocated GPU memory was 7.88 GB.

| Field | Zero macro-F1 / BA | Few-shot macro-F1 / BA | Interpretation |
|---|---:|---:|---|
| Clarity | 0.027 / 0.333 | 0.275 / 0.505 | Five-fold improvement, but the minority `模糊` class has only 8 cases. |
| Capillary count | 0.150 / 0.343 | 0.290 / 0.341 | Macro-F1 improves but balanced accuracy does not. |
| Crossing ratio | 0.062 / 0.249 | 0.252 / 0.284 | Rare classes are too sparse; no route decision. |
| Malformation ratio | 0.113 / 0.296 | 0.211 / 0.261 | Balanced accuracy declines. |
| Blood color | 0.379 / 0.415 | 0.360 / 0.415 | No improvement. |
| Exudation | 0.184 / 0.344 | 0.171 / 0.267 | Degrades. |
| Hemorrhage | 0.277 / 0.456 | 0.582 / 0.600 | Directional gain, but only 12 positive cases. |
| Venous plexus | 0.083 / 0.242 | 0.160 / 0.250 | No majority-fold balanced-accuracy gain. |
| Papilla | 0.205 / 0.303 | 0.184 / 0.331 | Metrics disagree. |
| Sweat duct | 1.000 / 1.000 | 1.000 / 1.000 | Only one retained target class; not interpretable. |

The prediction distributions show why report-format compliance must not be mistaken for medical understanding. In zero-shot, clarity was `模糊` for all 186 cases, venous plexus was `可见2排` for 183/186, and crossing ratio was `60--80%` for 162/186. Few-shot mostly changed these fixed priors toward its training demonstrations: count was `>=7` for 178/186, exudation and hemorrhage were `无` for all 186, and venous plexus was `不见` for 185/186.

## Decision

Qwen3-VL has excellent constrained-JSON control but this baseline does not demonstrate robust zero-shot image understanding. Keep QLoRA as an exploratory experiment only, with the frozen folds and field-specific metrics. Do not route a production field to Qwen from these results; any later routing claim must show both five-fold mean improvement and majority-fold directional consistency, while respecting rare-class limits.

## Artifacts

- `canonical_mapping.json`, `canonical_schema.json`, `class_counts.csv`, and `leakage_audit.json`
- `zero_shot_predictions.jsonl` and `few_shot_predictions.jsonl`
- `all_metrics_corrected.csv`, `five_fold_mean_metrics.csv`, and `few_shot_vs_zero_shot_five_fold.csv`
- `prediction_distributions.csv`, `field_mismatch_samples.json`, and `failure_samples.json`
