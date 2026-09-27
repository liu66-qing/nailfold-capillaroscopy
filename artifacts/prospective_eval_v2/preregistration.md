# Temporal extrapolation agreement evaluation of the frozen v2 model on future same-device examination reports

Preregistration. Written on 2026-09-28, before the first prospective case.
No line of this file may change after the first case is logged; amendments go in a dated appendix, and any case logged before an amendment is scored under the version it was logged under.

## 0. What this measures and what it does not

- It measures agreement between the frozen image model and the examination report that is produced later.
- The report is a device and doctor reference standard. It is not independent clinical truth.
- Where a report field is computed by the device software, the result measures "reproducing the device output".
- Where a report field is entered by a doctor reading the image, the result measures "reproducing the doctor's report".
- Neither is accuracy about the patient's health state.
- External reporting uses three layers:
  1. report agreement (this evaluation);
  2. cross-device generalisation (only with cases from a different device, scored separately);
  3. clinical validity (no data; no claim).

## 1. Frozen artefacts

| Item | Value |
|---|---|
| Bundle | `artifacts/models/rag_heads_v2/bundle.joblib` |
| Bundle sha256 | `a413286055a8d2f09efbd59d2a25eb4bdb049f6efffab9b756b7a66dbd3d8177` |
| Encoder weights sha256 | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` (DINOv2-B/14, direct resize 518×686) |
| Code | `src/nailfold_report/rag_inference.py` at the commit that adds this file |
| Training | 186 dev + 47 former locked = 233 cases; no untouched internal test set remains |
| Heads | StandardScaler → PCA64 → LogReg C=0.03, per pooling, then mean |
| Threshold | 0.5 |
| Abstain margins | stored per field in the bundle |
| Device recentring | only when the device detector fires (≥12 cases, q99 0.2835) |
| RAG routing | the `role` of each field in the bundle; `loop_length` is in `withheld` |

- The bundle file is git-ignored (`*.joblib`). The sha256 above is the binding reference.
- A copy must be kept outside the repo.
- Any logged line whose `bundle_sha256` differs is excluded from scoring and counted as a protocol deviation.

## 2. Entry to the test queue

- **Timing.** Only examinations performed after the freeze, which is the commit time of this file, enter the queue.
  Examinations dated on or before 2026-09-28 are never test cases, even if they were never seen.
- **Same device.** The same device model and report template as the three recovered archives enter the queue.
  Any other device is logged into a separate cross-device queue with the same rules and scored separately.
- **One person, one case.** The first examination per `subject_key` is kept.
  - `subject_key` is the patient ID if one exists.
  - Otherwise it is a stable pseudonym of the examined person.
  - If neither exists, the key is name, sex and birth year; with no name, it is the examined person together with the exam date.
  - Later examinations of the same key are logged but not scored.
- **Inclusion.** Every examination that produces at least one CAPorg image is included.
  - A failed inference is logged with `error` set and scored as abstained, and it counts in every denominator.
  - Nothing is removed for image quality; clarity is itself an endpoint.

## 3. Procedure

1. Inference runs at examination time through `RagFieldPredictor.predict`.
   `ShadowLog.record` appends one line with both the RAG fields and all shadow fields (`last_raw`, which includes the withheld `loop_length`).
2. Inference never reads the report. The log is append-only; no line is edited or deleted.
3. Report labels are extracted later by the existing OCR / RTF pipeline.
   They are paired with the log by `exam_id` in a separate script that is not run until the stopping point.
4. Before the stopping point, only label-free operational counts may be inspected: cases logged, error rate, abstention rate and device distance.
   No agreement metric is computed or looked at.
   No model, threshold, abstention rule or routing change is allowed; any such change ends this evaluation and starts a new one with a new freeze.

## 4. Stopping point

- The queue closes at **N = 150 scored persons**, all fields evaluated once at that point.
- If N is not reached by 2027-09-28, it is evaluated once at that date with whatever has accrued, and reported as underpowered.
- There is no interim look, no extension and no dropping of fields.
- The cross-device queue has its own N = 150 (or the same date limit) and is never pooled with the same-device queue.

## 5. Endpoints, fixed per field

- **Reference label.** The report value, mapped exactly as in `scripts/fit_rag_heads.py` `target()` (loop_length: > 250 µm).
- **Primary endpoint.** `clarity`, used as a quality gate. Everything else is secondary.
- **What is reported for every field.** Every field is reported even if it fails:
  - coverage;
  - BA and per-class recall on answered cases, with case-bootstrap 95% CI (2000 resamples);
  - full-coverage BA with abstentions scored as wrong;
  - the correct / wrong / abstained split;
  - abstention rate by true class;
  - calibration, as the gap between mean predicted probability and observed positive rate on answered cases.

| Field | Role in RAG | Min answered pos / neg | Min coverage | Pass: answered BA | Pass: each recall | Pass: CI lower bound | Max calibration gap | Max CI half-width |
|---|---|---|---|---|---|---|---|---|
| clarity | quality gate (primary) | 30 / 30 | 0.90 | ≥ 0.70 | ≥ 0.60 | > 0.60 | 0.10 | 0.10 |
| subpapillary_venous_plexus | 报告档位参考 | 30 / 30 | 0.90 | ≥ 0.70 | ≥ 0.60 | > 0.60 | 0.10 | 0.10 |
| exudation | 报告档位参考 | 30 / 30 | 0.90 | ≥ 0.65 | ≥ 0.55 | > 0.55 | 0.10 | 0.10 |
| blood_color | 报告档位参考 | 30 / 30 | 0.45 | ≥ 0.70 | ≥ 0.60 | > 0.60 | 0.10 | 0.12 |
| malformation_ratio | 报告档位参考 (locked CI crossed 0) | 30 / 30 | 0.45 | ≥ 0.65 | ≥ 0.50 | > 0.55 | 0.10 | 0.12 |
| microthrombus | shadow only; static correlation, never advice | 30 / 30 | 0.45 | ≥ 0.65 | ≥ 0.50 | > 0.55 | 0.10 | 0.12 |
| loop_length (> 250) | shadow only; withheld | 30 / 30 | 0.45 | ≥ 0.75 | ≥ 0.65 | > 0.60 | 0.10 | 0.12 |

- **Evidence rule.** If a field does not reach its minimum answered positives or negatives, or its minimum coverage, at the stopping point, it is reported as **insufficient evidence**. It is neither a pass nor a fail, and the queue is not extended for it.
- **Pass wording.** A pass permits at most the wording "报告档位参考 / 图像相关性提示" for that field. It never permits health-risk or disease wording.
- **What a pass does not change.** `microthrombus` stays out of advice regardless of its result, because its unit (count / min) needs dynamic observation.
  `loop_length` may enter RAG only if it passes here, and only as "报告档位相关性提示".

## 6. Label source per field (must be filled in before the first case)

| Field | Source | Confirmed by | Date |
|---|---|---|---|
| clarity | ☐ device software ☐ doctor entry ☐ mixed | | |
| subpapillary_venous_plexus | ☐ device software ☐ doctor entry ☐ mixed | | |
| exudation | ☐ device software ☐ doctor entry ☐ mixed | | |
| blood_color | ☐ device software ☐ doctor entry ☐ mixed | | |
| malformation_ratio | ☐ device software ☐ doctor entry ☐ mixed | | |
| microthrombus | ☐ device software ☐ doctor entry ☐ mixed | | |
| loop_length | ☐ device software ☐ doctor entry ☐ mixed | | |

Results are labelled "reproduces device output" or "reproduces doctor report" according to this table.

## 7. Known risk stated in advance

In the internal paired LOAO (`artifacts/experiments/v1_vs_v2_training_20260928/`), adding the 47 former locked cases changed exudation BA by −0.035, CI [−0.074, 0.000].
v2 is kept as the single frozen configuration anyway, because choosing per field between v1 and v2 on that same LOAO would be selection on the evaluation.
The prospective result is the test of that choice.

## 8. After scoring

- The 150 scored cases join training for v3.
- v3 is frozen with its own preregistration.
- Cases after v3's freeze form the next test queue.
- Each prospective batch is scored exactly once.
