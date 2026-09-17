# Auxiliary visual review protocol v2

Supersedes `../VISUAL_REVIEW_PROTOCOL.md` (v1). v1's labels are void — see
`visual_review_audit_v1_findings.json`.

## Status of these labels

Auxiliary image-observability labels. **Never a clinical gold label.** Every
admitted image is `PASS_AUXILIARY_CONDITIONALLY` under
`vascular_dataset_governance_20260830`, so downstream use is auxiliary-only.
Three label layers stay physically separate and are never merged:
original clinical labels / existing human XML boxes / these AI auxiliary labels.

## Scope

Original images only. Do not infer diagnosis, disease category, or absolute
micrometre values. Judge only what is visible in the image.

## Fields

Each image carries 2–13 vessel boxes (median 7), so **every judgement is an
explicit fraction over the boxes present**, not a single-vessel call. This is
the main fix over v1, where the rule was written per-vessel but recorded
per-image with no aggregation rule.

| Field | Values |
|---|---|
| `measurable_vessel_fraction` | `none` / `minority` / `majority` / `all` / `unknown` |
| `apex_visible_fraction` | `none` / `minority` / `majority` / `all` / `unknown` |
| `clarity_review` | `clear` / `unclear` / `unknown` |
| `image_usable_for_measurement` | `usable` / `unusable` / `unknown` |
| `annotation_confidence` | `high` / `medium` / `low` |

`minority` = fewer than half the boxes; `majority` = half or more but not all.

The first two fields are deliberately disjoint: a loop can have a visible apex
yet be unmeasurable (limb cut off at frame edge), and can be measurable in
width while its apex is occluded. In v1 the two corresponding fields were
identical on 16/16 rows and carried zero independent information. **If you find
yourself giving both the same value on most images, stop and report it** — that
means the distinction is not operational and the fields should be merged.

## Rules

1. Count against `vessel_box_count` in the queue row. A vessel counts as
   measurable only if apex and both descending limbs are inside the frame and
   distinguishable from background.
2. Blur, glare, severe background contamination, or vessel/background contrast
   too low to trace an outline → that vessel is not measurable. If this applies
   to all boxes, `image_usable_for_measurement = unusable`.
3. Any uncertainty about a boundary → `unknown` on that field, with a written
   `unknown_reason`. `unknown` is a valid answer and is not penalised.
4. `annotation_confidence = low` requires the same treatment as `unknown` —
   the row is routed to human review and excluded from agreement statistics.
   (v1 recorded 4 `low` rows and still counted them as reviewed.)
5. Record `review_resolution_px` as the actual pixel dimensions at which you
   viewed the image. Do not judge clarity or contrast from a thumbnail.
   v1's contact sheet gave ~240×190 per tile, about 6% of source pixels;
   that is not sufficient for rules 1–2.
6. Do not consult the heuristic quality label. It is withheld in
   `heuristic_predictions_heldback.csv` and joined only after review closes.
   v1's reviewer saw the heuristic's own `usable` picks and confirmed 16/16 of
   them, which made the exercise circular.

## Review order

Follow `review_order` (seed 20260917, recorded in the manifest). Do not skip
ahead and do not reorder by apparent quality — v1's 16 rows were exactly the
first 16 heuristic-`usable` images in filename order, which is a deterministic
selection, not a sample. The order already mixes both heuristic strata
(48 `usable` / 34 `uncertain` withheld), so the heuristic itself becomes
testable once review closes.

## Provenance

Every row must retain: `image_path`, `image_sha256`, `exam_case_id`,
`reviewer`, `review_timestamp`, `protocol_version`, `review_resolution_px`,
and `unknown_reason` where applicable.

## Acceptance gate — set before reviewing, not after

### What the XML boxes cannot adjudicate

Measured across all 82 admitted images before writing this gate:

- **box presence: 82/82 `usable`.** Single category, so kappa against it is
  undefined (~0) regardless of reviewer quality. Unusable as a gate.
- **in-frame box fraction: 76 `all` / 6 `majority`.** So skewed that 3
  disagreements take weighted kappa from 1.00 to 0.78, while a reviewer
  answering `all` on every image scores 0.00. A threshold on this measures
  marginal skew, not agreement.

Both thresholds originally drafted here were therefore withdrawn. This is a
property of the reference, not of the reviewer: these images were selected for
annotation *because* they contain traceable vessels, so the XML cannot speak to
whether an image is measurable.

### What the XML boxes can adjudicate

Re-measured on the 129-image pool after tier-2 admission. The thresholds were
frozen *before* the pool grew and are unchanged; this is the check that growing
the pool did not quietly degrade the reference.

- **`vessel_box_count`**: spread 2–13, all 12 values present, median 7
  (tier 1 alone: 2–13, 12 values; tier 2 alone: 2–12, 11 values).
- **per-image class presence**: `cross_vessel` **72/129 = 55.8%** (was 40/82 =
  48.8%), `malformed_vessel` 102/129 = 79.1%. `cross_vessel` moved *closer* to
  the 50/50 split at which kappa is most informative.
- **per-box geometry**: 569 tier-1 boxes, relative area median 0.0093 (IQR
  0.0061–0.0139), usable for IoU if the reviewer draws boxes.

### Gate

| Check | Reference | Threshold |
|---|---|---|
| vessel count | `vessel_box_count` | Spearman ρ ≥ 0.70 **and** ≥ 60% of images within ±1 |
| `cross_vessel` present | XML class presence | Cohen's kappa ≥ 0.50 |
| box localisation (only if boxes drawn) | 569 XML boxes | IoU ≥ 0.50 on ≥ 80% |
| **case-collapsed repeat** | same three, one row per case | must also pass |
| independent cases | `exam_case_id` | ≥ 20 |

The case-collapsed row is not redundant. 129 images rest on only **49 cases**,
and 6 clusters are resized siblings of one another, so a per-image statistic
counts the same patient up to 9 times. The scorer reports all three quantities
on four slices — per-image, case-collapsed, near-duplicate-removed, and
tier-1-only — and the verdict requires the case-collapsed slice to pass.

Verified on synthetic reviewers (`scripts/score_visual_review_agreement.py`):

| Synthetic reviewer | per-image ρ / within±1 / κ | case-collapsed | verdict |
|---|---|---|---|
| good (±1 noise, 8% class flips) | 0.965 / 1.00 / 0.812 | 0.972 / 1.00 / 0.900 | PASS |
| sloppy counts + random class | 0.709 / 0.364 / −0.137 | 0.801 / 0.449 / −0.028 | FAIL |
| constant answer (the v1 failure mode) | — / 0.341 / −0.00 | — / 0.286 / −0.00 | FAIL |
| perfect | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | PASS |

Note the sloppy reviewer passes ρ on both slices and is caught only by within±1
and kappa — a single-statistic gate would have let it through.

`measurable_vessel_fraction`, `apex_visible_fraction`,
`image_usable_for_measurement` and `clarity_review` have **no valid reference in
this dataset**. They are still collected, but they are `unvalidated` — they may
not be used to justify any downstream claim, and any report citing them must
say the agreement is unmeasured. Validating them needs a second independent
human reviewer, which is out of scope here.

Rows with `annotation_confidence = low` or any `unknown` are excluded from both
numerator and denominator, and the exclusion count is reported.

**Below gate → labels are not used, and the failure mode is written up.**
Above gate → auxiliary use only, and only for the two targets that are genuinely
image properties: image quality / measurability routing, and `loop_length`
relative grade. Never for clinical fields whose truth is a physician's judgement.

A pass here means the reviewer can count and classify vessels consistently with
a human annotator. It does **not** establish that the measurability judgements
are correct — see the `unvalidated` list above.

## Pool limit

**129 images / 49 development cases**, in two admission tiers.

| Tier | Images | Cases | Provenance evidence |
|---|---|---|---|
| 1 | 82 | 49 | byte-exact MD5 match to archive original |
| 2 | 47 | 25 | 32×32 grayscale cosine match, validated 123/123 against MD5-known cases |
| total | **129** | **49** | |

### Tier 2 adds images, not patients

All 25 tier-2 cases are already among the 49 tier-1 cases. Images per case rises
from mean 1.67 (max 4) to mean 2.63 (max 9); the case count does not move.

This matters for what the pilot can conclude. Agreement is only independent at
the case level, so **the 95% CI on a kappa near 0.6 is still governed by 49
cases — roughly ±0.15 — exactly as it was at 82 images.** Tier 2 buys
within-case reliability (the same patient seen at several resolutions), not
statistical power. A 300-image pilot remains unreachable: 326 of the 449
`UNMAPPED_LOCAL_SOURCE` images are still provenance-undecidable.

An earlier note in this project projected "129 images / 74 cases". That was
arithmetic error — 49 + 25 double-counted a subset — and is retracted.

### What discharged the tier-2 HOLD

The governance HOLD listed three clauses. Each has a citable artifact:

| Clause | Discharged by |
|---|---|
| source case | `provenance_recovery.csv`, perceptual match, 123/123 agreement with MD5-known cases |
| permission | data owner authorisation, 2026-09-17, recorded verbatim in the queue manifest |
| label semantics | `semantic_definition_evidence.json`; tier-2 XMLs use only the three already-defined classes (`vessel` 142, `malformed_vessel` 110, `cross_vessel` 49), so no new semantics enter |

Plus one bound the HOLD did not ask for but locked-test integrity does:

**locked-contamination gate** (`tier2_locked_safety.json`). Locked-specific
margin = similarity to the assigned development case minus similarity to the
nearest locked-47 original. Calibrated on labelled data, not judgement:

- 32 MD5-proven **locked** images, scored adversarially against their
  best-matching development case: margins −0.655 to **−0.043**
- 82 MD5-proven **development** images against their proven case:
  **+0.043** to +0.717
- AUC 1.000, zero overlap, empty gap of width 0.087 spanning zero

Threshold 0.02 sits inside that gap: 100% of adversarially-mislabelled locked
images caught, 0% of proven-clean images discarded. All 47 tier-2 images pass.

Two checks were **withdrawn** during calibration:

- *margin ≥ 0.10* (my first guess) — refuted: 22 of the 82 MD5-**proven-clean**
  images fail it. The tier-2 and proven-clean distributions are nearly identical
  (median 0.1617 vs 0.1600), so 0.10 rejected clean data.
- *assigned case must not resemble a locked case above 0.98* — refuted as
  non-discriminating: proven-clean development cases reach 0.9696 cross-similarity
  while two genuinely **different locked** cases reach 0.9577. Case-level
  appearance overlap is the baseline texture of nailfold imagery.

### What authorisation does not cover

Permission to annotate is not permission to treat the output as truth or to
send it anywhere. `clinical_gold_standard` is `False` on every row. Locked-47
images remain excluded at case level regardless of authorisation. Transmitting
any of these patient images to a third-party API is a separate decision and is
**not** granted by this protocol.

Residual risk that no appearance test can bound: if a locked-47 patient also
appears in the archive under a different case id with different frames, tier 2
could carry them. Perfect separation on 114 labelled images bounds the per-group
error rate at roughly <3% at 95% confidence, not at zero. This is why tier 2 is
labelled in the queue (`admission_tier`) and why every result must be reported
with the tier-1-only value alongside it.
