# Auxiliary visual review protocol v1

## Scope
Review original images only. Do not infer diagnosis or absolute micrometre values.

## Labels
- measurement_visibility: complete / partial / unusable / unknown
- loop_boundary: complete / partial / absent / unknown
- clarity_review: clear / unclear / unknown
- annotation_confidence: high / medium / low

## Rules
1. `complete` requires both visible apex and both descending limbs; cropped or occluded bases are `partial`.
2. Blur, glare, severe background contamination, or insufficient vessel contrast is `unusable`.
3. Any disagreement or uncertain boundary is `unknown`, routed to manual review.
4. These are auxiliary image-observability labels, not clinical gold labels.

## Provenance
Each row must retain image path, SHA-256, reviewer identity, timestamp, rule version, and unknown reason.
