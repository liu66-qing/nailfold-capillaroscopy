# External adversarial review — 2026-09-20

`astra_review_raw.md` is the unedited reply from an independent model
(`gpt-6-astra`, via Codex CLI 0.155.1, read-only sandbox), asked to review the
project's status adversarially and to name reasoning errors. It was given the
established experimental results, both external-dataset audits from the same day,
and five direct questions. It read no patient data.

It is kept verbatim because its value is as an outside check, and editing it
would destroy that. It is an opinion, not evidence: where it asserts something
testable, the test still has to be run here.

## Where it agrees with our own measurements

- Architecture search is not the path. It independently reaches "capacity
  fixation" as the error to avoid, and rules out more backbones, LoRA ranks,
  unfreezing, loss tricks, temporal input, and geometric features as the main
  rescue — the same list our paired comparisons already refuted.
- Retinal-vessel transfer cannot fix a detector whose annotations omit abnormal
  vessels; it would produce "a more confident model trained against the same
  wrong target".
- Neither external dataset is independent validation, and treating either as
  disease evidence would be a serious error.
- A 2-field product is legitimate only with narrow positioning, and clarity is a
  safety/gating feature rather than a health finding.

## Where it pushes back on us

Its substantive criticism is of **our acceptance ruler**, and it does not say the
ruler is too strict. It says the ruler is *too crude in one specific way*: making
"bootstrap CI excludes 0" the sole gate confounds usefulness with sample size and
ignores asymmetric harm. What it would add:

- a pre-specified minimum *meaningful* effect, not merely a positive one;
- CIs on sensitivity, specificity, calibration, and abstention rate, not accuracy
  alone;
- per-subgroup and per-device reporting;
- a defined "insufficient image quality" behaviour.

That is a tightening, not a loosening — and the first item is actionable here: we
have never declared a minimum effect size in advance, only tested against zero.

It explicitly refuses to relax independent validation, device robustness,
pre-defined thresholds, honest abstention, separation of image-quality outputs
from physiological findings, and the ban on disease claims without independent
patient labels. On the "this makes delivery impossible" argument it is blunt:
that is not evidence the ruler is too strict, it may mean the intended product
claims exceed what the data can support.

## Limitations of this review

- It saw our results as a written summary, not the artifacts, so it cannot verify
  any number it was given.
- Its recommended next action — a prospective, device-documented, independently
  labelled collection with double annotation — is a data-collection programme,
  not something executable from this repo. It is the same conclusion our own
  deployment-bar analysis reached, so it adds corroboration, not a new option.
- It was asked leading questions in places (e.g. question 4 embeds our own
  diagnosis of the detector). Its agreement there is weaker evidence than its
  unprompted criticism of the ruler.
