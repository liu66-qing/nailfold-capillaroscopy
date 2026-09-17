# A2 end-to-end LoRA + case-level attention pooling failure report

Status: `FAIL`; confidence: `HIGH`; development-only OOF; locked cases seen: 0.

## What was tested

Existing LoRA pipelines train at frame level and average logits at inference, so the
gradient never sees a case. A2 makes the case the unit of the forward pass: all frames of
one case (median 9, capped at 12) go through the frozen DINOv2 ViT-B/14 with LoRA (rank 8,
lr 1e-4, last 4 blocks `attn.qkv`+`attn.proj`), are pooled by a per-field gated attention
branch (`tanh(Vh) * sigmoid(Uh)`, softmax over frames), and are classified by 7 multiclass
heads. Loss is computed at case level, so attention weights are supervised directly by
case labels. Backbone stayed frozen throughout (E7 showed unfreezing overfits).

This is not a rerun of E5: E5 pooled **frozen** precomputed embeddings, whereas here the
representation is trained jointly with the pooling.

Protocol: `development_fold` 0..4 as test, val = `(test+1)%5`, remaining 3 folds train;
epoch selected on val only; mode baseline computed on the 3 training folds. 186 development
cases / 1708 frames; the 47 locked cases are filtered before any tensor is built and
asserted disjoint. 3 seeds (17/29/43), AMP bf16, 8-case gradient accumulation, grad clip
2.0, label smoothing 0.04, `sqrt(N/count)` class weights. Peak VRAM 3.0 GiB on GPU 0.

The required control arm is identical code, identical folds and seeds, with gated attention
replaced by non-learnable mean pooling.

## Results (3-seed mean OOF accuracy)

| Field | attention acc | mode baseline | attention delta | mean-pool delta | attention - mean | folds delta>=0.05 | adj1 | QWK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exudation | 0.650 +- 0.033 | 0.520 | +0.130 | +0.137 | **-0.008** | 4.7/5 | 1.000 | +0.358 |
| clarity | 0.721 +- 0.018 | 0.470 | +0.250 | +0.288 | **-0.038** | 4.7/5 | 1.000 | +0.442 |
| blood_color | 0.628 +- 0.011 | 0.468 | +0.160 | +0.166 | **-0.006** | 3.7/5 | 1.000 | +0.224 |
| subpapillary_venous_plexus | 0.564 +- 0.037 | 0.381 | +0.182 | +0.199 | **-0.017** | 5.0/5 | 0.972 | +0.325 |
| microthrombus | 0.578 +- 0.023 | 0.597 | -0.018 | -0.055 | +0.037 | 0.3/5 | 0.871 | +0.154 |
| papilla | 0.405 +- 0.045 | 0.425 | -0.020 | +0.015 | **-0.035** | 1.0/5 | 0.877 | +0.062 |
| capillary_count | 0.698 +- 0.008 | 0.687 | +0.011 | +0.005 | +0.005 | 0.3/5 | 0.932 | +0.127 |

Mean `attention - mean` across the 7 fields: **-0.009**.

Per-class recalls (n>=10), per-fold deltas, and the classes held out as `rare_classes`
(exudation `+++` n=4, capillary_count `<=4` n=15 kept, subpapillary `>2排` n=27 kept) are in
`metrics.json`. Attention weights per case per frame are in `attention_weights.npz`.

## Gate

Requirement: >=4/7 fields with `delta >= 0.05`, >=3/5 folds same direction, **and** an
improvement over the mean-pooling control.

Four fields (exudation, clarity, blood_color, subpapillary_venous_plexus) clear the delta
and fold-stability conditions. **All four fail the control-arm condition** - mean pooling
matches or beats gated attention on each. Fields passing all three conditions: **0/7**. The
gate is not met.

## Attribution

The learned attention collapsed to uniform. Mean normalized entropy of the attention
distribution is 0.972-0.980 (1.0 = exactly uniform) and mean max frame weight is ~0.19
across all 7 fields. The gated branch converged to an expensive reimplementation of mean
pooling, then lost a few tenths of a point to the extra parameters and the noisier
optimization. Making the loss case-level therefore did not by itself give attention
anything to latch onto.

The plausible reason is that case labels carry no frame-localized signal to learn from: a
single case-level class per field over ~9 frames provides roughly one bit of supervision
per field per case, against 186 cases. With 3 training folds (~110 cases) there is not
enough signal to identify which frames matter, so the minimum-risk solution is to weight
them equally. This is consistent with E2 (no pooling variant beat mean) and E5 (frozen
attention MIL 0.679 < 0.684 baseline): the failure survives making the features trainable,
which rules out "frozen features" as the explanation for E5 and points at the label
granularity instead.

Note the four strong fields do beat their mode baselines by a wide margin under **both**
pooling modes, which says the end-to-end case-level multiclass setup itself is sound. The
rejected component is specifically gated attention pooling.

## Conclusion

A2 is rejected with HIGH confidence. Do not adopt case-level gated attention pooling. Frame
aggregation is now closed under three independent designs (E2 heuristic pooling, E5 frozen
attention MIL, A2 end-to-end trained attention). Any further work on aggregation needs
frame-level or region-level labels first; without them, mean pooling is the correct default.

No comparison to baseline v2 (0.799) is made anywhere in this report: that route solves a
different task with different class counts and is not comparable.

## Artifacts

- `metrics.json` - both arms, 3 seeds, per-field and per-fold, `locked_cases_seen: 0`
- `attention_weights.npz` - 7812 arrays, per arm/seed/case/field
- `a2_train.py` - training script
- `train.log` - full run log
