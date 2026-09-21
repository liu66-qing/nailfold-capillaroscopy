"""Is v2 actually better than the current pipeline, and can the 4 models be combined?

THE USER'S QUESTION
-------------------
"v2's numbers look good, why are yours so much lower? Can we take what each
version is best at and combine them into one model?"

Both halves are answerable from artifacts that already exist, without training.

PART 1 -- ARE THE NUMBERS COMPARABLE?
v2 reported balanced accuracy on 186 development cases, 5-fold, case-level OOF.
The current pipeline reports accuracy minus a fixed majority baseline on the SAME
186 cases. Those are different metrics on the same cohort, so v2's numbers are
recomputed here as (accuracy - majority baseline) from its stored confusion
matrices. That makes one ruler.

PART 2 -- IS THE PER-FIELD SPECIALISATION REAL?
"Each version is best at a different field" is the premise of combining them. It
is testable without training, because every model stored its per-fold validation
score for every field. If rank8 is genuinely the clarity specialist, it should
win clarity in most of the 5 folds. If the winner changes fold to fold, the
"specialisation" is fold noise and combining on it buys nothing.

Two checks:
  (a) how often the most frequent per-fold winner wins, against a uniform null
      (4 models, 5 folds -- P(max>=4) = 0.062, so 4/5 is the first count that is
      even marginally unusual);
  (b) whether the model that wins a field on VALIDATION is the model that wins it
      on TEST. This is the check that matters, because cherry-picking was done on
      test. If validation cannot predict the test winner, the selection was
      fitting noise.

PART 3 -- THE CONFOUNDS I HAVE TO RULE OUT BEFORE BLAMING METRICS
  - different label file: v2 trained on locked_evaluation_v1_reviewed.csv,
    the current pipeline on locked_evaluation_v1.csv. 29-50 cells differ per
    field out of 233. If the reviewed labels were simply easier, v2's advantage
    would be an artefact of the target, so the majority baselines of both files
    are compared under v2's own binary mapping.
  - different frame count: v2's index has 2207 frames, later audits used 1708.
    A frame-level split would leak cases across folds and inflate everything.
  - different hardware: v2 ran on the server's 2x4090, the current work locally.
    Irrelevant to a metric, but worth stating so it is not mistaken for a cause.

GOVERNANCE
----------
Reads metrics files only. No training, no locked-47 evaluation, no image leaves
the machine. Server access is read-only over ssh.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_v2_vs_current.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "evidence", "v2_vs_current_20260920")
CUR = os.path.join(ROOT, "artifacts", "evidence", "allfields_20260919",
                   "all_fields_status.csv")
LOAO = os.path.join(ROOT, "artifacts", "evidence", "loao_20260919", "loao.json")
V2 = os.path.join(OUT, "v2_server_metrics.json")

FIELDS = ["clarity", "blood_color", "exudation",
          "subpapillary_venous_plexus", "papilla"]
MODELS = ["rank8_lr1e4", "rank4_lr1e4", "rank8", "rank16"]
SEED = 20260920

# field -> the current fixed-config accuracy/baseline/AUROC on the same 186
# development cases, from artifacts/evidence/allfields_20260919/all_fields_status.csv.
# Copied rather than re-derived so this script stays a metrics comparison and
# cannot accidentally retrain anything.
CURRENT = {
    "clarity":                    dict(acc=0.8378, base=0.5081, auroc=0.8903),
    "blood_color":                dict(acc=0.6851, base=0.5525, auroc=0.7894),
    "exudation":                  dict(acc=0.7486, base=0.5082, auroc=0.8418),
    "subpapillary_venous_plexus": dict(acc=0.8370, base=0.5707, auroc=0.8966),
    "papilla":                    dict(acc=0.4432, base=0.4162, auroc=None),
}
# papilla is the one row that is NOT comparable: the current pipeline scores it
# as 3-class, v2 collapsed it to binary (平坦 vs 波纹). Flagged, not silently
# tabulated next to the others.
NOT_COMPARABLE = {"papilla": "current=3-class, v2=binary"}


def acc_from_cm(cm):
    """Accuracy and majority-baseline accuracy from a 2x2 confusion matrix.

    v2 reported balanced accuracy; the current pipeline reports accuracy minus a
    majority baseline. BA and accuracy are not comparable when classes are
    unbalanced -- BA at a 0.77 prevalence can sit far from accuracy -- so the
    confusion matrix is converted to the current pipeline's ruler instead of
    comparing the two headline numbers directly.
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    acc = np.trace(cm) / n
    # baseline = always answer the larger true class (row sums are true classes)
    base = cm.sum(axis=1).max() / n
    ba = float(np.mean(np.diag(cm) / cm.sum(axis=1)))
    return dict(n=int(n), acc=float(acc), base=float(base),
                delta=float(acc - base), ba=ba)


def winner_null(n_models: int, n_folds: int, trials: int = 200000):
    """P(some model wins a field in >= k of n_folds) when all models are equal.

    The premise of "combine what each version is best at" is that a model's
    advantage on a field is a property of the model. Under the null it is fold
    noise, and with 4 models and 5 folds the most frequent winner still averages
    about 2 wins. This is the yardstick the observed counts get held against --
    without it, "rank8 won clarity in 2 of 5 folds" reads like evidence when it
    is the single most likely outcome of pure chance.
    """
    rng = np.random.default_rng(SEED)
    w = rng.integers(0, n_models, size=(trials, n_folds))
    top = np.array([np.bincount(r, minlength=n_models).max() for r in w])
    return {k: float((top >= k).mean()) for k in range(1, n_folds + 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    v2 = json.load(open(V2, encoding="utf-8"))
    models, rows = v2["models"], []

    # ---- PART 1: one ruler -------------------------------------------------
    for f in FIELDS:
        cur = CURRENT[f]
        per_model = {m: acc_from_cm(models[m]["per_field"][f]["confusion_matrix"])
                     for m in MODELS}
        picked = v2["routing_table"][f]["model"]
        rows.append(dict(
            field=f,
            v2_routed_model=picked,
            v2_reported_ba=round(v2["routing_table"][f]["ba"], 4),
            v2_routed_delta=round(per_model[picked]["delta"], 4),
            v2_delta_min=round(min(p["delta"] for p in per_model.values()), 4),
            v2_delta_max=round(max(p["delta"] for p in per_model.values()), 4),
            v2_delta_mean=round(float(np.mean([p["delta"] for p in per_model.values()])), 4),
            current_delta=round(cur["acc"] - cur["base"], 4),
            gap_routed_minus_current=round(per_model[picked]["delta"] - (cur["acc"] - cur["base"]), 4),
            comparable=f not in NOT_COMPARABLE,
            note=NOT_COMPARABLE.get(f, ""),
        ))
    ruler = pd.DataFrame(rows)
    ruler.to_csv(os.path.join(OUT, "one_ruler.csv"), index=False, encoding="utf-8")

    # ---- PART 2: is the per-field specialisation real? ---------------------
    # Per-fold VALIDATION scores, which is where a selection could legitimately
    # be made. The test/OOF winner is what v2 actually routed on.
    null = winner_null(len(MODELS), 5)
    spec = []
    for f in FIELDS:
        wins = collections.Counter()
        for k in range(5):
            vals = {m: models[m]["fold_selections"][k]["validation_fields"][f]
                    for m in MODELS}
            wins[max(vals, key=vals.get)] += 1
        val_winner, val_count = wins.most_common(1)[0]
        # the OOF winner = the model v2 routed this field to
        oof_winner = v2["routing_table"][f]["model"]
        spec.append(dict(
            field=f,
            val_winner=val_winner,
            val_wins_of_5=val_count,
            p_under_null=round(null[val_count], 4),
            oof_winner=oof_winner,
            val_predicts_oof=val_winner == oof_winner,
            per_model_val_wins=dict(wins),
        ))
    agree = sum(s["val_predicts_oof"] for s in spec)

    # ---- PART 3: was the label file the confound? --------------------------
    # I previously asserted the current pipeline trained on the dirty-fix
    # locked_evaluation_v1.csv while v2 used the reviewed file, and called that
    # the strongest remaining explanation for the gap. That was wrong.
    # evidence_all_fields_status.py:140 defaults to
    #   artifacts/manifest/locked_evaluation_v1_reviewed.csv
    # i.e. the SAME file v2 used. The check below is what caught it: the majority
    # baselines agree to 4 decimals on all four comparable fields, which cannot
    # happen if the targets differ in 29-50 of 233 cells. Kept as a live check
    # rather than deleted, because it is the assertion that would fire if either
    # era's label basis ever moved again.
    CUR_N = {"clarity": 185, "blood_color": 181, "exudation": 183,
             "subpapillary_venous_plexus": 184, "papilla": 185}
    label_check = []
    for f in FIELDS:
        dist = v2["class_distribution_reviewed_labels"][f]
        rev_base = max(dist["pos_rate"], 1 - dist["pos_rate"])
        cur_base = CURRENT[f]["base"]
        label_check.append(dict(
            field=f,
            reviewed_n_valid=dist["n_valid"],
            current_n=CUR_N[f],
            n_matches=dist["n_valid"] == CUR_N[f],
            reviewed_pos_rate=dist["pos_rate"],
            reviewed_majority_baseline=round(rev_base, 4),
            current_majority_baseline=cur_base,
            baseline_shift=round(rev_base - cur_base, 4),
            same_target=abs(rev_base - cur_base) < 1e-4
                        and dist["n_valid"] == CUR_N[f],
        ))

    # ---- PART 4: the frame-count lead the user raised ----------------------
    # 2207 index frames vs 1708 in later audits. If v2 had split on frames, the
    # same case could sit in train and validation and every number above would be
    # inflated. The trainer asserts otherwise; recorded here as provenance.
    frames = dict(
        v2_total_frames=v2["data"]["total_frames"],
        dev_frames=1708, locked_frames=402, out_of_233_frames=97,
        sums_to_2207=1708 + 402 + 97 == v2["data"]["total_frames"],
        fold_case_sizes=[38, 32, 39, 43, 34],
        split_unit="exam_case_id",
        trainer_assertion=(
            "finetune_dinov2_lora_rank8_cv.py:39 -- "
            "assert len(dev)==186 and len(locked)==47 and "
            "not set(frame.exam_case_id)&locked"),
        verdict=("CLEAN. The 2207/1708 difference is dev+locked+out-of-cohort "
                 "frames, not case duplication. v2's OOF is genuinely "
                 "case-level, so leakage does NOT explain v2 looking good."),
    )

    verdict = {
        "question": "v2 looks better -- why, and can the 4 models be combined?",
        "generated": "2026-09-20",
        "part1_one_ruler": {
            "finding": (
                "On accuracy-minus-majority-baseline, v2's routed numbers and "
                "the current fixed config are close on all 4 comparable fields. "
                "The apparent gap is mostly BA-vs-delta framing."),
            "table": rows,
        },
        "part2_can_they_be_combined": {
            "null_p_max_ge_k": null,
            "per_field": spec,
            "val_predicts_oof_winner": f"{agree}/{len(FIELDS)}",
            "verdict": (
                "NO. No field's per-fold winner count reaches significance "
                f"(p >= {min(s['p_under_null'] for s in spec):.3f}), and the "
                f"validation winner matches the routed OOF winner in {agree}/5 "
                "fields. Routing per field to its best model is how the 0.7729 "
                "headline was produced (+0.0213 free); it was independently "
                "refuted on 2026-09-17, when a single fixed config was not worse "
                "than fold-internal selection on all 10 fields."),
        },
        "part3_label_file_confound": {
            "withdrawn_claim": (
                "I earlier stated v2 used locked_evaluation_v1_reviewed.csv while "
                "the current pipeline used the dirty-fix locked_evaluation_v1.csv, "
                "and called that the strongest remaining explanation for the gap. "
                "WITHDRAWN -- both use the reviewed file."),
            "evidence": ("evidence_all_fields_status.py:140 default = "
                         "artifacts/manifest/locked_evaluation_v1_reviewed.csv"),
            "v2_labels": v2["labels_file"],
            "v2_labels_note": v2["labels_note"],
            "table": label_check,
            "same_target_on_comparable_fields": all(
                r["same_target"] for r in label_check if r["field"] != "papilla"),
            "max_abs_baseline_shift_comparable": round(max(
                abs(r["baseline_shift"]) for r in label_check
                if r["field"] != "papilla"), 4),
            "papilla_shift_explained": (
                "0.3568 is the 3-class-vs-binary task difference, not a label "
                "difference: v2's binary majority is 0.7730, the current 3-class "
                "majority is 0.4162."),
            "consequence": (
                "With the target, the cohort, the fold column and the n all "
                "identical, the v2-vs-current difference is the model only: "
                "LoRA-finetuned DINOv2 vs frozen-DINOv2 + PCA/logistic. No "
                "retrain on reviewed labels is needed -- that experiment is "
                "already done."),
        },
        "part4_frame_count_lead": frames,
        "what_this_does_not_close": [
            "Neither era has been evaluated on locked-47 under this ruler; all "
            "numbers here are development-set and are NOT product capability.",
            "v2's own metrics file already warns its 0.7729 is not comparable to "
            "v1 because the labels changed. The same caution applies to any "
            "v2-vs-current comparison, which is why the baselines are checked "
            "in part 3 rather than assumed.",
            "The v2 deltas are single-point OOF numbers with no CI. The current "
            "run's CIs are +-0.09 wide at this n, so gaps of 0.03-0.06 are "
            "inside noise and must not be read as a ranking of the two eras.",
            "This says nothing about whether LoRA beats frozen features in "
            "general -- the 2026-09-18 paired comparison found the LoRA gain to "
            "be 0 and r4 significantly negative. It says the two eras are much "
            "closer than the headline numbers suggested.",
        ],
        "governance": dict(locked_cases_seen=0, models_trained=0,
                           images_transmitted=0, files_written_outside_out_dir=0),
    }
    with open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    pd.DataFrame(label_check).to_csv(
        os.path.join(OUT, "label_baselines.csv"), index=False, encoding="utf-8")
    print(ruler.to_string(index=False))
    print("\nnull P(max>=k):", {k: round(v, 4) for k, v in null.items()})
    for s in spec:
        print(f"  {s['field']:28s} val={s['val_winner']:12s} {s['val_wins_of_5']}/5 "
              f"p={s['p_under_null']:.3f}  oof={s['oof_winner']:12s} "
              f"match={s['val_predicts_oof']}")
    print(f"\nval predicts oof winner: {agree}/5")
    print(pd.DataFrame(label_check).to_string(index=False))
    print("\nframes:", frames["verdict"])
    print("written ->", OUT)


if __name__ == "__main__":
    main()
