"""papilla: does whole-image DA (A1) plus automatic local morphology (A2) beat A1?

Pre-registered in artifacts/experiments/rescue_external_20260922/
preregistration_papilla_fusion.md, written before this script existed. Four arms
only -- A0, A1, A2, A1+A2 -- no pooling search, no attention, no hyperparameter
search, no class-scheme change. Everything comes from the shipped ruler by import,
so nothing here can quietly re-tune it.

The reference arm for the keep/discard decision is A1, NOT A0. A1 already carries
+0.0967 BA over A0; comparing the fusion against A0 would re-report that gain as
if the fusion had produced it. That is exactly the weak-anchor inflation the
external-DA ladder round had to retract.

Two LOAO answers are reported, because they are different questions:
  - partition: pooled 5-fold OOF split by archive. Every archive was in training
    when its own cases were predicted, so this asks "does the advantage live in
    one archive".
  - refit: the archive is held out of TRAINING entirely and the model refits on
    the other two. Only this asks "does it survive a site it never saw". Its
    baseline is the majority answer of the TRAINING archives, because that is what
    a deployed model carries; the held-out archive's own prevalence is unknown at
    deployment time.

development case-level OOF. Not a product capability claim. locked-47 is not read.

Run:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src \
    /c/Users/liujunqing/anaconda3/envs/pytorch_gpu/python.exe \
    scripts/run_papilla_fusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402
    ANCHOR, LABELS, N_BOOT, POOLINGS, SEED, UNIT, fit_predict, fold_directions,
    hard, load_arm, oof, paired, score, target,
)
from run_rescue_ladder import (  # noqa: E402
    BELOW_RESOLUTION, MIN_EFFECT, abstention, calibration, loao, verdict,
)
from run_rescue_ladder_local import attach, load_local  # noqa: E402

FIELD = "papilla"
A1_ARM = "dinov2b_da_external"
LOCAL_TAG = "local"          # the A2 table: detector trained on local human boxes
EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
OUT = EXP / "papilla_fusion"

# Pre-registered decision rule. All three must hold for A1+A2 to be kept.
KEEP_RULE = dict(
    reference_arm="A1",
    ba_gain_min=MIN_EFFECT,
    ba_ci_must_exclude_zero=True,
    no_per_class_recall_may_drop=True,
    loao_no_direction_may_drop_by=0.05,
)

# From ladder.json / ladder_local.json, recorded in the pre-registration before
# this run. Asserted so a fusion built on drifted inputs fails loudly instead of
# being compared against numbers that are no longer true.
EXPECTED = {
    "A0": dict(n=185, accuracy=0.4432, baseline_constant=0.4162,
               balanced_accuracy=0.4149),
    "A1": dict(n=185, accuracy=0.5297, baseline_constant=0.4162,
               balanced_accuracy=0.5115),
    "A2": dict(n=185, accuracy=0.4703, baseline_constant=0.4162,
               balanced_accuracy=0.4567),
}


def recall_drops(cand: dict, ref: dict) -> dict:
    """Which classes lost recall relative to the reference arm.

    A BA gain that comes with a class losing recall is a re-allocation between
    classes, not an improvement -- A2 already does this to class 1 (0.5065 ->
    0.4675). papilla is a 3-class field and stays 3-class, so every class is
    checked; a field cannot be rescued by quietly abandoning its smallest class.
    """
    out = {}
    for k in sorted(set(cand) | set(ref)):
        c, r = cand.get(k), ref.get(k)
        if c is None or r is None:
            out[k] = None
            continue
        out[k] = round(float(c) - float(r), 4)
    dropped = [k for k, v in out.items() if v is not None and v < 0]
    return dict(per_class_delta=out, classes_dropped=dropped,
                any_class_dropped=bool(dropped))


def refit_loao(base_feats, base_ix, extra, y_case, order, arch_arr, classes,
               multiclass):
    """Hold each archive out of TRAINING entirely and refit on the other two.

    The ladder's loao() never refits; it partitions predictions made with all
    three archives in the training pool. Both are reported, but only this one
    speaks to a site the model never saw. The baseline is the majority answer of
    the TRAINING archives, since the held-out archive's prevalence is not
    available at deployment time.
    """
    feats = base_feats if extra is None else attach(base_feats, base_ix, extra)
    dm = base_ix.exam_case_id.isin(y_case.index).to_numpy()
    out = {}
    for a in sorted(set(arch_arr)):
        te_cases = {c for c, ar in zip(order, arch_arr) if ar == a}
        tr_cases = set(order) - te_cases
        trm = dm & base_ix.exam_case_id.isin(tr_cases).to_numpy()
        tem = dm & base_ix.exam_case_id.isin(te_cases).to_numpy()
        ytr_img = y_case.reindex(base_ix.exam_case_id[trm]).to_numpy()
        ids = base_ix.exam_case_id[tem].to_numpy()
        got = [fit_predict(feats[p][trm], ytr_img, feats[p][tem], multiclass, classes)
               for p in POOLINGS]
        te_order = sorted(te_cases)
        # The probability matrix is the mean over poolings, per case -- carried for
        # AUROC and calibration only, exactly as oof() does it.
        prob = (pd.DataFrame(np.mean([g[1] for g in got], axis=0), index=ids)
                .groupby(level=0).mean().reindex(te_order).to_numpy())
        if multiclass:
            # The shipped multiclass label is a MAJORITY VOTE over poolings (and
            # over a case's images), not the argmax of the averaged probabilities.
            # Those two disagree, and using argmax here would mean the refit LOAO
            # scored a different decision rule than the one being shipped.
            df = pd.DataFrame(np.stack([g[0] for g in got]).T, index=ids)
            votes = {}
            for c, grp in df.groupby(level=0):
                vals, cnt = np.unique(grp.to_numpy().ravel(), return_counts=True)
                votes[c] = float(vals[cnt.argmax()])
            pred = np.array([votes[c] for c in te_order], float)
        else:
            s = (pd.Series(np.mean([g[0] for g in got], axis=0), index=ids)
                 .groupby(level=0).mean().reindex(te_order))
            pred = hard(s.to_numpy(), False)
        yte = y_case.reindex(te_order).to_numpy()
        ytr_case = y_case.reindex(sorted(tr_cases)).to_numpy()
        vals, cnt = np.unique(ytr_case, return_counts=True)
        const_tr = float(vals[cnt.argmax()])
        out[a] = dict(
            n=int(len(yte)),
            training_archive_majority=const_tr,
            baseline_from_training_archives=round(float((yte == const_tr).mean()), 4),
            accuracy=round(float((yte == pred).mean()), 4),
            balanced_accuracy=round(float(np.mean(
                [(pred[yte == c] == c).mean() for c in sorted(set(yte))])), 4),
            pred=pred.tolist(), y=yte.tolist(), cases=te_order,
            prob=np.asarray(prob).tolist(),
        )
        out[a]["delta_vs_training_majority"] = round(
            out[a]["accuracy"] - out[a]["baseline_from_training_archives"], 4)
    return out


def main() -> None:
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    locked = set(man.index[man.development_fold.isna()])
    if len(locked) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(locked))

    a0_ix, a0_feats, a0_meta = load_arm(ANCHOR)
    a1_ix, a1_feats, a1_meta = load_arm(A1_ARM)
    for nm, meta in (("A0", a0_meta), ("A1", a1_meta)):
        if meta.get("locked_cases_seen", 0) != 0:
            raise RuntimeError("%s arm reports locked cases" % nm)
    if not a0_ix.exam_case_id.equals(a1_ix.exam_case_id):
        raise RuntimeError("A0 and A1 image indices differ; the fusion would be "
                           "comparing different image sets")

    extra = load_local(LOCAL_TAG)
    if extra is None:
        raise SystemExit("local feature table %r missing" % LOCAL_TAG)
    if locked & set(extra.index):
        raise RuntimeError("the local feature table contains locked cases")

    y_case, n_classes, target_report = target(dev, FIELD)
    order = sorted(y_case.index)
    yv = y_case.reindex(order).to_numpy()
    classes = sorted(set(float(v) for v in yv))
    multiclass = n_classes > 2
    if n_classes != 3:
        raise RuntimeError("papilla must stay 3-class; got %d" % n_classes)
    fold_map = dev.development_fold.to_dict()
    arch_map = dev.archive.to_dict()
    folds_arr = np.array([fold_map[c] for c in order], float)
    arch_arr = np.array([arch_map[c] for c in order], object)
    unit, static_ok = UNIT[FIELD]

    absent = sorted(set(order) - set(extra.index))
    if absent:
        raise RuntimeError("%d cases absent from the local table: %s"
                           % (len(absent), absent[:5]))

    # The four pre-registered arms. A1+A2 attaches the SAME local columns to the
    # A1 poolings -- the identical attach() the A2 rung used, so the fusion adds
    # no new mechanism, only a different whole-image backbone underneath.
    arms = {
        "A0": (a0_feats, a0_ix, None),
        "A1": (a1_feats, a1_ix, None),
        "A2": (a0_feats, a0_ix, extra),
        "A1+A2": (a1_feats, a1_ix, extra),
    }
    rng = np.random.default_rng(SEED)
    preds = {}
    for k, (feats, ix, ex) in arms.items():
        use = feats if ex is None else attach(feats, ix, ex)
        preds[k] = oof(use, ix, y_case, fold_map, order, multiclass, classes)

    # Score every arm before any verdict is read, so which arms carry LOAO,
    # calibration and abstention cannot depend on what the numbers turned out to be.
    scores, detail = {}, {}
    for k in arms:
        p, prob = preds[k]
        scores[k] = score(yv, p, prob, multiclass, classes, rng)

    # Reproduction check on the three arms that already exist in the record.
    drift = {}
    for k, exp in EXPECTED.items():
        for m, want in exp.items():
            got = float(scores[k]["n"] if m == "n" else scores[k][m])
            if abs(float(want) - got) > 1e-4:
                drift["%s.%s" % (k, m)] = (want, got)
    if drift:
        raise RuntimeError("the fusion's own A0/A1/A2 do not reproduce the recorded "
                           "ladder; the comparison would be against stale numbers: %s"
                           % drift)

    ref_p = preds[KEEP_RULE["reference_arm"]][0]
    a0_p = preds["A0"][0]
    for k in arms:
        p, prob = preds[k]
        sc = scores[k]
        # Two paired contrasts: against the reference arm A1 (the decision) and
        # against A0 (comparability with the published ladder). Both recorded.
        vs_ref = None if k == KEEP_RULE["reference_arm"] else paired(
            yv, p, ref_p, multiclass, rng)
        vs_a0 = None if k == "A0" else paired(yv, p, a0_p, multiclass, rng)
        detail[k] = dict(
            score=sc,
            paired_vs_reference_arm=vs_ref,
            paired_vs_a0=vs_a0,
            verdict_vs_reference_arm=None if vs_ref is None else verdict(vs_ref, sc),
            recall_vs_reference_arm=None if k == KEEP_RULE["reference_arm"] else
            recall_drops(sc.get("per_class_recall", {}),
                         scores[KEEP_RULE["reference_arm"]].get("per_class_recall", {})),
            loao_pooled_oof_partition=loao(yv, p, folds_arr, arch_arr, multiclass,
                                           None if k == "A0" else a0_p),
            loao_refit_archive_held_out=refit_loao(
                arms[k][0], arms[k][1], arms[k][2], y_case, order, arch_arr,
                classes, multiclass),
            calibration=calibration(yv, prob, multiclass, classes),
            abstention=abstention(yv, p, prob, multiclass, classes),
            folds_vs_reference_arm=None if k == KEEP_RULE["reference_arm"] else
            fold_directions(yv, p, ref_p, folds_arr, multiclass),
        )

    # ---- the pre-registered three-part decision, applied mechanically ----
    ref = KEEP_RULE["reference_arm"]
    cand, r = detail["A1+A2"], detail[ref]
    pair = cand["paired_vs_reference_arm"]
    cond_ba = bool(pair["ba_gain"] >= KEEP_RULE["ba_gain_min"]
                   and pair["ba_ci_excludes_zero"])
    cond_recall = not cand["recall_vs_reference_arm"]["any_class_dropped"]
    # Refit LOAO: no archive may lose more than 0.05 of delta relative to the
    # reference arm on the same held-out archive.
    loao_delta = {}
    for a in cand["loao_refit_archive_held_out"]:
        c = cand["loao_refit_archive_held_out"][a]["delta_vs_training_majority"]
        b = r["loao_refit_archive_held_out"][a]["delta_vs_training_majority"]
        loao_delta[a] = round(c - b, 4)
    cond_loao = all(v > -KEEP_RULE["loao_no_direction_may_drop_by"]
                    for v in loao_delta.values())
    keep = bool(cond_ba and cond_recall and cond_loao)
    if keep:
        decision = "kept_as_candidate"
    elif pair["ba_gain"] >= BELOW_RESOLUTION:
        decision = "signal_only_below_threshold"
    else:
        decision = "no_incremental_effect_over_%s" % ref
    # The failure mode the pre-registration names explicitly: better than A0 while
    # not better than A1 is "no incremental effect", never "the fusion works".
    vs_a0 = cand["paired_vs_a0"]
    beats_a0_only = bool(vs_a0["ba_gain"] >= MIN_EFFECT and not cond_ba)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in arms:
        sc, d = scores[k], detail[k]
        pr = d["paired_vs_reference_arm"]
        rows.append(dict(
            field=FIELD, arm=k, n=sc["n"], unit_of_observation=unit,
            baseline_constant=sc["baseline_constant"], accuracy=sc["accuracy"],
            delta=sc["delta"], balanced_accuracy=sc["balanced_accuracy"],
            auroc_macro_ovr=sc.get("auroc_macro_ovr"),
            collapsed_to_one_class=sc["collapsed_to_one_class"],
            recall_0=sc.get("per_class_recall", {}).get("0"),
            recall_1=sc.get("per_class_recall", {}).get("1"),
            recall_2=sc.get("per_class_recall", {}).get("2"),
            ba_gain_vs_a1=None if pr is None else pr["ba_gain"],
            ba_ci_excludes_zero_vs_a1=None if pr is None else pr["ba_ci_excludes_zero"],
            ba_gain_vs_a0=None if d["paired_vs_a0"] is None
            else d["paired_vs_a0"]["ba_gain"],
            ece=d["calibration"].get("expected_calibration_error"),
            verdict_vs_a1=d["verdict_vs_reference_arm"]))
    pd.DataFrame(rows).to_csv(OUT / "papilla_fusion.csv", index=False,
                              encoding="utf-8-sig")

    # Per-case predictions for all four arms, so this never has to be re-run to be
    # audited -- the local ladder persisted metrics only, which is why the A2x
    # freeze had to recompute from scratch.
    pc = pd.DataFrame(dict(exam_case_id=order, archive=arch_arr,
                           development_fold=folds_arr, y_true=yv))
    for k in arms:
        pc[k.replace("+", "_plus_").lower() + "_pred"] = hard(preds[k][0], multiclass)
    pc.to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    (OUT / "papilla_fusion.json").write_text(json.dumps(dict(
        run="papilla fusion: whole-image DA (A1) + automatic local morphology (A2)",
        preregistration=str((EXP / "preregistration_papilla_fusion.md")
                            .relative_to(ROOT)).replace("\\", "/"),
        field=FIELD, n_classes=n_classes, n=int(len(yv)),
        unit_of_observation=unit, observable_from_static_image=static_ok,
        arms=dict(A0=ANCHOR, A1=A1_ARM,
                  A2="%s + local_features/%s" % (ANCHOR, LOCAL_TAG),
                  **{"A1+A2": "%s + local_features/%s" % (A1_ARM, LOCAL_TAG)}),
        keep_rule=KEEP_RULE,
        min_effect=MIN_EFFECT, below_resolution=BELOW_RESOLUTION,
        seed=SEED, n_boot=N_BOOT, poolings=list(POOLINGS),
        decision=dict(
            decision=decision, keep=keep,
            condition_ba=cond_ba, condition_per_class_recall=cond_recall,
            condition_loao_refit=cond_loao,
            loao_refit_delta_vs_reference_arm=loao_delta,
            beats_a0_but_not_a1=beats_a0_only,
            note=("A1+A2 is judged against A1, the best existing arm. Beating A0 "
                  "alone is 'no incremental effect', not a working fusion.")),
        detail=detail,
        target_report=target_report,
        locked_cases_seen=0,
        locked_note=("This run read only development_fold-labelled cases. It says "
                     "nothing about the project's earlier locked-47 consumption."),
        validation_discipline=dict(
            human_boxes_used_on_validation_cases=False,
            local_features_from="detectors trained on training folds only",
            note=("Validation cases contribute no human boxes and no images to the "
                  "detector; their local regions are automatically detected.")),
        limitations=[
            "development case-level OOF; this is not a product capability claim",
            "papilla's absolute BA is near 0.51 before fusion, so even a positive "
            "result would not by itself reach a product threshold",
            "no external same-definition test set exists for papilla",
            "the pre-registration fixed four arms; no pooling, attention or "
            "hyperparameter search was run, so this does not establish that no "
            "fusion could work -- only that the pre-registered one did or did not",
        ],
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    print("papilla fusion, n=%d, 3 classes" % len(yv))
    for k in arms:
        sc, d = scores[k], detail[k]
        pr = d["paired_vs_reference_arm"]
        print("  %-6s acc=%.4f base=%.4f delta=%+.4f BA=%.4f  recall=%s  "
              "BAgain_vs_A1=%s" % (
                  k, sc["accuracy"], sc["baseline_constant"], sc["delta"],
                  sc["balanced_accuracy"], sc.get("per_class_recall"),
                  "ref" if pr is None else "%+.4f %s" % (
                      pr["ba_gain"], pr["ba_ci"])))
    print("  recall drops vs A1:", cand["recall_vs_reference_arm"])
    print("  refit LOAO delta vs A1:", loao_delta)
    print("  conditions: BA=%s recall=%s loao=%s -> %s"
          % (cond_ba, cond_recall, cond_loao, decision))
    print("  wrote", OUT)


if __name__ == "__main__":
    main()
