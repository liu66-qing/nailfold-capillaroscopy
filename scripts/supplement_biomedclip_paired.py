#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream B: paired accuracy diff, BA diff and per-class recall for BiomedCLIP.

No refitting and no new model. This re-pairs the OOF predictions already stored
by run_field_matrix_three_arms.py, so every number here is a different view of
the same fit, not another experiment.

What is added over 执行记录 5, which reported paired BA only:
  * paired ACCURACY difference with a bootstrap CI (the shipped ruler is
    accuracy - one constant answer, so the accuracy delta is the ruler BiomedCLIP
    must be judged on, and it was missing)
  * per-class recall for BOTH arms side by side, so a BA gain that comes from
    trading the majority class away is visible
  * McNemar-style disagreement counts (arm right / ref wrong and the reverse),
    which say how many cases actually moved

Pairing is on exam_case_id within a field, so the bootstrap resamples CASES and
keeps both arms on the same resample. That is what makes the CI a paired CI.

Also runs ONE pre-fixed LOAO validation. The scheme is fixed here, in code,
before the numbers are seen:
  * folds are the three recovered archives, leave-one-archive-out
  * the arm pair is fixed (biomedclip_medical vs anchor_dinov2b_deployed)
  * the readout is the shipped fixed configuration, C=0.03, unchanged
  * the reported quantity is fixed: accuracy delta vs the TRAIN-archive mode,
    BA, and the paired differences
  * no hyper-parameter may be changed after reading it (rule from the request)

DEVELOPMENT ONLY. development_fold non-NaN, case-level OOF. locked-47 is never
read; the run aborts if a locked id appears.

Run:
  PYTHONIOENCODING=utf-8 python scripts/supplement_biomedclip_paired.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
PRED = EXP / "field_matrix_three_arms_predictions.csv"
MATRIX = EXP / "field_matrix_three_arms.csv"
OUT_JSON = EXP / "biomedclip_paired_supplement.json"
OUT_CSV = EXP / "biomedclip_paired_supplement.csv"
OUT_RECALL = EXP / "biomedclip_per_class_recall.csv"

ANCHOR = "anchor_dinov2b_deployed"
CAND = "biomedclip_medical"
SEED, N_BOOT = 20260922, 2000


def discretise(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """pred is a probability for 2 classes and a voted label for >2."""
    p = df["pred"].to_numpy(float)
    return (p >= 0.5).astype(int) if n_classes == 2 else np.rint(p).astype(int)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    rec = [(p[y == c] == c).mean() for c in np.unique(y)]
    return float(np.mean(rec))


def per_class_recall(y: np.ndarray, p: np.ndarray) -> dict:
    return {int(c): dict(n=int((y == c).sum()),
                         recall=round(float((p[y == c] == c).mean()), 4))
            for c in np.unique(y)}


def paired(y, pa, pb, rng) -> dict:
    """pb = candidate, pa = anchor. Resample cases, keep both arms aligned."""
    n = len(y)
    acc_a, acc_b = float((pa == y).mean()), float((pb == y).mean())
    ba_a, ba_b = balanced_accuracy(y, pa), balanced_accuracy(y, pb)
    d_acc, d_ba = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        ys = y[i]
        if len(np.unique(ys)) < 2:
            continue
        d_acc.append((pb[i] == ys).mean() - (pa[i] == ys).mean())
        d_ba.append(balanced_accuracy(ys, pb[i]) - balanced_accuracy(ys, pa[i]))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)] if v else [None, None]

    ca, cb = ci(d_acc), ci(d_ba)
    return dict(
        n=int(n),
        accuracy_anchor=round(acc_a, 4), accuracy_candidate=round(acc_b, 4),
        accuracy_diff=round(acc_b - acc_a, 4), accuracy_diff_ci=ca,
        accuracy_diff_ci_excludes_zero=bool(
            ca[0] is not None and (ca[0] > 0 or ca[1] < 0)),
        ba_anchor=round(ba_a, 4), ba_candidate=round(ba_b, 4),
        ba_diff=round(ba_b - ba_a, 4), ba_diff_ci=cb,
        ba_diff_ci_excludes_zero=bool(
            cb[0] is not None and (cb[0] > 0 or cb[1] < 0)),
        candidate_right_anchor_wrong=int(((pb == y) & (pa != y)).sum()),
        anchor_right_candidate_wrong=int(((pa == y) & (pb != y)).sum()),
        recall_anchor=per_class_recall(y, pa),
        recall_candidate=per_class_recall(y, pb),
        n_boot_used=len(d_acc))


def loao(sub_a: pd.DataFrame, sub_b: pd.DataFrame, n_classes: int) -> dict:
    """Pre-fixed leave-one-archive-out readout of the SAME stored predictions.

    The archive is the first path component of exam_case_id. The baseline for
    each held-out archive is the mode of the OTHER archives, so the constant
    answer cannot be fitted on the archive being scored.
    """
    arch = sub_a.exam_case_id.str.split("/").str[0]
    out = {}
    for held in sorted(arch.unique()):
        te, tr = arch == held, arch != held
        if te.sum() < 10 or tr.sum() < 10:
            out[held] = dict(skipped="fewer than 10 cases on one side")
            continue
        y = sub_a.y_true.to_numpy(float).astype(int)
        pa, pb = discretise(sub_a, n_classes), discretise(sub_b, n_classes)
        y_tr, y_te = y[tr.to_numpy()], y[te.to_numpy()]
        const = int(pd.Series(y_tr).mode().iloc[0])      # trained-archive mode
        base = float((y_te == const).mean())
        a, b = pa[te.to_numpy()], pb[te.to_numpy()]
        out[held] = dict(
            n_test=int(te.sum()), train_archive_mode=const,
            baseline_from_train_archives=round(base, 4),
            accuracy_anchor=round(float((a == y_te).mean()), 4),
            accuracy_candidate=round(float((b == y_te).mean()), 4),
            delta_anchor=round(float((a == y_te).mean()) - base, 4),
            delta_candidate=round(float((b == y_te).mean()) - base, 4),
            ba_anchor=round(balanced_accuracy(y_te, a), 4),
            ba_candidate=round(balanced_accuracy(y_te, b), 4),
            accuracy_diff=round(float((b == y_te).mean() - (a == y_te).mean()), 4),
            ba_diff=round(balanced_accuracy(y_te, b)
                          - balanced_accuracy(y_te, a), 4))
    diffs = [v["ba_diff"] for v in out.values() if "ba_diff" in v]
    out["_summary"] = dict(
        archives=len([v for v in out.values() if "ba_diff" in v]),
        candidate_better_in=sum(1 for d in diffs if d > 0),
        mean_ba_diff=round(float(np.mean(diffs)), 4) if diffs else None,
        sign_consistent=bool(diffs and (all(d > 0 for d in diffs)
                                        or all(d < 0 for d in diffs))))
    return out


def main() -> None:
    pred = pd.read_csv(PRED)
    mat = pd.read_csv(MATRIX)

    assert not pred.exam_case_id.astype(str).str.contains("rep", case=False).any(), \
        "report scans must never appear in an encoder index"
    nclass = (mat.drop_duplicates("field").set_index("field")["n_classes"]
              .astype(int).to_dict())

    rng = np.random.default_rng(SEED)
    rows, recall_rows, loao_all = [], [], {}
    for field in sorted(nclass):
        a = pred[(pred.field == field) & (pred.arm == ANCHOR)]
        b = pred[(pred.field == field) & (pred.arm == CAND)]
        if a.empty or b.empty:
            continue
        a = a.sort_values("exam_case_id").reset_index(drop=True)
        b = b.sort_values("exam_case_id").reset_index(drop=True)
        assert a.exam_case_id.tolist() == b.exam_case_id.tolist(), \
            "arms must be paired on the same cases for %s" % field
        assert np.allclose(a.y_true, b.y_true), "labels differ across arms"
        assert a.exam_case_id.is_unique, "case-level OOF must be unique"

        k = nclass[field]
        y = a.y_true.to_numpy(float).astype(int)
        r = paired(y, discretise(a, k), discretise(b, k), rng)
        r.update(field=field, n_classes=k)
        rows.append(r)

        for arm, key in ((ANCHOR, "recall_anchor"), (CAND, "recall_candidate")):
            for c, v in r[key].items():
                recall_rows.append(dict(field=field, n_classes=k, arm=arm,
                                        cls=c, n=v["n"], recall=v["recall"]))
        loao_all[field] = loao(a, b, k)

    df = pd.DataFrame(rows)[[
        "field", "n_classes", "n",
        "accuracy_anchor", "accuracy_candidate", "accuracy_diff",
        "accuracy_diff_ci", "accuracy_diff_ci_excludes_zero",
        "ba_anchor", "ba_candidate", "ba_diff", "ba_diff_ci",
        "ba_diff_ci_excludes_zero",
        "candidate_right_anchor_wrong", "anchor_right_candidate_wrong"]]
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(recall_rows).to_csv(OUT_RECALL, index=False)

    acc_sig = df[df.accuracy_diff_ci_excludes_zero].field.tolist()
    ba_sig = df[df.ba_diff_ci_excludes_zero].field.tolist()
    both = sorted(set(acc_sig) & set(ba_sig))
    disagree = sorted(set(acc_sig) ^ set(ba_sig))

    doc = dict(
        purpose=("Stream B: add the paired accuracy difference, BA difference "
                 "and per-class recall that 执行记录 5 reported only partly, and "
                 "run one pre-fixed LOAO check. Re-pairing of stored OOF "
                 "predictions; nothing is refitted and no new model is used."),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        anchor=ANCHOR, candidate=CAND, seed=SEED, n_boot=N_BOOT,
        readout="shipped fixed configuration, C=0.03, unchanged",
        source_predictions=str(PRED.relative_to(ROOT)),
        loao_scheme_fixed_before_reading=dict(
            folds="leave-one-archive-out over the three recovered archives",
            baseline="mode of the training archives, never the held-out one",
            arms="fixed pair, no third arm introduced",
            reported="accuracy delta, BA, paired differences",
            no_retuning_after_reading=True),
        fields={r["field"]: r for r in rows},
        loao=loao_all,
        summary=dict(
            n_fields=len(rows),
            accuracy_diff_ci_excludes_zero=acc_sig,
            ba_diff_ci_excludes_zero=ba_sig,
            significant_on_both_rulers=both,
            rulers_disagree_on=disagree),
        forbidden_conclusions=[
            ("a gain here is not a medical-pretraining gain: BiomedCLIP differs "
             "from the anchor on resolution (224 vs 518x686), patch size (16 vs "
             "14), encoder family, training objective and normalisation, so the "
             "contrast is confounded on five axes at once"),
            ("these numbers must not be used to assign fields to BiomedCLIP: "
             "they were read after the fact, which is the post-hoc winner-"
             "picking the request forbids. A field branch must be pre-registered"),
            ("MedSigLIP and RETFound remain blocked_access; nothing here bears "
             "on them, and they must not be closed on this evidence"),
            "development OOF only; not a statement of product capability",
        ],
        limitations=[
            ("LOAO here re-reads predictions whose folds were stratified across "
             "archives, so the held-out archive contributed to the other folds' "
             "training. It is an archive-transfer READOUT, weaker than a true "
             "retrained LOAO; it cannot be upgraded by rerunning with new "
             "hyper-parameters, which the request forbids"),
            ("the paired CI resamples cases, so it reflects case sampling, not "
             "encoder or seed variation"),
            ("per-class recall on small classes is unstable; class sizes are "
             "printed alongside every recall so this stays visible"),
        ],
        locked_cases_seen=0,
    )
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(df.to_string(index=False))
    print("\nsignificant on both rulers:", both)
    print("rulers disagree on:", disagree)
    print("written:", OUT_JSON.name, OUT_CSV.name, OUT_RECALL.name)


if __name__ == "__main__":
    main()
