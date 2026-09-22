"""Apply the addendum section H gates to the round-one probe results.

Reads only the artefacts written by eval_medical_encoders.py. Computes nothing
new about the encoders; it re-reads the case-level out-of-fold predictions so
the two gate clauses that the summary table cannot express are checked:

  - "at least 3/5 outer folds move the same way" needs per-fold BA gain,
  - "no minority-class recall collapse" needs per-class recall.

The gates are quoted from the addendum and evaluated literally. A clause that
cannot be evaluated is reported as unevaluable, not as passed.

  PYTHONIOENCODING=utf-8 python scripts/check_medical_encoder_gates.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
ANCHOR = "anchor_dinov2b_deployed"
PRIMARY = "malformation_ratio"
RETAINED = ["clarity", "exudation", "subpapillary_venous_plexus", "blood_color"]

# verbatim from addendum section H
RD_GATE = dict(min_paired_ba_gain=0.03, min_folds_same_direction=3, n_folds=5,
               no_minority_recall_collapse=True)
ADOPT_GATE = dict(min_paired_ba_gain=0.05, ci_lo_above_zero=True,
                  loao_mean_gain_above_zero=True, max_archive_drop=0.03,
                  max_retained_field_drop=0.02)


MULTICLASS_FIELDS = {"papilla"}


def harden(d: pd.DataFrame, field: str) -> pd.DataFrame:
    """predictions_oof.csv stores probabilities for binary fields and class
    indices for multiclass ones, so binary predictions need the protocol's
    0.5 threshold before any classification metric is taken."""
    d = d.copy()
    if field not in MULTICLASS_FIELDS:
        d["pred"] = (d["pred"] >= 0.5).astype(float).where(d["pred"].notna())
    return d


def per_fold_gain(oof: pd.DataFrame, field: str, arm: str) -> dict:
    """BA per outer fold for arm and anchor, on the cases both predicted."""
    oof = harden(oof, field)
    a = oof[(oof.field == field) & (oof.arm == ANCHOR)].set_index("exam_case_id")
    b = oof[(oof.field == field) & (oof.arm == arm)].set_index("exam_case_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    keep = a.pred.notna() & b.pred.notna() & a.y_true.notna()
    a, b = a[keep], b[keep]
    out = {}
    for f in sorted(a.fold.dropna().unique()):
        m = a.fold == f
        if a.loc[m, "y_true"].nunique() < 2:
            out[str(int(f))] = None            # BA undefined in this fold
            continue
        out[str(int(f))] = round(
            balanced_accuracy_score(b.loc[m, "y_true"], b.loc[m, "pred"])
            - balanced_accuracy_score(a.loc[m, "y_true"], a.loc[m, "pred"]), 4)
    return out


def recalls(oof: pd.DataFrame, field: str, arm: str) -> dict:
    oof = harden(oof, field)
    d = oof[(oof.field == field) & (oof.arm == arm)]
    d = d[d.pred.notna() & d.y_true.notna()]
    labs = sorted(d.y_true.unique())
    r = recall_score(d.y_true, d.pred, labels=labs, average=None, zero_division=0)
    n = d.y_true.value_counts()
    minority = min(labs, key=lambda k: n[k])
    return dict(per_class={str(int(k)): round(float(v), 4) for k, v in zip(labs, r)},
                minority_class=str(int(minority)), minority_n=int(n[minority]),
                minority_recall=round(float(r[labs.index(minority)]), 4))


def main() -> None:
    oof = pd.read_csv(EXP / "predictions_oof.csv", dtype={"exam_case_id": str})
    met = json.loads((EXP / "paired_metrics.json").read_text(encoding="utf-8"))
    summ = pd.read_csv(EXP / "paired_summary.csv")
    loao = met["loao"]

    arms = [a for a in sorted(oof.arm.unique()) if a != ANCHOR]
    verdicts = {}
    for arm in arms:
        row = summ[(summ.field == PRIMARY) & (summ.arm == arm)].iloc[0]
        gain, ci_lo = float(row.ba_gain), float(row.ci_lo)
        folds = per_fold_gain(oof, PRIMARY, arm)
        pos = sum(1 for v in folds.values() if v is not None and v > 0)
        rec_arm = recalls(oof, PRIMARY, arm)
        rec_anc = recalls(oof, PRIMARY, ANCHOR)
        collapse = (rec_arm["minority_recall"] < 0.5 * rec_anc["minority_recall"]
                    or rec_arm["minority_recall"] == 0.0)

        la = loao[PRIMARY][ANCHOR]
        lb = loao[PRIMARY][arm]
        per_arch = {k: round(lb[k]["balanced_accuracy"]
                             - la[k]["balanced_accuracy"], 4) for k in la}
        loao_mean = round(float(np.mean(list(per_arch.values()))), 4)
        worst_arch = min(per_arch.values())

        ret = {}
        for f in RETAINED:
            r = summ[(summ.field == f) & (summ.arm == arm)]
            ret[f] = round(float(r.iloc[0].ba_gain), 4) if len(r) else None
        worst_ret = min(v for v in ret.values() if v is not None)

        rd = dict(
            ba_gain_ge_0_03=bool(gain >= RD_GATE["min_paired_ba_gain"]),
            folds_same_direction=f"{pos}/{RD_GATE['n_folds']}",
            folds_gate_met=bool(pos >= RD_GATE["min_folds_same_direction"]),
            minority_recall_collapse=bool(collapse))
        rd["passed"] = bool(rd["ba_gain_ge_0_03"] and rd["folds_gate_met"]
                            and not rd["minority_recall_collapse"])
        ad = dict(
            ba_gain_ge_0_05=bool(gain >= ADOPT_GATE["min_paired_ba_gain"]),
            ci_lo_above_zero=bool(ci_lo > 0),
            loao_mean_gain=loao_mean,
            loao_mean_above_zero=bool(loao_mean > 0),
            worst_archive_gain=worst_arch,
            no_archive_drop_over_0_03=bool(worst_arch >= -ADOPT_GATE["max_archive_drop"]),
            worst_retained_field_gain=worst_ret,
            retained_fields_within_0_02=bool(
                worst_ret >= -ADOPT_GATE["max_retained_field_drop"]))
        ad["passed"] = bool(all(ad[k] for k in [
            "ba_gain_ge_0_05", "ci_lo_above_zero", "loao_mean_above_zero",
            "no_archive_drop_over_0_03", "retained_fields_within_0_02"]))

        status = ("adopted_by_numbers" if ad["passed"]
                  else "rd_gate_passed" if rd["passed"] else "inconclusive")
        verdicts[arm] = dict(
            role=met["arms"].get(arm, {}).get("role") if isinstance(
                met.get("arms"), dict) else None,
            primary_ba_gain=gain, ci=[ci_lo, float(row.ci_hi)],
            per_fold_ba_gain=folds,
            recall_anchor=rec_anc, recall_arm=rec_arm,
            per_archive_ba_gain=per_arch, retained_field_ba_gain=ret,
            rd_gate=rd, adoption_gate=ad, numeric_status=status)

    out = dict(
        source=dict(summary="paired_summary.csv", metrics="paired_metrics.json",
                    oof="predictions_oof.csv"),
        gates_quoted_from="rescue_plan_medical_encoder_addendum_20260921.md section H",
        rd_gate=RD_GATE, adoption_gate=ADOPT_GATE,
        primary_field=PRIMARY, anchor=ANCHOR, verdicts=verdicts,
        caveats=[
            "development set only; locked-47 was not read by this run",
            "numeric_status is the arithmetic result of the gates, not a launch "
            "decision; the addendum calls these project investment gates",
            "the medical arm runs at 224x224 with patch 16 while the anchor runs "
            "at 518x686 with patch 14, so medical pretraining is confounded with "
            "input resolution and cannot be isolated by this round",
            "malformation_ratio truth is the doctor's banded report value, not a "
            "counted per-vessel ratio",
        ])
    (EXP / "gate_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["verdicts"], ensure_ascii=False, indent=2))
    print("\nwrote %s" % (EXP / "gate_check.json"))


if __name__ == "__main__":
    main()
