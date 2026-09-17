"""Why are 8 fields dead? Separate 'model too weak' from 'label carries no signal'.

Two model-side routes have now failed to rescue any of them (full field of view,
then patch-level max/topk/std pooling). Before trying a third, measure the
properties of the LABEL, which no model can fix:

  prevalence      how many positive cases exist at all. With 186 cases, a field at
                  5% has ~9 positives; a 5-fold CV then trains on ~7. No estimator
                  is stable there, and the majority baseline is ~0.95 which almost
                  nothing can beat.
  ceiling         the accuracy a perfect classifier would need to beat the baseline
                  by more than the 0.08 seed-noise band. If prevalence < 0.08 the
                  field is arithmetically unable to show a detectable delta at this
                  sample size, whatever the model.
  n_eff           positives in the smallest fold. If any fold has 0 or 1 positives,
                  per-fold training is degenerate by construction.
  agreement       how often the audit's own prediction equals the majority class.
                  1.000 means the model never left the baseline -- collapse, not
                  a hard problem.

This reads only the frozen OOF tables; it fits nothing and touches no locked case.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OOF = Path("/root/nailfold/artifacts/experiments/threshold_tuned_20260916")
ROLES = Path("/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
NOISE_BAND = 0.08

DEAD = ["capillary_count", "crossing_ratio", "hemorrhage", "malformation_ratio",
        "microthrombus", "overall_assessment", "rbc_aggregation", "wbc_count",
        "sweat_duct", "vasomotion"]
USABLE = ["blood_color", "clarity", "exudation", "subpapillary_venous_plexus"]


def main():
    roles = pd.read_csv(ROLES, usecols=["exam_case_id", "evaluation_role"])
    locked = set(roles.loc[roles.evaluation_role.ne("development"), "exam_case_id"])

    rows = []
    for path in sorted(OOF.glob("*__binary_oof.csv")):
        field = path.name.replace("__binary_oof.csv", "")
        t = pd.read_csv(path).drop_duplicates("case_id")
        bad = set(t.case_id) & locked
        assert not bad, "%s OOF contains locked cases: %s" % (field, sorted(bad))
        y = t["truth"].astype(int)
        pred = t["prediction"].astype(int)
        folds = t["fold"].astype(int)

        n = len(y)
        pos = int(y.sum())
        prev = pos / n
        minority = min(prev, 1 - prev)
        per_fold_pos = [int(y[folds == f].sum()) for f in sorted(folds.unique())]
        maj = int(y.mode().iloc[0])
        # how often the model just repeated the majority class
        collapse = float((pred == maj).mean())
        # a perfect model scores 1.0; baseline is 1-minority; so the largest
        # delta any model could possibly show is the minority share
        headroom = minority
        rows.append({
            "field": field, "n": n, "pos": pos, "prevalence": prev,
            "minority": minority, "min_fold_pos": min(per_fold_pos),
            "per_fold_pos": per_fold_pos,
            "max_possible_delta": headroom,
            "detectable": headroom > NOISE_BAND,
            "pred_equals_majority": collapse,
            "group": "usable" if field in USABLE else "dead",
        })

    df = pd.DataFrame(rows).sort_values(["group", "prevalence"])
    print("%-28s %5s %5s %8s %8s %8s %9s %7s" % (
        "field", "n", "pos", "prev", "minor", "maxdelta", "predmaj", "minfold"))
    for _, r in df.iterrows():
        flag = "" if r["detectable"] else "  <- cannot show >0.08 delta at n=%d" % r["n"]
        print("%-28s %5d %5d %8.3f %8.3f %8.3f %9.3f %7d%s" % (
            r["field"], r["n"], r["pos"], r["prevalence"], r["minority"],
            r["max_possible_delta"], r["pred_equals_majority"],
            r["min_fold_pos"], flag))

    print()
    print("=" * 92)
    und = df[~df["detectable"]]
    print("ARITHMETICALLY UNDETECTABLE (minority share <= %.2f seed-noise band):" % NOISE_BAND)
    if len(und):
        for _, r in und.iterrows():
            print("  %-26s %d/%d positives (%.1f%%). Even a perfect classifier gains "
                  "at most %+.3f." % (r["field"], r["pos"], r["n"],
                                      100 * r["prevalence"], r["max_possible_delta"]))
        print("  -> no model, no feature and no amount of GPU changes this. It needs")
        print("     more positive CASES, or the field must be dropped from the product.")
    else:
        print("  none")

    print()
    coll = df[(df["group"] == "dead") & (df["pred_equals_majority"] > 0.98)]
    print("COLLAPSED (audit model predicted the majority class for >98%% of cases):")
    for _, r in coll.iterrows():
        print("  %-26s pred_equals_majority=%.3f" % (r["field"], r["pred_equals_majority"]))
    if not len(coll):
        print("  none")

    print()
    real = df[(df["group"] == "dead") & df["detectable"] &
              (df["pred_equals_majority"] <= 0.98)]
    print("GENUINELY HARD (enough positives, model did try, still fails):")
    for _, r in real.iterrows():
        print("  %-26s %d/%d positives (%.1f%%), headroom %+.3f, min fold %d" % (
            r["field"], r["pos"], r["n"], 100 * r["prevalence"],
            r["max_possible_delta"], r["min_fold_pos"]))
    print("  -> these are the only dead fields where a better model could still help.")

    out = Path("/root/autodl-tmp/nailfold/artifacts/experiments/spatial_20260917/dead_field_diagnosis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": str(OOF),
        "noise_band": NOISE_BAND,
        "locked_cases_used": 0,
        "limitations": [
            "development set only; prevalence is the development prevalence, not "
            "the intended deployment prevalence",
            "max_possible_delta is an upper bound from class balance alone; it does "
            "not imply any model can reach it",
        ],
        "fields": df.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print()
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
