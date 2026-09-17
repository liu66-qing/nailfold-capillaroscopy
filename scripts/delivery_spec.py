"""Per-field delivery spec: what ships for each field, and on what evidence.

Decision rule, applied in order:
  1. minority share <= 0.08          -> CONSTANT (fixed majority answer, by design)
  2. delta_vs_constant CI lower > 0  -> MODEL (beats shipping a fixed answer)
  3. otherwise                       -> CONSTANT (a model here is not justified:
                                        it cannot be shown to beat doing nothing)

Case 3 is the important one. Earlier reports called these fields "signal only" and
left them in limbo. Under the user's instruction -- extreme-skew fields answer the
majority class -- the honest resolution is that any field whose model cannot beat a
fixed answer should ALSO ship as a fixed answer. That is a smaller product, but
every field in it is defensible.

The bar is delta_vs_constant, not delta_vs_per_fold_baseline. The per-fold baseline
takes the training majority separately in each fold, which on a near-balanced field
can flip class and score below chance, flattering the model by several points
(clarity: +0.324 vs +0.270). A deployed no-model system ships ONE fixed answer, so
that is the bar.

Preset choice per field would itself be a selection, so the preset is fixed to
`native` for the headline and native_hi is reported only as a sensitivity check.
"""
import json
from pathlib import Path

import pandas as pd

DIR = Path("/root/autodl-tmp/nailfold/artifacts/experiments/perfield_20260917")
HEADLINE = "native.json"
SENSITIVITY = "native_hi.json"


def load(name):
    return json.loads((DIR / name).read_text(encoding="utf-8"))


def main():
    head = load(HEADLINE)
    sens = load(SENSITIVITY)

    rows = []
    for field, e in sorted(head["results"].items()):
        if "skipped" in e:
            continue
        alt = sens["results"].get(field, {})
        if e.get("delivery") == "constant_majority":
            rows.append({
                "field": field, "ship": "CONSTANT", "reason": "minority<=0.08",
                "answer": e["constant_answer"], "acc": e["accuracy"],
                "const_base": e["baseline_constant"],
                "gain": 0.0, "gain_lo": 0.0,
                "alt_gain": 0.0, "model": "-", "sens": None, "spec": None,
                "prev": e["prevalence"],
            })
            continue
        lo = e["delta_vs_constant_ci95"][0]
        beats = lo > 0
        picks = {}
        for c in e.get("per_fold_choices", []):
            k = "+".join(c["features"]) + " | " + c["model"]
            picks[k] = picks.get(k, 0) + 1
        model = max(picks.items(), key=lambda kv: kv[1])[0] if picks else "-"
        rows.append({
            "field": field,
            "ship": "MODEL" if beats else "CONSTANT",
            "reason": "beats fixed answer" if beats else "cannot beat fixed answer",
            "answer": None if beats else int(e["prevalence"] >= 0.5),
            "acc": e["accuracy"] if beats else e["baseline_constant"],
            "const_base": e["baseline_constant"],
            "gain": e["delta_vs_constant"], "gain_lo": lo,
            "alt_gain": alt.get("delta_vs_constant", float("nan")),
            "model": model if beats else "-",
            "sens": e.get("sensitivity"), "spec": e.get("specificity"),
            "prev": e["prevalence"],
        })

    df = pd.DataFrame(rows)
    df["_o"] = df.ship.map({"MODEL": 0, "CONSTANT": 1})
    df = df.sort_values(["_o", "gain_lo"], ascending=[True, False]).drop(columns="_o")

    print("SHIP DECISION  (bar = beat a single fixed majority answer)")
    print("%-28s %-9s %7s %7s %8s %8s %8s  %s" % (
        "field", "ship", "prev", "acc", "vs_const", "ci_lo", "hi_check", "model"))
    for _, r in df.iterrows():
        print("%-28s %-9s %7.3f %7.3f %+8.3f %+8.3f %+8.3f  %s" % (
            r["field"], r["ship"], r["prev"], r["acc"], r["gain"], r["gain_lo"],
            r["alt_gain"] if r["alt_gain"] == r["alt_gain"] else 0.0, r["model"]))

    ship_model = df[df.ship == "MODEL"]
    print()
    print("=" * 104)
    print("SHIPS A MODEL (%d): %s" % (len(ship_model), ", ".join(ship_model.field)))
    for _, r in ship_model.iterrows():
        print("  %-26s acc %.3f vs fixed %.3f  gain %+.3f CI_lo %+.3f  "
              "sens %.3f spec %.3f" % (
                  r["field"], r["acc"], r["const_base"], r["gain"], r["gain_lo"],
                  r["sens"], r["spec"]))

    ship_const = df[df.ship == "CONSTANT"]
    print()
    print("SHIPS A FIXED ANSWER (%d):" % len(ship_const))
    for _, r in ship_const.iterrows():
        why = ("prevalence %.3f, minority %.3f below noise band"
               % (r["prev"], min(r["prev"], 1 - r["prev"]))
               if r["reason"] == "minority<=0.08" else
               "model gain %+.3f, CI lower %+.3f includes 0"
               % (r["gain"], r["gain_lo"]))
        print("  %-26s answer=%d  acc %.3f  (%s)" % (
            r["field"], r["answer"], r["acc"], why))

    # Sensitivity: does the preset choice flip any decision?
    flips = []
    for _, r in df.iterrows():
        if r["ship"] == "CONSTANT" and r["reason"] == "minority<=0.08":
            continue
        alt = sens["results"].get(r["field"], {})
        alo = alt.get("delta_vs_constant_ci95", [float("nan")])[0]
        if alo == alo and ((alo > 0) != (r["gain_lo"] > 0)):
            flips.append((r["field"], r["gain_lo"], alo))
    print()
    print("PRESET SENSITIVITY (would native_hi change the ship decision?)")
    if flips:
        for f, a, b in flips:
            print("  %-26s native CI_lo %+.3f vs native_hi %+.3f  <- DECISION FLIPS"
                  % (f, a, b))
        print("  These fields are not robust to an arbitrary preset choice and their")
        print("  decision should be treated as undetermined, not as a result.")
    else:
        print("  no field's decision flips between presets")

    out = DIR / "delivery_spec.json"
    out.write_text(json.dumps({
        "bar": "delta vs single fixed majority answer, CI lower bound > 0",
        "headline_preset": HEADLINE.replace(".json", ""),
        "constant_threshold": head["constant_threshold"],
        "locked_cases_used": 0,
        "counts": df.ship.value_counts().to_dict(),
        "limitations": [
            "development set out-of-fold; NOT product capability; external validation is 0",
            "CONSTANT fields have no measured skill by construction; they are a "
            "delivery choice and must never be quoted as model performance",
            "per-field model/feature/threshold were selected inside the fold loop, so "
            "these deltas are not inflated by that search; the inner fold is ~37 "
            "cases so the choice itself is noisy",
            "headline preset fixed to native to avoid a second selection; native_hi "
            "shown only as a sensitivity check",
        ],
        "fields": df.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print()
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
