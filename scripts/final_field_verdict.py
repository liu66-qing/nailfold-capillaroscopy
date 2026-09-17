"""Consolidated field verdict: label-deliverable, signal-only, or impossible.

Combines the two rulers that were measured separately:
  delta = accuracy - per-fold-train majority baseline   -> can it ship as a LABEL
  AUC   = out-of-fold discrimination                    -> is there ANY signal

and the class-balance arithmetic (minority share vs the 0.08 seed-noise band),
which caps what any model could achieve at n=186 regardless of features.

Four verdicts:
  DELIVERABLE   delta CI lower bound > 0 on the prespecified pooling
  SIGNAL_ONLY   AUC CI lower bound > 0.5 but delta CI includes 0. Real ordering
                information that cannot survive being forced into a hard label at
                this prevalence. Usable for ranking/review triage only.
  IMPOSSIBLE_N  minority share <= 0.08, so even a perfect classifier cannot show a
                detectable delta at this sample size. Needs more positive cases.
  NO_SIGNAL     AUC CI includes or sits below 0.5.

The rule-out NPV from eval_signal_auc is deliberately NOT carried forward: it was
achieved at specificity ~0.02, i.e. by flagging ~98% of cases, which is not a
screener. That is recorded as a rejected option, not a result.
"""
import json
from pathlib import Path

import pandas as pd

DIR = Path("/root/autodl-tmp/nailfold/artifacts/experiments/spatial_20260917")
NOISE_BAND = 0.08
PRESPEC = "topk_mean"
MIN_USEFUL_SPEC = 0.20  # below this a "rule-out" flags nearly everyone


def load_delta():
    """Prespecified-pooling delta, best over presets, from the spatial eval."""
    out = {}
    for name in ["summary.json", "summary_hi.json"]:
        rep = json.loads((DIR / name).read_text(encoding="utf-8"))
        for field, e in rep["results"].items():
            if "skipped" in e:
                continue
            keys = [k for k in e if k.endswith(":" + PRESPEC)]
            if not keys:
                continue
            best = max(keys, key=lambda k: e[k]["delta_ci95"][0])
            cur = out.get(field)
            if cur is None or e[best]["delta_ci95"][0] > cur["delta_ci95"][0]:
                out[field] = dict(e[best], view=best,
                                  audit_delta=e["reference_audit"]["delta"])
    return out


def load_auc():
    out = {}
    for name in ["auc_native.json", "auc_native_hi.json"]:
        rep = json.loads((DIR / name).read_text(encoding="utf-8"))
        for field, e in rep["results"].items():
            if "skipped" in e or not e.get("views"):
                continue
            v = e["views"].get(PRESPEC)
            if v is None:
                continue
            cur = out.get(field)
            if cur is None or v["auc_ci95"][0] > cur["auc_ci95"][0]:
                out[field] = dict(v, positives=e["positives"], n=e["n"],
                                  prevalence=e["prevalence"], preset=name)
    return out


def main():
    delta = load_delta()
    auc = load_auc()
    diag = {r["field"]: r for r in
            json.loads((DIR / "dead_field_diagnosis.json").read_text(encoding="utf-8"))["fields"]}

    rows = []
    for field in sorted(set(delta) & set(auc)):
        d, a = delta[field], auc[field]
        minority = float(diag[field]["minority"])
        auc_sig = a["auc_ci95"][0] > 0.5
        delta_sig = d["delta_ci95"][0] > 0
        if delta_sig:
            verdict = "DELIVERABLE"
        elif minority <= NOISE_BAND:
            verdict = "IMPOSSIBLE_N"
        elif auc_sig:
            verdict = "SIGNAL_ONLY"
        else:
            verdict = "NO_SIGNAL"
        ro = a.get("rule_out") or {}
        rows.append({
            "field": field, "verdict": verdict,
            "pos": a["positives"], "n": a["n"], "prev": a["prevalence"],
            "minority": minority,
            "audit_delta": d["audit_delta"], "delta": d["delta"],
            "delta_lo": d["delta_ci95"][0], "delta_hi": d["delta_ci95"][1],
            "auc": a["auc"], "auc_lo": a["auc_ci95"][0], "auc_hi": a["auc_ci95"][1],
            "ruleout_spec": ro.get("specificity", float("nan")),
            "view": d["view"],
        })

    df = pd.DataFrame(rows)
    order = {"DELIVERABLE": 0, "SIGNAL_ONLY": 1, "NO_SIGNAL": 2, "IMPOSSIBLE_N": 3}
    df = df.sort_values(["verdict", "auc"], key=lambda s: s.map(order) if s.name == "verdict" else s,
                        ascending=[True, False])

    print("%-28s %-13s %6s %7s %8s %8s %7s %16s" % (
        "field", "verdict", "pos/n", "prev", "delta", "delta_lo", "auc", "auc_ci95"))
    for _, r in df.iterrows():
        print("%-28s %-13s %6s %7.3f %+8.3f %+8.3f %7.3f  [%.3f,%.3f]" % (
            r["field"], r["verdict"], "%d/%d" % (r["pos"], r["n"]), r["prev"],
            r["delta"], r["delta_lo"], r["auc"], r["auc_lo"], r["auc_hi"]))

    print()
    print("=" * 100)
    for v in ["DELIVERABLE", "SIGNAL_ONLY", "NO_SIGNAL", "IMPOSSIBLE_N"]:
        sub = df[df.verdict == v]
        print("%s (%d): %s" % (v, len(sub), ", ".join(sub.field) or "none"))

    print()
    print("CHANGE vs audit, deliverable fields only:")
    for _, r in df[df.verdict == "DELIVERABLE"].iterrows():
        print("  %-26s %+.3f -> %+.3f  (%+.3f)  CI[%+.3f,%+.3f]  %s" % (
            r["field"], r["audit_delta"], r["delta"], r["delta"] - r["audit_delta"],
            r["delta_lo"], r["delta_hi"], r["view"]))

    print()
    print("REJECTED OPTION: rule-out thresholds at 90%% sensitivity")
    bad = df[df.ruleout_spec < MIN_USEFUL_SPEC]
    print("  %d/%d fields reach 90%% sensitivity only at specificity < %.2f," % (
        len(bad), len(df), MIN_USEFUL_SPEC))
    print("  i.e. by flagging nearly every case. The NPV of ~1.0 there is an artifact")
    print("  of flagging everyone, not screening ability. Not carried forward.")

    out = DIR / "final_field_verdict.json"
    out.write_text(json.dumps({
        "noise_band": NOISE_BAND,
        "prespecified_pooling": PRESPEC,
        "locked_cases_used": 0,
        "verdict_counts": df.verdict.value_counts().to_dict(),
        "limitations": [
            "development set only, out-of-fold; NOT product capability",
            "delta and AUC are each best-over-2-presets on the prespecified pooling, "
            "a mild optimistic bias",
            "SIGNAL_ONLY means ranking support only; it does not license a label",
            "IMPOSSIBLE_N is a statement about n=186 and class balance, not about "
            "whether the finding is real in clinic",
        ],
        "fields": df.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print()
    print("WROTE %s" % out)


if __name__ == "__main__":
    main()
