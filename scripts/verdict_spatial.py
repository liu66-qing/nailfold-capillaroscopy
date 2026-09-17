"""Final verdict on the spatial-pooling route: did any dead field become usable?

Two columns matter and they must not be conflated:

  prespec  the pooling chosen BEFORE seeing results (topk_mean), the only number
           that may be quoted as a result
  best     the maximum over ~55 (preset x pooling) views, reported for
           transparency but optimistically biased by that selection

A field is USABLE only if its CI lower bound > 0. A dead field counts as RESCUED
only if it was <= 0 in the audit and its PRESPECIFIED CI lower bound is now > 0.
Anything that only crosses on `best` is a multiple-comparison artifact and is
labelled as such.

Degenerate cases: a delta of exactly 0.000 with CI [0,0] means the classifier
predicted the majority class for every case. That is not capability, it is
collapse, and it is flagged rather than counted.
"""
import json
from pathlib import Path

DIR = Path("/root/autodl-tmp/nailfold/artifacts/experiments/spatial_20260917")
PRESPEC = "topk_mean"


def load():
    merged = {}
    for name in ["summary.json", "summary_hi.json"]:
        rep = json.loads((DIR / name).read_text(encoding="utf-8"))
        for field, entry in rep["results"].items():
            if "skipped" in entry:
                continue
            tgt = merged.setdefault(field, {})
            for k, v in entry.items():
                if k == "reference_audit":
                    tgt["reference_audit"] = v
                else:
                    tgt[k] = v
    return merged


def main():
    res = load()
    print("%-28s %8s %8s %8s  %-26s %8s %8s" % (
        "field", "audit", "prespec", "pre_lo", "best_view", "best", "best_lo"))

    rescued, usable, dead, degenerate = [], [], [], []
    for field in sorted(res):
        e = res[field]
        audit = e["reference_audit"]["delta"]
        views = {k: v for k, v in e.items() if k != "reference_audit"}
        if not views:
            continue
        pre_keys = [k for k in views if k.endswith(":" + PRESPEC)]
        if not pre_keys:
            continue
        pk = max(pre_keys, key=lambda k: views[k]["delta_ci95"][0])
        pre = views[pk]
        bk = max(views, key=lambda k: views[k]["delta_ci95"][0])
        best = views[bk]

        print("%-28s %+8.3f %+8.3f %+8.3f  %-26s %+8.3f %+8.3f" % (
            field, audit, pre["delta"], pre["delta_ci95"][0],
            bk, best["delta"], best["delta_ci95"][0]))

        collapsed = (abs(best["delta"]) < 1e-9 and
                     abs(best["delta_ci95"][0]) < 1e-9 and
                     abs(best["delta_ci95"][1]) < 1e-9)
        if collapsed:
            degenerate.append(field)
        elif pre["delta_ci95"][0] > 0:
            (rescued if audit <= 0 else usable).append((field, audit, pk, pre))
        elif best["delta_ci95"][0] > 0:
            dead.append((field, audit, "crosses only on post-hoc best (%s)" % bk))
        else:
            dead.append((field, audit, "CI still includes or is below 0"))

    print()
    print("=" * 96)
    print("RESCUED  (audit <= 0, prespecified CI lower bound now > 0)")
    for f, a, k, v in rescued:
        print("  %-26s %+.3f -> %+.3f  CI[%+.3f,%+.3f]  %s" % (
            f, a, v["delta"], v["delta_ci95"][0], v["delta_ci95"][1], k))
    if not rescued:
        print("  none")

    print()
    print("USABLE, already was  (prespecified CI lower bound > 0)")
    for f, a, k, v in usable:
        print("  %-26s %+.3f -> %+.3f  CI[%+.3f,%+.3f]  sens %.3f spec %.3f  %s" % (
            f, a, v["delta"], v["delta_ci95"][0], v["delta_ci95"][1],
            v["sensitivity"], v["specificity"], k))

    print()
    print("STILL NOT USABLE")
    for f, a, why in dead:
        print("  %-26s audit %+.3f  %s" % (f, a, why))

    print()
    print("DEGENERATE (predicts majority class for every case; delta and CI all 0)")
    print("  " + (", ".join(degenerate) if degenerate else "none"))

    print()
    print("COUNT  usable=%d (rescued=%d)  not_usable=%d  degenerate=%d" % (
        len(usable) + len(rescued), len(rescued), len(dead), len(degenerate)))

    # Where does max/topk/std actually win over the plain mean? This is the
    # mechanistic question, independent of the usable/not-usable verdict.
    print()
    print("=" * 96)
    print("SPATIAL vs MEAN, per field (prespecified topk_mean and max, against mean)")
    print("%-28s %8s %8s %8s" % ("field", "mean", "topk", "max"))
    for field in sorted(res):
        e = res[field]
        def g(suffix):
            ks = [k for k in e if k.endswith(":" + suffix)]
            if not ks:
                return None
            return e[max(ks, key=lambda k: e[k]["delta_ci95"][0])]["delta"]
        m, t, x = g("mean"), g(PRESPEC), g("max")
        if None in (m, t, x):
            continue
        print("%-28s %+8.3f %+8.3f %+8.3f" % (field, m, t, x))


if __name__ == "__main__":
    main()
