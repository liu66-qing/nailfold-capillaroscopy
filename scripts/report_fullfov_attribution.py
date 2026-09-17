"""Attribute the fullfov gains: field of view, or a different loader?

square_control uses THIS pipeline at 518x518 with the same CenterCrop as the
original artifact. So:

  square_control vs reference_audit  = pipeline/loader effect (should be ~0 if the
                                       audit is reproducible; anything large here
                                       means my loader differs, not the FOV)
  native vs square_control           = field-of-view effect, the thing under test
  native_hi vs native                = extra resolution on top of full FOV

Printing all three keeps me from reporting a loader artifact as a capability gain.
A field only counts as rescued if its CI lower bound crosses from <=0 to >0 AND
native beats square_control by more than the 0.08 seed-noise band recorded in
AUDIT_all_fields_20260915 section 7.
"""
import json
from pathlib import Path

SUMMARY = Path("/root/nailfold/artifacts/experiments/fullfov_20260917/summary.json")
NOISE_BAND = 0.08


def best_of(entry, prefix):
    """Best variant of one preset, ranked by CI lower bound (not point estimate)."""
    cands = [(k, v) for k, v in entry.items()
             if k.startswith(prefix + ":") and isinstance(v, dict) and "delta" in v]
    if not cands:
        return None, None
    return max(cands, key=lambda kv: kv[1]["delta_ci95"][0])


def main():
    report = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = report["results"]

    print("Bootstrap:", report["bootstrap"], " locked used:",
          report["locked_cases_used_for_fitting_or_scoring"])
    print()
    head = ("field", "audit_d", "sq_d", "sq_lo", "nat_d", "nat_lo", "hi_d", "hi_lo", "nat-sq")
    print("%-28s %8s %8s %8s %8s %8s %8s %8s %8s" % head)

    rescued, improved, unchanged = [], [], []
    for field in sorted(results):
        entry = results[field]
        if "skipped" in entry:
            continue
        audit = entry["reference_audit"]["delta"]
        sq_name, sq = best_of(entry, "square_control")
        nat_name, nat = best_of(entry, "native")
        hi_name, hi = best_of(entry, "native_hi")
        if not (sq and nat and hi):
            continue
        gap = nat["delta"] - sq["delta"]
        print("%-28s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f" % (
            field, audit, sq["delta"], sq["delta_ci95"][0],
            nat["delta"], nat["delta_ci95"][0],
            hi["delta"], hi["delta_ci95"][0], gap))

        best_new = max([nat, hi], key=lambda v: v["delta_ci95"][0])
        crossed = audit <= 0 < best_new["delta_ci95"][0]
        beat_noise = (best_new["delta"] - sq["delta"]) > NOISE_BAND
        if crossed and beat_noise:
            rescued.append((field, audit, best_new))
        elif best_new["delta_ci95"][0] > 0 and (best_new["delta"] - audit) > NOISE_BAND:
            improved.append((field, audit, best_new))
        else:
            unchanged.append(field)

    print()
    print("=" * 78)
    print("LOADER CHECK  (square_control vs audit -- should be small)")
    diffs = []
    for field in sorted(results):
        entry = results[field]
        if "skipped" in entry:
            continue
        _, sq = best_of(entry, "square_control")
        if sq:
            diffs.append(abs(sq["delta"] - entry["reference_audit"]["delta"]))
    if diffs:
        print("  |square_control - audit| : median %.3f  max %.3f" % (
            sorted(diffs)[len(diffs) // 2], max(diffs)))
        print("  NOTE: square_control picks the best of 3 poolings by CI lower bound,")
        print("        the audit used one fixed source. Some positive gap is expected")
        print("        from that selection alone, so it is NOT all loader effect.")

    print()
    print("RESCUED  (audit delta <= 0, now CI lower bound > 0, and FOV gain > %.2f)" % NOISE_BAND)
    if rescued:
        for f, a, v in rescued:
            print("  %-26s %+.3f -> %+.3f  CI[%+.3f,%+.3f]  sens %.3f spec %.3f" % (
                f, a, v["delta"], v["delta_ci95"][0], v["delta_ci95"][1],
                v["sensitivity"], v["specificity"]))
    else:
        print("  none")

    print()
    print("IMPROVED  (already usable, gain > %.2f over audit)" % NOISE_BAND)
    for f, a, v in improved:
        print("  %-26s %+.3f -> %+.3f  CI[%+.3f,%+.3f]" % (
            f, a, v["delta"], v["delta_ci95"][0], v["delta_ci95"][1]))
    if not improved:
        print("  none")

    print()
    print("UNCHANGED  (%d): %s" % (len(unchanged), ", ".join(unchanged)))


if __name__ == "__main__":
    main()
