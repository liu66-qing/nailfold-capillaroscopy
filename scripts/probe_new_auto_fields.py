# -*- coding: utf-8 -*-
"""Assess the three AUTOMATIC quantities the field table demands be evaluated as
NEW fields rather than as stand-ins for the clinical ones.

  1. automatic crossing box count       pooled_n_crossing  (external 4-class detector)
  2. visible loop count in the frame    pooled_n_total
  3. locally suspicious red dots        -- no detector class exists, reported as absent

There is NO human gold standard for any of these. Nobody annotated "how many loops
should be visible in this frame" or "how many crossings are in this image"; the
report gives a BAND over the whole examination (`30--60%`, `>=7`). So accuracy is
not computable and is not claimed. What IS computable, and what the table actually
asks ("is the alternative information equivalent?"), is whether the automatic count
carries a monotone relation to the clinical field it could be confused with, and
how large the scale offset is.

development only. locked-47 is never opened; the split comes from
server_code_audit/locked_evaluation_v1_reviewed.csv:development_fold (NaN == locked),
and every feature table is filtered case-level before any statistic is computed.
"""
import io
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "artifacts/audits/new_field_probe_20260925"
os.makedirs(OUT, exist_ok=True)

EXT = "artifacts/experiments/rescue_external_20260922/local_features/external/case_features.csv"
LOC = "artifacts/experiments/rescue_external_20260922/local_features/local/case_features.csv"
REV = "server_code_audit/locked_evaluation_v1_reviewed.csv"


def write(name, obj):
    """json.dump on Windows picks the ANSI codepage; force utf-8."""
    io.open(os.path.join(OUT, name), "w", encoding="utf-8").write(
        json.dumps(obj, ensure_ascii=False, indent=1))


rev = pd.read_csv(REV, dtype={"exam_case_id": str})
dev_ids = set(rev.loc[rev.development_fold.notna(), "exam_case_id"])
locked_ids = set(rev.loc[rev.development_fold.isna(), "exam_case_id"])
assert len(dev_ids) == 186 and len(locked_ids) == 47

feat = {}
dropped = {}
for nm, path in (("external", EXT), ("local", LOC)):
    d = pd.read_csv(path, dtype={"exam_case_id": str})
    hit = sorted(set(d.exam_case_id) & locked_ids)
    dropped[nm] = len(hit)
    d = d[~d.exam_case_id.isin(locked_ids)].copy()
    assert not (set(d.exam_case_id) & locked_ids)
    feat[nm] = d

LOCKED_SEEN = 0  # nothing locked is read, decoded, or featurised here

# Band -> a single number to correlate against. These are ORDINAL ranks, not
# micrometres and not percentages: ranking is all a rank correlation needs, and
# committing to a midpoint would imply a precision the labels do not carry.
RANKS = {
    "crossing_ratio": {"<=30%": 0, "[<30%]": 0, "10--30%": 0,
                       "30--60%": 1, "60--80%": 2, ">80%": 3},
    "capillary_count": {"<1": 0, "3--4": 1, "5--6": 2, ">=7": 3},
    "hemorrhage": {u"无": 0, "1--2": 1},
}
# values that are a unit string or a bracketed unconfirmable token go to unknown,
# never silently to normal or abnormal
DROP = {u"条/mm", u"个/min", u"管袢/一指甲襞"}


def ranked(series, field):
    m = RANKS[field]
    out = series.map(lambda v: m.get(v, np.nan) if isinstance(v, str) else np.nan)
    unmapped = sorted({v for v in series.dropna().unique()
                       if isinstance(v, str) and v not in m})
    return out, unmapped


def spearman(x, y):
    ok = (~pd.isna(x)) & (~pd.isna(y))
    x, y = np.asarray(x[ok], float), np.asarray(y[ok], float)
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "rho": None, "p": None,
                "why": "fewer than 10 paired cases or a constant column"}
    rho, p = stats.spearmanr(x, y)
    # bootstrap the rho so a reader sees the resolution, not just a point estimate
    rng = np.random.default_rng(20260917)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        boot.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(x)), "rho": round(float(rho), 4),
            "p": float("%.4g" % p),
            "rho_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


results = {}

# ---- new field 1: automatic crossing box count -------------------------------
ext = feat["external"].merge(
    rev[["exam_case_id", "crossing_ratio", "capillary_count", "hemorrhage"]],
    on="exam_case_id", how="inner")
cr_rank, cr_unmapped = ranked(ext.crossing_ratio, "crossing_ratio")

results["auto_crossing_box_count"] = {
    "what_it_is": "pooled_n_crossing from the external-pretrained 4-class detector, "
                  "pooled over a case's images",
    "human_gold_standard": "none exists -- no one annotated a crossing count per "
                           "image; the report gives a whole-examination band",
    "so_accuracy_is": "not computable, and not claimed",
    "distribution": {
        "n_cases": int(ext.pooled_n_crossing.notna().sum()),
        "median": float(ext.pooled_n_crossing.median()),
        "p25": float(ext.pooled_n_crossing.quantile(.25)),
        "p75": float(ext.pooled_n_crossing.quantile(.75)),
        "min": float(ext.pooled_n_crossing.min()),
        "max": float(ext.pooled_n_crossing.max()),
        "zero_share": round(float((ext.pooled_n_crossing == 0).mean()), 4),
    },
    "vs_clinical_crossing_ratio_band": spearman(ext.pooled_n_crossing, cr_rank),
    "unmapped_label_values_sent_to_unknown": cr_unmapped,
}

# ---- new field 2: visible loop count in the frame ----------------------------
cc_rank, cc_unmapped = ranked(ext.capillary_count, "capillary_count")
results["visible_loop_count_in_frame"] = {
    "what_it_is": "pooled_n_total from the same detector: how many loops the "
                  "detector finds inside the frame",
    "human_gold_standard": "none exists for the in-frame count",
    "not_equivalent_to": "the clinical field is loops per millimetre. This count "
                         "has no length denominator and the device is UNCALIBRATED "
                         "(device_calibration_status.json), so no per-mm value may "
                         "be emitted from it",
    "distribution": {
        "n_cases": int(ext.pooled_n_total.notna().sum()),
        "median": float(ext.pooled_n_total.median()),
        "p25": float(ext.pooled_n_total.quantile(.25)),
        "p75": float(ext.pooled_n_total.quantile(.75)),
        "min": float(ext.pooled_n_total.min()),
        "max": float(ext.pooled_n_total.max()),
    },
    "vs_clinical_capillary_count_band": spearman(ext.pooled_n_total, cc_rank),
    "unmapped_label_values_sent_to_unknown": cc_unmapped,
}

# same probe using the local 3-class detector, as a second opinion
loc = feat["local"].merge(
    rev[["exam_case_id", "crossing_ratio", "capillary_count"]],
    on="exam_case_id", how="inner")
lcr, _ = ranked(loc.crossing_ratio, "crossing_ratio")
lcc, _ = ranked(loc.capillary_count, "capillary_count")
results["second_opinion_local_detector"] = {
    "why": "if the two independently trained detectors disagree in SIGN, neither "
           "count is carrying the clinical quantity",
    "n_cross_vs_crossing_band": spearman(loc.pooled_n_cross, lcr),
    "n_total_vs_count_band": spearman(loc.pooled_n_total, lcc),
}

# ---- new field 3: locally suspicious red dots --------------------------------
det_classes_external = ["bushy", "crossing", "hairpin", "tortuous"]
det_classes_local = ["vessel", "malformed", "cross"]
hem = rev.loc[rev.development_fold.notna(), "hemorrhage"]
results["locally_suspicious_red_dots"] = {
    "detector_class_available": False,
    "external_detector_classes": det_classes_external,
    "local_detector_classes": det_classes_local,
    "conclusion": "no detector on disk emits a haemorrhage/red-dot class, so this "
                  "new field has no automatic quantity to evaluate. Building one "
                  "needs region annotations, of which we hold zero.",
    "label_side_note": {
        "development_distribution": {str(k): int(v) for k, v in
                                     hem.value_counts(dropna=False).items()},
        "correction": "the clinical hemorrhage field is NOT constant on development: "
                      "168 are 'wu' but 12 are '1--2'. Fixing the output to 'wu' "
                      "would be wrong on those 12 cases; that is a coverage "
                      "decision, not a measured accuracy",
    },
}

results["_provenance"] = {
    "split_source": REV + " :development_fold (NaN == locked-47)",
    "development_cases": len(dev_ids),
    "locked_cases": len(locked_ids),
    "locked_rows_dropped_from_feature_tables": dropped,
    "locked_cases_seen_this_run": LOCKED_SEEN,
    "seed": 20260917,
    "n_boot": 2000,
    "limitations": [
        "a rank correlation says the automatic count moves WITH the clinical band; "
        "it does not say the count is right, and it cannot be turned into accuracy",
        "the clinical bands are coarse (4 levels for crossing, 4 for count), which "
        "caps any achievable rho",
        "counts are detector outputs; a detector miss and a genuine absence are "
        "indistinguishable here",
        "development only: nothing here is evidence about held-out performance",
    ],
}

write("new_field_probe.json", results)

for k, v in results.items():
    if k.startswith("_"):
        continue
    print("=" * 72)
    print(k)
    print(json.dumps(v, ensure_ascii=False, indent=1))
print("\nlocked_cases_seen_this_run:", LOCKED_SEEN)
