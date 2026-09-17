"""Score auxiliary visual review against existing human XML annotation.

Thresholds are hard-coded here, before any review exists, so they cannot be
chosen after seeing results. v1 skipped this step entirely (finding F05).

Run only after visual_review_queue_v2.csv has reviewer-filled rows:
    python scripts/score_visual_review_agreement.py

Exit code 1 = gate not met. Labels must then NOT be used, and the failure mode
gets written up.
"""

import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "artifacts/annotations/v2"
ANN_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/annotations"

# Frozen acceptance thresholds (protocol v2).
#
# The first draft gated on box presence (kappa >= 0.60) and in-frame fraction
# (weighted kappa >= 0.50). Both were withdrawn after measuring the reference
# across all 82 admitted images: box presence is 82/82 'usable' (single
# category -> kappa undefined), and in-frame fraction is 76 'all' / 6
# 'majority' (so skewed that 3 disagreements drop weighted kappa 1.00 -> 0.78
# while a constant 'all' answer scores 0.00). Those thresholds measured
# marginal skew, not agreement.
#
# What survives is what the XML actually has variance in. Re-measured on the
# 129-image pool after tier-2 admission (2026-09-17): count spans 2-13 with all
# 12 values present, cross_vessel 72/129 = 55.8% (closer to the 50/50 that makes
# kappa most informative than tier 1's 40/82 = 48.8%), malformed 102/129 = 79.1%.
# Tier-2 admission therefore did not degrade the reference, and the thresholds
# below are unchanged from before the pool grew -- they were frozen first.
GATE_COUNT_SPEARMAN = 0.70
GATE_COUNT_WITHIN1_FRAC = 0.60
GATE_CROSS_VESSEL_KAPPA = 0.50
GATE_IOU = 0.50
GATE_IOU_FRAC = 0.80

# Case-level requirement. 129 images sit on only 49 cases (tier 2 added images,
# not patients), and 6 near-duplicate clusters are resized siblings of each
# other. Agreement computed per image therefore overstates the evidence: the
# independent unit is the case. The gate is evaluated on BOTH the per-image and
# the case-collapsed statistic, and must pass on the case-collapsed one.
GATE_REQUIRE_CASE_LEVEL_PASS = True
# Minimum independent cases before any gate verdict is meaningful at all.
GATE_MIN_CASES = 20

# Collected but unvalidatable against this dataset's reference.
UNVALIDATED_FIELDS = [
    "measurable_vessel_fraction",
    "apex_visible_fraction",
    "image_usable_for_measurement",
    "clarity_review",
]

FRACTION_ORDER = ["none", "minority", "majority", "all"]


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa on two aligned label sequences."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = [0.0] * len(cats)
    pb = [0.0] * len(cats)
    for x in a:
        pa[idx[x]] += 1 / n
    for y in b:
        pb[idx[y]] += 1 / n
    exp = sum(pa[i] * pb[i] for i in range(len(cats)))
    if exp == 1.0:
        return None
    return (obs - exp) / (1 - exp)


def weighted_kappa(a, b, order):
    """Linearly-weighted kappa for ordered categories."""
    pos = {c: i for i, c in enumerate(order)}
    a = [x for x in a]
    b = [y for y in b]
    n = len(a)
    if n == 0:
        return None
    k = len(order)
    maxd = k - 1
    obs = sum(1 - abs(pos[x] - pos[y]) / maxd for x, y in zip(a, b)) / n
    ca = [sum(1 for x in a if x == c) / n for c in order]
    cb = [sum(1 for y in b if y == c) / n for c in order]
    exp = sum(
        ca[i] * cb[j] * (1 - abs(i - j) / maxd) for i in range(k) for j in range(k)
    )
    if exp == 1.0:
        return None
    return (obs - exp) / (1 - exp)


def spearman(a, b):
    """Spearman rho via Pearson on midranks. No scipy dependency."""
    n = len(a)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mid
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def xml_reference(oid):
    """Image-level reference from the human boxes.

    Returns only quantities the XML has real variance in: total box count and
    per-class presence. Box presence as a measurability proxy was dropped --
    it is 82/82 'usable' across the admitted pool, because these images were
    annotated precisely because they contain traceable vessels.
    """
    x = ANN_DIR / f"{oid}.xml"
    if not x.exists():
        return None
    root = ET.parse(x).getroot()
    objs = root.findall("object")
    names = [o.find("name").text for o in objs]
    return {
        "n_boxes": len(objs),
        "has_cross_vessel": "cross_vessel" in names,
        "has_malformed_vessel": "malformed_vessel" in names,
    }


def main():
    qf = V2 / "visual_review_queue_v2.csv"
    q = pd.read_csv(qf)
    done = q[q.review_status != "pending"].copy()

    report = {
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queue": str(qf.relative_to(ROOT)),
        "queue_rows": int(len(q)),
        "reviewed_rows": int(len(done)),
        "thresholds": {
            "count_spearman": GATE_COUNT_SPEARMAN,
            "count_within1_fraction": GATE_COUNT_WITHIN1_FRAC,
            "cross_vessel_kappa": GATE_CROSS_VESSEL_KAPPA,
            "iou": GATE_IOU,
            "iou_fraction": GATE_IOU_FRAC,
        },
        "reference": "existing human XML boxes: total box count (2-13, all 12 "
        "values present across the 129-image pool) and cross_vessel presence "
        "(72/129 = 55.8%). Auxiliary reference only, never a clinical gold "
        "standard.",
        "unvalidated_fields": UNVALIDATED_FIELDS,
        "unvalidated_note": "No valid reference exists in this dataset for these "
        "fields: box presence is 82/82 usable and in-frame fraction is 76 all / "
        "6 majority, both too degenerate to gate on. They are collected but may "
        "not justify any downstream claim; validating them needs a second "
        "independent human reviewer.",
        "independence_note": "129 images rest on 49 cases and include 6 "
        "near-duplicate clusters, so per-image n overstates the evidence. Every "
        "statistic is reported on four slices (per-image, case-collapsed, "
        "near-duplicate-removed, tier-1-only) and the gate verdict requires the "
        "case-collapsed slice to pass.",
    }

    if len(done) == 0:
        report["gate"] = "NOT_EVALUABLE"
        report["reason"] = "no reviewed rows yet"
        out = V2 / "agreement_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # Protocol v2 rule 4: low confidence and unknown are excluded from both
    # numerator and denominator, and the exclusion count is reported.
    excl_low = int((done.annotation_confidence == "low").sum())
    keep = done[done.annotation_confidence != "low"].copy()
    report["excluded_low_confidence"] = excl_low
    report["scored_after_exclusions"] = int(len(keep))

    if "reviewer_vessel_count" not in keep.columns:
        report["gate"] = "NOT_EVALUABLE"
        report["reason"] = (
            "queue has no reviewer_vessel_count column; the count check is the "
            "primary gate and cannot be computed"
        )
        out = V2 / "agreement_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    rows = []
    for r in keep.itertuples():
        ref = xml_reference(int(r.original_id))
        if ref is None:
            continue
        try:
            rev_n = int(r.reviewer_vessel_count)
        except (TypeError, ValueError):
            continue
        rev_cv = str(getattr(r, "reviewer_cross_vessel_present", "")).strip().lower()
        if rev_cv not in ("yes", "no"):
            rev_cv = None
        rows.append(
            {
                "original_id": int(r.original_id),
                "exam_case_id": r.exam_case_id,
                "rev_n": rev_n,
                "ref_n": ref["n_boxes"],
                "rev_cv": rev_cv,
                "ref_cv": "yes" if ref["has_cross_vessel"] else "no",
            }
        )
    df = pd.DataFrame(rows)
    report["scored_with_reference"] = int(len(df))
    report["unique_cases_scored"] = int(df.exam_case_id.nunique()) if len(df) else 0

    if len(df) < 3:
        report["gate"] = "NOT_EVALUABLE"
        report["reason"] = f"only {len(df)} scorable rows; need >= 3 for a rank correlation"
    else:
        def stats(sub, label):
            """Agreement on one slice. Returns the three gate quantities."""
            if len(sub) < 3:
                return {"slice": label, "n": int(len(sub)), "evaluable": False}
            rho = spearman(sub.rev_n.tolist(), sub.ref_n.tolist())
            w1 = float((sub.rev_n - sub.ref_n).abs().le(1).mean())
            cvv = sub[sub.rev_cv.notna()]
            k = (
                cohen_kappa(cvv.rev_cv.tolist(), cvv.ref_cv.tolist())
                if len(cvv) >= 3
                else None
            )
            return {
                "slice": label,
                "n": int(len(sub)),
                "cases": int(sub.exam_case_id.nunique()),
                "evaluable": True,
                "count_spearman": None if rho is None else round(rho, 4),
                "count_within1_fraction": round(w1, 4),
                "count_mean_abs_error": round(
                    float((sub.rev_n - sub.ref_n).abs().mean()), 3
                ),
                "cross_vessel_scored_rows": int(len(cvv)),
                "cross_vessel_kappa": None if k is None else round(k, 4),
                "passes": {
                    "count_spearman": rho is not None and rho >= GATE_COUNT_SPEARMAN,
                    "count_within1": w1 >= GATE_COUNT_WITHIN1_FRAC,
                    "cross_vessel_kappa": k is not None and k >= GATE_CROSS_VESSEL_KAPPA,
                },
            }

        # Case-collapsed slice: one row per case, so 49 cases carrying 129 images
        # cannot inflate the sample. Counts are averaged (mean of reviewer vs mean
        # of reference preserves the rank relation being tested); cross_vessel
        # takes the case-level "any", matching how a case-level claim would be made.
        by_case = (
            df.groupby("exam_case_id")
            .agg(
                rev_n=("rev_n", "mean"),
                ref_n=("ref_n", "mean"),
                rev_cv=("rev_cv", lambda s: "yes" if (s == "yes").any() else ("no" if s.notna().any() else None)),
                ref_cv=("ref_cv", lambda s: "yes" if (s == "yes").any() else "no"),
            )
            .reset_index()
        )
        by_case["exam_case_id"] = by_case.exam_case_id

        # Deduplicated slice: one image per near-duplicate cluster (lowest id),
        # so resized siblings are not double-counted.
        if "near_duplicate_group" in keep.columns:
            grp = dict(zip(keep.original_id.astype(int), keep.near_duplicate_group.fillna("")))
            seen, keep_ids = set(), []
            for oid in sorted(df.original_id):
                g = grp.get(oid, "")
                if g and g in seen:
                    continue
                if g:
                    seen.add(g)
                keep_ids.append(oid)
            dedup = df[df.original_id.isin(keep_ids)]
        else:
            dedup = df

        per_image = stats(df, "per_image_all_tiers")
        per_case = stats(by_case, "case_collapsed")
        report["slices"] = {
            "per_image_all_tiers": per_image,
            "case_collapsed": per_case,
            "per_image_near_dup_removed": stats(dedup, "per_image_near_dup_removed"),
        }
        if "admission_tier" in keep.columns:
            t = dict(zip(keep.original_id.astype(int), keep.admission_tier))
            t1 = df[df.original_id.map(t) == "tier1_md5"]
            report["slices"]["tier1_only"] = stats(t1, "tier1_md5_only")

        # Headline numbers stay per-image for continuity with the frozen
        # thresholds, but the VERDICT is governed by the case-collapsed slice.
        for k_ in (
            "count_spearman",
            "count_within1_fraction",
            "count_mean_abs_error",
            "cross_vessel_scored_rows",
            "cross_vessel_kappa",
        ):
            report[k_] = per_image.get(k_)
        report["degenerate_note"] = (
            "kappa is undefined when one rater used a single category; that is a "
            "failure signal, not a pass"
        )

        n_cases = int(df.exam_case_id.nunique())
        report["independent_cases"] = n_cases
        checks = dict(per_image["passes"])
        checks["case_collapsed_pass"] = bool(
            per_case.get("evaluable") and all(per_case["passes"].values())
        )
        checks["enough_independent_cases"] = n_cases >= GATE_MIN_CASES
        report["checks"] = checks
        report["gate"] = "PASS" if all(checks.values()) else "FAIL"
        report["verdict_basis"] = (
            f"per-image AND case-collapsed must both pass; {n_cases} independent "
            f"cases (minimum {GATE_MIN_CASES}). Per-image alone is not sufficient "
            "because 129 images sit on 49 cases."
        )

        # v1 finding F06: if the two fraction fields track each other everywhere,
        # the distinction is not operational and they should be merged.
        if {"measurable_vessel_fraction", "apex_visible_fraction"} <= set(keep.columns):
            same = int((keep.measurable_vessel_fraction == keep.apex_visible_fraction).sum())
            report["fraction_fields_identical_rows"] = same
            report["fraction_fields_collapsed"] = bool(len(keep) > 0 and same == len(keep))

    out = V2 / "agreement_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("gate") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
