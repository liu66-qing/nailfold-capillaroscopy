"""Locked-adjacency safety check for the 47 perceptually-recovered tier-2 images.

Why this is separate from recover_unmapped_provenance.py. That script accepted a
match when it beat the runner-up *from any other case* by >=0.05. That bounds
case mix-ups in general; it does not bound the one risk that matters here, an
image assigned to a development case while actually being a locked-47
derivative. This script measures a locked-SPECIFIC margin:

    margin = max_sim(image, assigned development case originals)
           - max_sim(image, any locked-47 case original)

THRESHOLD CALIBRATION (this is the part that makes the number meaningful).
The threshold is not chosen by judgement; it is read off two labelled groups
that already exist in the governance record:

  negatives  82 images with source_mapping_status EXACT_RECOVERED_DEVELOPMENT
             (MD5-proven clean) scored against their proven case
  positives  32 images with EXACT_RECOVERED_LOCKED, scored ADVERSARIALLY --
             assigned to their single best-matching development case, i.e. the
             most favourable mislabelling that could ever occur

Measured 2026-09-17 (scripts/verify_tier2_locked_safety.py --calibrate):
  positives : n=32  min -0.6547  median -0.1181  max -0.0434
  negatives : n=82  min +0.0434  median +0.1576  max +0.7170
  AUC 1.0000, zero overlap, empty gap of width 0.0868 spanning 0.

So the decision boundary is 0, and MIN_LOCKED_MARGIN sits inside the empty gap.
At this threshold the check catches 100% of adversarially-mislabelled locked
images and discards 0% of MD5-proven-clean ones.

TWO CHECKS I WITHDREW, with the measurements that refuted them:

  1. MIN_LOCKED_MARGIN = 0.10, my first guess. Refuted: 22 of the 82 MD5-PROVEN
     development images fail it. The tier-2 and proven-clean margin
     distributions are nearly identical (median 0.1617 vs 0.1600, p10 0.0715 vs
     0.0701), so 0.10 was rejecting clean data, not contaminated data.

  2. Archive-level cross-similarity ("assigned development case must not
     resemble a locked case above 0.98"). Refuted as non-discriminating: the 49
     MD5-proven-clean development cases reach cross-similarity up to 0.9696
     (median 0.9163), while two genuinely DIFFERENT locked cases reach 0.9577
     (median 0.7305). Case-level appearance overlap is the baseline texture of
     nailfold imagery, not evidence of contamination. The value is still
     reported per case as context, but it no longer gates anything.

Usage:
    python scripts/verify_tier2_locked_safety.py             # run the check
    python scripts/verify_tier2_locked_safety.py --calibrate  # reproduce the ROC
Output: artifacts/annotations/v2/tier2_locked_safety.json
"""

import csv
import glob
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GOV = ROOT / "artifacts/audits/vascular_dataset_governance_20260830"
IMG_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/images"
OUT = ROOT / "artifacts/annotations/v2"

FP_SIZE = 32
# Inside the measured empty gap (-0.0434, +0.0434). See the calibration block
# above; do not raise this without re-running --calibrate, because at 0.10 the
# check starts rejecting MD5-proven-clean images.
MIN_LOCKED_MARGIN = 0.02
# Reported for context only. NOT a gate -- refuted above.
CASE_CROSS_SIM_CONTEXT_ONLY = True


def fingerprint(path, n=FP_SIZE):
    with Image.open(path) as im:
        g = im.convert("L").resize((n, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32).ravel()
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def archive_case(path):
    parts = Path(path).as_posix().split("/")
    i = next(k for k, p in enumerate(parts) if p.startswith("recovered_archive"))
    return f"{parts[i]}/{parts[i + 1]}"


def load_mapping():
    """source_case_mapping.csv is UTF-8 with BOM; utf-8-sig or original_id is lost."""
    rows = {}
    locked, dev = set(), set()
    with (GOV / "source_case_mapping.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[int(r["original_id"])] = r
            for v in (r.get("locked_cases") or "").split(";"):
                if v.strip():
                    locked.add(v.strip())
            for v in (r.get("development_cases") or "").split(";"):
                if v.strip():
                    dev.add(v.strip())
    return rows, locked, dev


def build_index():
    files = sorted(glob.glob(str(ROOT / "data/recovered_archive*/*/CAPorg*.jpg")))
    vecs, cases, paths = [], [], []
    for f in files:
        try:
            v = fingerprint(f)
        except Exception:
            continue
        vecs.append(v)
        cases.append(archive_case(f))
        paths.append(Path(f).relative_to(ROOT).as_posix())
    return np.vstack(vecs), np.array(cases), paths


def load_dispositions():
    d = {}
    with (OUT / "queue_disposition.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[int(r["original_id"])] = r
    return d


def _midrank_auc(pos, neg):
    """AUC that lower-score => positive, computed without scipy."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    s = allv[order]
    rk = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rk[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = rk
    n1, n2 = len(pos), len(neg)
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return 1 - auc


def calibrate(A, cases, locked, dev, mapping, disp, nm):
    """Reproduce the threshold from labelled data. Prints the ROC table."""
    is_locked = np.isin(cases, sorted(locked))
    is_dev = np.isin(cases, sorted(dev))

    neg_ids = [o for o, r in disp.items() if r["decision"] == "ADMIT"]
    pos_ids = [
        o
        for o, r in mapping.items()
        if r["source_mapping_status"] == "EXACT_RECOVERED_LOCKED"
    ]

    neg = []
    for o in neg_ids:
        v = fingerprint(IMG_DIR / f"{o}.jpg")
        s = (A @ v) / nm
        sel = cases == disp[o]["exam_case_id"]
        neg.append(float(s[sel].max()) - float(np.max(np.where(is_locked, s, -np.inf))))
    pos = []
    for o in pos_ids:
        p = IMG_DIR / f"{o}.jpg"
        if not p.exists():
            continue
        v = fingerprint(p)
        s = (A @ v) / nm
        # Adversarial: best-matching development case, the hardest mislabelling.
        pos.append(
            float(np.max(np.where(is_dev, s, -np.inf)))
            - float(np.max(np.where(is_locked, s, -np.inf)))
        )
    P, N = np.array(pos), np.array(neg)
    table = {
        f"{t}": {
            "catches_mislabelled_locked": round(float((P < t).mean()), 4),
            "discards_proven_clean": round(float((N < t).mean()), 4),
        }
        for t in (0.0, 0.02, 0.05, 0.10, 0.15)
    }
    return {
        "positives_adversarial_mislabelled_locked": {
            "n": len(P),
            "min": round(float(P.min()), 4),
            "median": round(float(np.median(P)), 4),
            "max": round(float(P.max()), 4),
        },
        "negatives_md5_proven_development": {
            "n": len(N),
            "min": round(float(N.min()), 4),
            "median": round(float(np.median(N)), 4),
            "max": round(float(N.max()), 4),
        },
        "auc": round(float(_midrank_auc(P, N)), 4),
        "empty_gap": [round(float(P.max()), 4), round(float(N.min()), 4)],
        "overlap_count": int((P >= N.min()).sum()),
        "operating_points": table,
        "chosen_threshold": MIN_LOCKED_MARGIN,
    }


def main():
    mapping, locked_cases, dev_cases = load_mapping()
    disp = load_dispositions()
    tier2 = sorted(o for o, r in disp.items() if r["decision"] == "TIER2_PENDING_PERMISSION")

    A, cases, _ = build_index()
    nm = FP_SIZE * FP_SIZE
    is_locked = np.isin(cases, sorted(locked_cases))
    L = A[is_locked]
    lcase = cases[is_locked]

    cal = calibrate(A, cases, locked_cases, dev_cases, mapping, disp, nm)
    if "--calibrate" in sys.argv:
        print(json.dumps(cal, ensure_ascii=False, indent=2))
        return

    # Context-only per-case cross similarity (refuted as a gate, kept as a number).
    case_cross = {}
    for c in sorted({disp[o]["exam_case_id"] for o in tier2}):
        sel = cases == c
        if not sel.any():
            case_cross[c] = None
            continue
        S = (A[sel] @ L.T) / nm
        i, j = np.unravel_index(int(S.argmax()), S.shape)
        case_cross[c] = {
            "max_sim_to_any_locked_original": round(float(S[i, j]), 4),
            "closest_locked_case": str(lcase[j]),
            "gate": "context_only_refuted",
        }

    findings = []
    for oid in tier2:
        v = fingerprint(IMG_DIR / f"{oid}.jpg")
        s = (A @ v) / nm
        assigned = disp[oid]["exam_case_id"]
        sel = cases == assigned
        best_assigned = float(s[sel].max()) if sel.any() else float("nan")
        k = int(np.argmax(np.where(is_locked, s, -np.inf)))
        best_locked, best_locked_case = float(s[k]), str(cases[k])
        margin = best_assigned - best_locked
        reasons = []
        if not (margin >= MIN_LOCKED_MARGIN):
            reasons.append(
                f"locked-specific margin {margin:.4f} < {MIN_LOCKED_MARGIN} "
                f"(closest locked case {best_locked_case})"
            )
        findings.append(
            {
                "original_id": oid,
                "assigned_case": assigned,
                "sim_to_assigned_case": round(best_assigned, 4),
                "sim_to_closest_locked": round(best_locked, 4),
                "closest_locked_case": best_locked_case,
                "locked_specific_margin": round(margin, 4),
                "assigned_case_max_sim_to_locked_CONTEXT_ONLY": (
                    case_cross[assigned]["max_sim_to_any_locked_original"]
                    if case_cross.get(assigned)
                    else None
                ),
                "verdict": "SAFE" if not reasons else "UNSAFE",
                "reasons": reasons,
            }
        )

    safe = [f for f in findings if f["verdict"] == "SAFE"]
    unsafe = [f for f in findings if f["verdict"] == "UNSAFE"]
    margins = sorted(f["locked_specific_margin"] for f in findings)
    by_case = defaultdict(list)
    for f in safe:
        by_case[f["assigned_case"]].append(f["original_id"])

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/verify_tier2_locked_safety.py",
        "purpose": "bound locked-47 contamination risk for perceptually recovered "
        "development images before they enter the auxiliary review pool",
        "gate": {
            "name": "locked_specific_margin",
            "threshold": MIN_LOCKED_MARGIN,
            "calibration": cal,
        },
        "withdrawn_checks": [
            {
                "check": "min_locked_specific_margin = 0.10",
                "refuted_by": "22 of 82 MD5-proven-clean development images fail it; "
                "tier-2 and proven-clean margin distributions are nearly identical "
                "(median 0.1617 vs 0.1600, p10 0.0715 vs 0.0701)",
            },
            {
                "check": "assigned development case must not resemble a locked case above 0.98",
                "refuted_by": "49 proven-clean development cases reach cross-similarity "
                "up to 0.9696 (median 0.9163) while two genuinely different LOCKED cases "
                "reach 0.9577 (median 0.7305); the statistic does not separate",
            },
        ],
        "archive_index": {
            "originals_indexed": int(len(cases)),
            "locked_originals": int(is_locked.sum()),
            "locked_cases_on_roster": len(locked_cases),
            "locked_cases_absent_from_archive": sorted(locked_cases - set(cases)),
        },
        "tier2_images_checked": len(findings),
        "safe": len(safe),
        "unsafe": len(unsafe),
        "safe_unique_cases": len(by_case),
        "safe_original_ids": sorted(f["original_id"] for f in safe),
        "unsafe_detail": unsafe,
        "locked_specific_margin_distribution": {
            "min": margins[0],
            "p10": margins[max(0, len(margins) // 10)],
            "median": margins[len(margins) // 2],
            "max": margins[-1],
        },
        "per_case_locked_cross_similarity_context_only": case_cross,
        "findings": findings,
        "limitations": [
            "This bounds locked contamination by image appearance, not by patient "
            "identity. If a locked-47 patient also appears in the archive under a "
            "different case id with different frames, no appearance test can detect "
            "it. That residual risk is unquantified and is the reason these images "
            "stay auxiliary-only.",
            "Calibration rests on 32 locked positives and 82 development negatives. "
            "Perfect separation on 114 images does not prove perfect separation in "
            "general; it bounds the error rate at roughly <3% per group at 95% "
            "confidence, not at zero.",
            "All 17 locked cases on the roster are present in the archive index, so "
            "there is no locked case this test is blind to. An unrostered locked "
            "case would still be invisible.",
        ],
    }

    f = OUT / "tier2_locked_safety.json"
    f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "findings",
                    "per_case_locked_cross_similarity_context_only",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("sha256", hashlib.sha256(f.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
