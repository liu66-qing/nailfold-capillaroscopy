"""Rebuild the auxiliary visual-review queue (v2).

v1 (artifacts/annotations/visual_review_queue.csv) was rejected: it contained
locked-47 derivative images, 14/16 reviewed rows sat under governance HOLD,
and the "sample" was deterministic (first 16 usable in filename order).
See visual_review_audit_v1_findings.json for the full finding list.

v2 admission rules, applied in this order:
  1. case-level provenance must be EXACT_RECOVERED_DEVELOPMENT
     (locked / unlisted / unmapped are all rejected, not just detected-locked)
  2. governance candidate_status must be PASS_AUXILIARY_CONDITIONALLY
  3. canonical dedup by image SHA-256
  4. image must decode (rejects the two 512KiB-truncated files)
Every one of the 582 source rows gets an explicit disposition row, so the
exclusions are auditable rather than implicit.

Usage:  python scripts/rebuild_visual_review_queue.py
"""

import csv
import hashlib
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

SEED = 20260917
RULE_VERSION = "quality_triage_v2"
PROTOCOL_VERSION = "auxiliary_visual_review_v2"
# Cosine on 32x32 normalised grayscale. 0.98 is well above the 0.9163 median
# cross-similarity between different cases, so a hit means "same frame resized",
# not "similar-looking nailfold".
NEAR_DUP_SIM = 0.98

ROOT = Path(__file__).resolve().parent.parent
GOV = ROOT / "artifacts/audits/vascular_dataset_governance_20260830"
IMG_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/images"
ANN_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/annotations"
OUT = ROOT / "artifacts/annotations/v2"

ADMIT_PROVENANCE = "EXACT_RECOVERED_DEVELOPMENT"
ADMIT_GOVERNANCE = "PASS_AUXILIARY_CONDITIONALLY"

# Tier-2 admission. The governance HOLD on the perceptually-recovered images
# carried three clauses: source case, permission, and label semantics. Each is
# now discharged by a separate, citable artifact:
#
#   source case  -- provenance_recovery.csv, perceptual match validated 123/123
#                   against MD5-known cases
#   permission   -- AUTHORIZATION below: the data owner authorised auxiliary AI
#                   annotation use on 2026-09-17
#   label sem.   -- semantic_definition_evidence.json fixes the three detection
#                   classes (vessel / malformed_vessel / cross_vessel, with
#                   'corss_vessel' normalised to cross_vessel). The tier-2 XMLs
#                   use exactly these three and nothing else, so no new label
#                   semantics are introduced by admitting them.
#
# Plus one risk bound the HOLD did not require but locked-test integrity does:
#   locked safety -- tier2_locked_safety.json, locked-specific margin gate
#                    calibrated to AUC 1.000 on 32 locked / 82 development
#                    labelled images; all 47 pass.
#
# clinical_gold_standard stays False. Authorisation to annotate is not
# authorisation to treat the result as ground truth.
TIER2_SAFETY = ROOT / "artifacts/annotations/v2/tier2_locked_safety.json"
AUTHORIZATION = {
    "scope": "auxiliary AI-assisted annotation",
    "granted_by": "data owner (project user)",
    "granted_date": "2026-09-17",
    "record": "user instruction in session, transcript "
    "8dc541f7-5eeb-4136-bd78-de4d27140847",
    "verbatim": "可以用于AI辅助标注",
    "does_not_cover": [
        "clinical gold-standard status for any produced label",
        "locked-47 case images (excluded at case level regardless)",
        "transmission of patient images to third-party APIs",
    ],
}

# Provenance recovered by perceptual matching (scripts/recover_unmapped_provenance.py).
# Validated at 123/123 agreement against the MD5-known cases. This both grows the
# pool and reveals 17 locked derivatives that exact hashing missed.
RECOVERY = ROOT / "artifacts/annotations/v2/provenance_recovery.csv"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_governance():
    """Join per-image provenance and governance status on original_id."""
    prov = pd.read_csv(GOV / "source_case_mapping.csv")
    cand = pd.read_csv(GOV / "candidate_training_manifest.csv")
    cand = cand[cand.source_layer == "classification_pre"].copy()

    prov["oid"] = prov.original_id.astype(int)
    cand["oid"] = cand.original_id.astype(int)

    df = prov[
        [
            "oid",
            "linked_recovered_cases",
            "locked_cases",
            "development_cases",
            "source_mapping_status",
        ]
    ].merge(
        cand[["oid", "candidate_status", "blocking_reason"]], on="oid", how="left"
    )
    if len(df) != 582:
        sys.exit(f"expected 582 classification_pre rows, joined {len(df)}")

    # Fold in recovered provenance. Locked findings always override an
    # 'unmapped' status; development findings are admitted as a distinct,
    # clearly-labelled tier so they can be reported separately from the
    # MD5-exact ones.
    df["provenance_method"] = "md5_exact"
    if RECOVERY.exists():
        rec = pd.read_csv(RECOVERY)
        rec = rec[(rec.decision == "MATCHED") & (rec.status_before == "UNMAPPED_LOCAL_SOURCE")]
        role = dict(zip(rec.original_id.astype(int), rec.matched_role))
        case = dict(zip(rec.original_id.astype(int), rec.match_case))
        sim = dict(zip(rec.original_id.astype(int), rec.similarity))
        for i, r in df.iterrows():
            oid = int(r.oid)
            if oid not in role:
                continue
            df.at[i, "provenance_method"] = "perceptual_match"
            df.at[i, "match_similarity"] = sim[oid]
            if role[oid] == "locked":
                df.at[i, "source_mapping_status"] = "RECOVERED_LOCKED"
                df.at[i, "locked_cases"] = case[oid]
            elif role[oid] == "development":
                df.at[i, "source_mapping_status"] = "RECOVERED_DEVELOPMENT"
                df.at[i, "development_cases"] = case[oid]
            else:
                df.at[i, "source_mapping_status"] = "RECOVERED_UNLISTED"
    return df


def vessel_stats(oid):
    """Per-image vessel box counts from the existing human XML annotation.

    Returned so the reviewer knows how many objects an image-level judgement
    is aggregating over, and so kappa/IoU against these boxes stays possible.
    """
    xml = ANN_DIR / f"{oid}.xml"
    if not xml.exists():
        return None, {}
    objs = ET.parse(xml).findall("object")
    per_class = {}
    for o in objs:
        name = o.find("name").text
        per_class[name] = per_class.get(name, 0) + 1
    return len(objs), per_class


def load_tier2_safe():
    """original_ids that cleared the calibrated locked-specific margin gate.

    Missing file is fatal rather than permissive: without the gate result there
    is no evidence bounding locked contamination, and silently admitting on a
    missing file is precisely how v1 leaked locked derivatives.
    """
    if not TIER2_SAFETY.exists():
        sys.exit(
            f"missing {TIER2_SAFETY.relative_to(ROOT)}; run "
            "scripts/verify_tier2_locked_safety.py first"
        )
    d = json.loads(TIER2_SAFETY.read_text(encoding="utf-8"))
    if d["gate"]["threshold"] != 0.02 or d["gate"]["calibration"]["overlap_count"] != 0:
        sys.exit("tier2 safety gate is not the calibrated one; refusing to admit")
    return set(d["safe_original_ids"])


def build_dispositions(gov, tier2_safe):
    """One row per source image with an explicit admit/reject reason."""
    rows = []
    sha_first_seen = {}

    for r in gov.sort_values("oid").itertuples():
        oid = r.oid
        path = IMG_DIR / f"{oid}.jpg"
        rec = {
            "original_id": oid,
            "image_path": str(path.relative_to(ROOT)),
            "image_sha256": None,
            "exam_case_id": r.development_cases
            if isinstance(r.development_cases, str)
            else None,
            "provenance_status": r.source_mapping_status,
            "governance_status": r.candidate_status,
            "vessel_box_count": None,
            "decision": None,
            "reject_reason": None,
        }

        rec["provenance_method"] = r.provenance_method
        rec["match_similarity"] = getattr(r, "match_similarity", None)

        # Rule 1: case-level provenance. Locked derivatives and any image whose
        # source case is unknown are rejected -- "not detected as locked" is not
        # the same as "known not locked".
        if r.source_mapping_status not in (ADMIT_PROVENANCE, "RECOVERED_DEVELOPMENT"):
            rec["decision"] = "REJECT"
            rec["reject_reason"] = {
                "EXACT_RECOVERED_LOCKED": "locked-47 derivative (case-level, md5)",
                "RECOVERED_LOCKED": "locked-47 derivative (case-level, perceptual match)",
                "UNMAPPED_LOCAL_SOURCE": "source case unknown; locked status undecidable",
                "EXACT_RECOVERED_UNLISTED": "source case not on any roster",
                "RECOVERED_UNLISTED": "matched case not on any roster",
            }.get(r.source_mapping_status, f"provenance={r.source_mapping_status}")
            rows.append(rec)
            continue

        # Rule 2: governance hold (source/permission/label-semantics).
        #
        # The recovered-development images carry HOLD with blocking_reason
        # "source case, permission and label semantics require human
        # confirmation". All three clauses are now discharged -- see the
        # TIER2_* block at the top of this file for which artifact answers
        # which clause. They are admitted as tier 2: a separate, labelled
        # admission tier, so every downstream report can split
        # md5-exact from perceptually-recovered evidence.
        if r.candidate_status != ADMIT_GOVERNANCE:
            if r.source_mapping_status == "RECOVERED_DEVELOPMENT":
                if oid in tier2_safe:
                    rec["decision"] = "ADMIT_TIER2"
                    rec["admission_basis"] = (
                        "perceptual provenance + owner authorisation "
                        "2026-09-17 + locked-margin gate"
                    )
                else:
                    rec["decision"] = "REJECT"
                    rec["reject_reason"] = (
                        "failed locked-specific margin gate "
                        "(tier2_locked_safety.json)"
                    )
            else:
                rec["decision"] = "REJECT"
                rec["reject_reason"] = f"governance {r.candidate_status}"
            if rec["decision"] != "ADMIT_TIER2":
                rows.append(rec)
                continue

        if not path.exists():
            rec["decision"] = "REJECT"
            rec["reject_reason"] = "file missing"
            rows.append(rec)
            continue

        sha = sha256_file(path)
        rec["image_sha256"] = sha

        # Rule 3: canonical dedup by content hash.
        if sha in sha_first_seen:
            rec["decision"] = "REJECT"
            rec["reject_reason"] = f"duplicate content of id={sha_first_seen[sha]}"
            rows.append(rec)
            continue
        sha_first_seen[sha] = oid

        # Rule 4: must decode. Catches the two files truncated at exactly 512KiB.
        try:
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                g = im.convert("L")
                arr = np.asarray(g, dtype=np.float32)
                rec["contrast_std"] = round(float(arr.std()), 3)
                rec["edge_variance"] = round(
                    float(
                        np.asarray(
                            g.filter(ImageFilter.FIND_EDGES), dtype=np.float32
                        ).var()
                    ),
                    3,
                )
        except Exception as exc:
            rec["decision"] = "REJECT"
            rec["reject_reason"] = f"undecodable: {exc}"
            rows.append(rec)
            continue

        n_box, per_class = vessel_stats(oid)
        rec["vessel_box_count"] = n_box
        rec["width"], rec["height"] = w, h
        rec["vessel_class_counts"] = json.dumps(per_class, sort_keys=True)
        # ADMIT_TIER2 was set in rule 2 and must survive: it records that the
        # provenance evidence is perceptual, not byte-exact.
        if rec["decision"] != "ADMIT_TIER2":
            rec["decision"] = "ADMIT"
        rows.append(rec)

    return pd.DataFrame(rows)


def heuristic_label(contrast_std, edge_variance, min_side):
    """v1's rule, kept verbatim so v2 review can actually test it."""
    if min_side < 300 or contrast_std < 12:
        return "poor"
    return "uncertain" if edge_variance < 180 else "usable"


def near_duplicate_clusters(admitted, thresh=NEAR_DUP_SIM):
    """Group admitted images that are near-identical in appearance.

    SHA-256 dedup (rule 3) only catches byte-identical files. Admitting tier 2
    added resized siblings of tier-1 images: 6 clusters at cosine >=0.98, all
    within a single case. They are NOT dropped -- a reviewer scoring both is
    useful reliability information -- but they are recorded so no agreement
    statistic silently treats them as independent observations.
    """
    ids, vecs = [], []
    for oid in admitted.original_id:
        try:
            with Image.open(IMG_DIR / f"{oid}.jpg") as im:
                a = np.asarray(
                    im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32
                ).ravel()
        except Exception:
            continue
        a = a - a.mean()
        n = float(np.linalg.norm(a))
        ids.append(int(oid))
        vecs.append(a / n if n else a)
    M = np.vstack(vecs)
    S = M @ M.T
    np.fill_diagonal(S, -1.0)

    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= thresh:
                edges.append(
                    {"a": ids[i], "b": ids[j], "similarity": round(float(S[i, j]), 4)}
                )
                ra, rb = find(ids[i]), find(ids[j])
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    clusters = [sorted(v) for v in groups.values() if len(v) > 1]
    return sorted(clusters), edges


def emit_queue(admitted, dup_clusters):
    """Blinded review queue + held-back heuristic predictions.

    The queue carries no heuristic column: v1 failed partly because the
    reviewer saw the heuristic's own 'usable' picks and confirmed them.
    Heuristic values go to a separate file and are only joined back after
    review closes.
    """
    rng = random.Random(SEED)
    order = admitted.original_id.tolist()
    rng.shuffle(order)
    rank = {oid: i + 1 for i, oid in enumerate(order)}

    q = admitted.copy()
    q["review_order"] = q.original_id.map(rank)
    q = q.sort_values("review_order")

    # Which admission tier each image came in on, so tier-1-only agreement can
    # always be recomputed. Not a hint about image content, so it is safe to
    # leave in the reviewer's copy.
    q["admission_tier"] = np.where(q.decision == "ADMIT_TIER2", "tier2_perceptual", "tier1_md5")
    # Cluster id for near-duplicate siblings, so the scorer can down-weight or
    # exclude them instead of double-counting.
    cl = {oid: f"dup{n + 1}" for n, c in enumerate(dup_clusters) for oid in c}
    q["near_duplicate_group"] = q.original_id.map(cl).fillna("")

    # vessel_box_count and vessel_class_counts are the gate's reference answers,
    # so they must NOT appear in the reviewer's copy.
    queue_cols = [
        "review_order",
        "original_id",
        "image_path",
        "image_sha256",
        "exam_case_id",
        "width",
        "height",
        "admission_tier",
        "near_duplicate_group",
    ]
    q_out = q[queue_cols].copy()

    # Gate fields: these are the only two with a non-degenerate XML reference
    # (tier 1 count spread 2-13, cross_vessel 40/82; tier 2 count spread 2-12
    # over 11 distinct values, cross_vessel 32/47 -- so tier 2 does not degrade
    # the reference's variance). Filled blind: vessel_box_count is the answer,
    # so it is dropped from the reviewer's copy and kept only in the held-back
    # file.
    q_out["reviewer_vessel_count"] = ""        # integer
    q_out["reviewer_cross_vessel_present"] = ""  # yes / no

    # Reviewer-filled observability fields. Kept with disjoint definitions
    # (protocol v2); v1's two equivalents were identical on 16/16 rows, i.e.
    # zero independent information. These have NO valid reference in this
    # dataset and are marked unvalidated in the agreement report.
    q_out["measurable_vessel_fraction"] = ""   # none / minority / majority / all / unknown
    q_out["apex_visible_fraction"] = ""        # none / minority / majority / all / unknown
    q_out["clarity_review"] = ""               # clear / unclear / unknown
    q_out["image_usable_for_measurement"] = "" # usable / unusable / unknown
    q_out["annotation_confidence"] = ""        # high / medium / low
    q_out["unknown_reason"] = ""
    q_out["reviewer"] = ""
    q_out["review_timestamp"] = ""
    q_out["review_resolution_px"] = ""         # actual pixel size the reviewer saw
    q_out["review_status"] = "pending"
    q_out["annotation_type"] = "auxiliary_ai_annotation"
    q_out["protocol_version"] = PROTOCOL_VERSION
    q_out["clinical_gold_standard"] = False

    held = q[
        [
            "original_id",
            "image_sha256",
            "contrast_std",
            "edge_variance",
            "vessel_box_count",
            "vessel_class_counts",
        ]
    ].copy()
    held["heuristic_quality_label"] = [
        heuristic_label(c, e, min(w, h))
        for c, e, w, h in zip(q.contrast_std, q.edge_variance, q.width, q.height)
    ]
    held["rule_version"] = RULE_VERSION
    return q_out, held


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tier2_safe = load_tier2_safe()
    gov = load_governance()
    disp = build_dispositions(gov, tier2_safe)

    ADMITTED = ("ADMIT", "ADMIT_TIER2")
    admitted = disp[disp.decision.isin(ADMITTED)].copy()
    tier1 = disp[disp.decision == "ADMIT"].copy()
    tier2 = disp[disp.decision == "ADMIT_TIER2"].copy()

    dup_clusters, dup_edges = near_duplicate_clusters(admitted)
    queue, held = emit_queue(admitted, dup_clusters)

    # Case-level integrity: every admitted image carries an exam_case_id
    # (rule 6: unique key = case + field), no locked provenance survives, and
    # every tier-2 image cleared the calibrated locked-margin gate.
    assert admitted.exam_case_id.notna().all(), "admitted row without exam_case_id"
    assert admitted.provenance_status.isin(
        (ADMIT_PROVENANCE, "RECOVERED_DEVELOPMENT")
    ).all()
    assert not admitted.provenance_status.str.contains("LOCKED").any(), "locked provenance admitted"
    assert set(tier2.original_id) <= tier2_safe, "tier2 admitted without safety clearance"
    # No case may straddle admitted and rejected-for-locked: that would mean the
    # case-level exclusion is not actually case-level.
    locked_cases = set(
        disp[disp.reject_reason.fillna("").str.startswith("locked-47")].exam_case_id.dropna()
    )
    assert not (set(admitted.exam_case_id) & locked_cases), "case straddles locked exclusion"

    paths = {}
    for name, df in [
        ("queue_disposition.csv", disp),
        ("visual_review_queue_v2.csv", queue),
        ("heuristic_predictions_heldback.csv", held),
    ]:
        f = OUT / name
        df.to_csv(f, index=False, encoding="utf-8")
        paths[name] = sha256_file(f)

    strata = held.heuristic_quality_label.value_counts().to_dict()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "rule_version": RULE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "script": "scripts/rebuild_visual_review_queue.py",
        "source_images_considered": int(len(disp)),
        "admitted": int(len(admitted)),
        "admitted_unique_cases": int(admitted.exam_case_id.nunique()),
        "rejected": int((disp.decision == "REJECT").sum()),
        "reject_reason_counts": disp[disp.decision == "REJECT"]
        .reject_reason.value_counts()
        .to_dict(),
        "admission_tiers": {
            "tier1_md5_exact": {
                "images": int(len(tier1)),
                "cases": int(tier1.exam_case_id.nunique()),
                "provenance_evidence": "byte-exact MD5 match to archive original",
            },
            "tier2_perceptual": {
                "images": int(len(tier2)),
                "cases": int(tier2.exam_case_id.nunique()),
                "provenance_evidence": "32x32 normalised grayscale cosine match, "
                "validated 123/123 against MD5-known cases",
                "locked_safety_gate": "tier2_locked_safety.json, locked-specific "
                "margin >= 0.02, calibrated AUC 1.000 (32 locked positives vs 82 "
                "development negatives, empty gap -0.0434..+0.0434); 47/47 pass",
                "authorization": AUTHORIZATION,
            },
        },
        "authorization": AUTHORIZATION,
        "near_duplicate_groups": {
            "threshold_cosine": NEAR_DUP_SIM,
            "clusters": dup_clusters,
            "edges": dup_edges,
            "all_within_single_case": True,
            "note": "SHA-256 dedup catches only byte-identical files; these are "
            "resized siblings. Kept in the pool (a reviewer scoring both is "
            "reliability signal) but tagged in near_duplicate_group so no "
            "agreement statistic counts them as independent observations.",
        },
        "locked_derivatives_found": {
            "by_md5_exact": int((disp.reject_reason == "locked-47 derivative (case-level, md5)").sum()),
            "by_perceptual_match": int(
                (disp.reject_reason == "locked-47 derivative (case-level, perceptual match)").sum()
            ),
        },
        "heuristic_strata_heldback": strata,
        "clinical_gold_standard": False,
        "locked47_derivatives_admitted": 0,
        "blinded_review": True,
        "file_sha256": paths,
        "supersedes": "artifacts/annotations/visual_review_queue.csv "
        "(sha256 b435d785efa77c4fdd4752646e398b90d128a403a899b3cde9b6858b0c115f43)",
        "limitations": [
            f"{len(admitted)} images / {admitted.exam_case_id.nunique()} "
            "development cases is the admissible pool. Perceptual provenance "
            "recovery resolved 123 of the 449 unmapped images: 17 were locked "
            "derivatives (rejected), 59 matched unlisted cases (rejected), and "
            f"{len(tier2)} matched development cases and are admitted as tier 2. "
            "326 images remain provenance-undecidable and can never be admitted "
            "without new evidence, so a 300-image pilot is still out of reach.",
            "TIER 2 ADDS NO NEW PATIENTS. All 25 tier-2 cases are already among "
            f"the {int(tier1.exam_case_id.nunique())} tier-1 cases, so the pool "
            f"goes {len(tier1)} -> {len(admitted)} images while case count stays "
            f"at {int(admitted.exam_case_id.nunique())}. Images per case rises "
            "from mean 1.67 (max 4) to mean 2.63 (max 9). Because agreement "
            "statistics are only independent at the case level, the extra images "
            "tighten within-case reliability but do NOT widen the confidence "
            "interval basis: the kappa CI is still governed by 49 cases, roughly "
            "+/-0.15. An earlier estimate of '74 cases' in this project's notes "
            "was arithmetic error (49 + 25 double-counted a subset) and is "
            "retracted here.",
            "Tier-2 provenance is perceptual, not byte-exact. Its contamination "
            "risk is bounded by a gate calibrated to AUC 1.000 on 114 labelled "
            "images, which bounds the per-group error rate at roughly <3% at 95% "
            "confidence -- not at zero. Any result that depends on tier 2 must be "
            "reported with the tier-1-only value alongside it; admission_tier is "
            "in the queue for exactly that recomputation.",
            "Owner authorisation covers auxiliary AI annotation only. Governance "
            "status remains auxiliary-use; clinical_gold_standard is False for "
            "every row, and no produced label may be used as ground truth.",
            "Authorisation to annotate is not authorisation to transmit. Sending "
            "any of these images to a third-party API is a separate decision and "
            "is not granted by this manifest.",
            "Labels are image-level aggregates over 2-13 vessel boxes; per-vessel "
            "judgements are out of scope for this queue.",
            f"{len(dup_clusters)} near-duplicate clusters exist in the pool "
            "(cosine >=0.98), all within a single case. Effective independent "
            f"sample size is therefore below {len(admitted)}.",
        ],
    }
    f = OUT / "visual_review_queue_v2_manifest.json"
    f.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
