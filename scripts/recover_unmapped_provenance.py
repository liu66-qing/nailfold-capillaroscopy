"""Recover source-case provenance for the 449 UNMAPPED_LOCAL_SOURCE images.

Why exact MD5 missed them: all 96 already-mapped classification images are
natively 1024x768, while 403 of the 449 unmapped ones have odd ~4:3 sizes
(1001x751, 957x717, ...). They are resizes of archive originals, so byte
hashing cannot match them. The XML <path> fields say "cropfromvoc", which is
consistent with a resize/crop pipeline.

Method: normalised 32x32 grayscale fingerprint, cosine similarity against every
recovered_archive*/*/CAPorg*.jpg. A match is only accepted when it is both
strong in absolute terms and clearly better than the runner-up, so ambiguous
images stay unmapped rather than being guessed into a case.

This decides locked-vs-development status for images the governance record
could not, which is the only way the 82-image pool can grow.

Usage:  python scripts/recover_unmapped_provenance.py
"""

import glob
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GOV = ROOT / "artifacts/audits/vascular_dataset_governance_20260830"
IMG_DIR = ROOT / "data/血管数据集/分类数据集/扩充之前/images"
OUT = ROOT / "artifacts/annotations/v2"

FP_SIZE = 32
# Accept only when the best match is strong AND clearly ahead of the runner-up.
MIN_SIM = 0.90
MIN_MARGIN = 0.05


def fingerprint(path, n=FP_SIZE):
    with Image.open(path) as im:
        g = im.convert("L").resize((n, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32).ravel()
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def archive_case(path):
    """recovered_archiveN/<case> from a path, matching the governance format."""
    parts = Path(path).as_posix().split("/")
    i = next(k for k, p in enumerate(parts) if p.startswith("recovered_archive"))
    return f"{parts[i]}/{parts[i + 1]}"


def build_archive_index():
    files = sorted(glob.glob(str(ROOT / "data/recovered_archive*/*/CAPorg*.jpg")))
    vecs, cases, paths = [], [], []
    for f in files:
        try:
            vecs.append(fingerprint(f))
            cases.append(archive_case(f))
            paths.append(Path(f).relative_to(ROOT).as_posix())
        except Exception:
            continue
    return np.vstack(vecs), cases, paths


def load_roles():
    """Which archive cases are locked, which are development."""
    m = pd.read_csv(GOV / "source_case_mapping.csv")
    locked, dev = set(), set()
    for v in m.locked_cases.dropna():
        locked.update(str(v).split(";"))
    for v in m.development_cases.dropna():
        dev.update(str(v).split(";"))
    return {c.strip() for c in locked}, {c.strip() for c in dev}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(GOV / "source_case_mapping.csv")
    locked_cases, dev_cases = load_roles()

    A, cases, apaths = build_archive_index()
    norm = FP_SIZE * FP_SIZE

    # Validate the method on the 133 images whose case is already known by
    # exact MD5. If it cannot recover those, it cannot be trusted on the rest.
    known = m[m.linked_recovered_cases.notna()]
    rows = []
    for r in m.itertuples():
        p = IMG_DIR / f"{r.original_id}.jpg"
        if not p.exists():
            continue
        try:
            v = fingerprint(p)
        except Exception as exc:
            rows.append(
                {
                    "original_id": r.oid if hasattr(r, "oid") else r.original_id,
                    "status_before": r.source_mapping_status,
                    "decision": "UNDECIDABLE",
                    "reason": f"undecodable: {exc}",
                }
            )
            continue
        sims = (A @ v) / norm
        order = np.argsort(-sims)
        best, second = order[0], order[1]
        best_case = cases[best]
        # Runner-up from a *different* case; same-case duplicates are expected.
        diff = next((j for j in order[1:] if cases[j] != best_case), second)
        margin = float(sims[best] - sims[diff])

        rec = {
            "original_id": int(r.original_id),
            "status_before": r.source_mapping_status,
            "known_case": r.linked_recovered_cases
            if isinstance(r.linked_recovered_cases, str)
            else None,
            "match_case": best_case,
            "match_path": apaths[best],
            "similarity": round(float(sims[best]), 4),
            "runner_up_case": cases[diff],
            "margin": round(margin, 4),
        }
        if sims[best] >= MIN_SIM and margin >= MIN_MARGIN:
            rec["decision"] = "MATCHED"
            rec["matched_role"] = (
                "locked"
                if best_case in locked_cases
                else "development"
                if best_case in dev_cases
                else "unlisted"
            )
        else:
            rec["decision"] = "UNDECIDABLE"
            rec["matched_role"] = None
            rec["reason"] = (
                f"similarity {sims[best]:.3f} < {MIN_SIM}"
                if sims[best] < MIN_SIM
                else f"margin {margin:.3f} < {MIN_MARGIN} vs {cases[diff]}"
            )
        rows.append(rec)

    df = pd.DataFrame(rows)

    # Accuracy on the images whose case is already known.
    val = df[df.known_case.notna() & (df.decision == "MATCHED")]
    agree = int((val.match_case == val.known_case).sum())
    val_n = int(len(val))
    known_total = int(len(known))

    f = OUT / "provenance_recovery.csv"
    df.to_csv(f, index=False, encoding="utf-8")

    unm = df[df.status_before == "UNMAPPED_LOCAL_SOURCE"]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/recover_unmapped_provenance.py",
        "method": f"normalised {FP_SIZE}x{FP_SIZE} grayscale cosine similarity vs "
        f"{len(cases)} recovered_archive CAPorg originals",
        "thresholds": {"min_similarity": MIN_SIM, "min_margin_vs_other_case": MIN_MARGIN},
        "validation_on_md5_known_cases": {
            "md5_known_images": known_total,
            "matched_by_method": val_n,
            "agreed_with_md5_case": agree,
            "agreement_rate": round(agree / val_n, 4) if val_n else None,
        },
        "unmapped_input": int(len(unm)),
        "unmapped_resolved": int((unm.decision == "MATCHED").sum()),
        "unmapped_still_undecidable": int((unm.decision == "UNDECIDABLE").sum()),
        "unmapped_resolved_roles": unm[unm.decision == "MATCHED"]
        .matched_role.value_counts()
        .to_dict(),
        "file_sha256": {
            "provenance_recovery.csv": hashlib.sha256(f.read_bytes()).hexdigest()
        },
    }
    (OUT / "provenance_recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
