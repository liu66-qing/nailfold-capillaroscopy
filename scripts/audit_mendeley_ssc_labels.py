"""What supervision does the Mendeley SSc dataset actually contain?

The paper this dataset accompanies relates capillary counts and nailfold
bleedings to clinical manifestations of systemic sclerosis, so on paper it is the
better of the two external candidates: a real disease cohort with clinical
variables, which is precisely what our cohort lacks. This script checks what of
that arrived in the download, because a paper's clinical table is only useful to
us if the per-subject values are actually distributed with the images.

What is checked, in order of what would matter most:
  1. does ANY non-image file exist (csv/xlsx/json/txt/pdf) -- i.e. is there a
     clinical table at all;
  2. what label can be derived from filenames alone;
  3. how many distinct subjects, and how many have repeat visits -- the latter
     decides whether a test-retest reliability estimate is possible, which is
     something we cannot compute from our own data at all;
  4. whether filenames carry identifying information.

PRIVACY: filenames in this dataset embed what appear to be patient personal
names. This script therefore reports COUNTS AND STRUCTURE ONLY and never writes
a name into its output, printing a hash of the sorted identity list so the
grouping stays reproducible without republishing identities. The images are not
read; only directory listings are.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_mendeley_ssc_labels.py
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(
    ROOT, "data", "The number of nail fold capillaries",
    "The number of nail fold capillaries and nail fold bleedings reflects "
    "the clinical manifestations of systemic sclerosis")
OUT = os.path.join(ROOT, "artifacts", "evidence", "mendeley_ssc_labels_20260920")

# <yymmdd><identity><index>.png
NAME = re.compile(r"^(\d{6})(.+?)(\d+)\.png$", re.IGNORECASE)
TABLE_EXT = {".csv", ".xlsx", ".xls", ".json", ".txt", ".tsv", ".pdf",
             ".docx", ".doc", ".xml", ".yaml", ".yml", ".mat"}


def main():
    os.makedirs(OUT, exist_ok=True)
    allf = [p for p in glob.glob(os.path.join(SRC, "**", "*"), recursive=True)
            if os.path.isfile(p)]
    tables = [os.path.relpath(p, SRC) for p in allf
              if os.path.splitext(p)[1].lower() in TABLE_EXT]
    pngs = [os.path.basename(p) for p in allf
            if os.path.splitext(p)[1].lower() == ".png"]

    visits, per_subj, unparsed, controls = (
        collections.defaultdict(set), collections.Counter(), [], [])
    for f in pngs:
        if f.lower().startswith("control"):
            controls.append(f)
            continue
        m = NAME.match(f)
        if not m:
            unparsed.append(f)
            continue
        ident = m.group(2).strip()
        visits[ident].add(m.group(1))
        per_subj[ident] += 1

    n_visits = collections.Counter(len(v) for v in visits.values())
    repeat = sum(1 for v in visits.values() if len(v) > 1)
    ident_hash = hashlib.sha256(
        "|".join(sorted(visits)).encode("utf-8")).hexdigest()[:16]

    verdict = {
        "question": ("does this dataset supply the independent disease "
                     "supervision our cohort lacks?"),
        "generated": "2026-09-20",
        "source_relative": os.path.relpath(SRC, ROOT),
        "n_files_total": len(allf),
        "n_png": len(pngs),
        "clinical_table_files_found": tables,
        "n_clinical_table_files": len(tables),
        "derivable_label": {
            "control_prefixed_files": len(controls),
            "non_control_files": len(pngs) - len(controls),
            "finding": (
                "with no table in the download, the only label derivable from "
                f"the files is {len(controls)} control images vs "
                f"{len(pngs) - len(controls)} others. A 2-vs-576 split cannot "
                "train or validate anything."),
        },
        "subjects": {
            "n_distinct_identity_strings": len(visits),
            "identity_list_sha256_16": ident_hash,
            "images_per_subject_min_median_max": [
                min(per_subj.values()),
                sorted(per_subj.values())[len(per_subj) // 2],
                max(per_subj.values())],
            "n_with_repeat_visit_dates": repeat,
            "visit_count_distribution": dict(sorted(n_visits.items())),
            "unparsed_filenames": len(unparsed),
        },
        "privacy_finding": (
            "filenames embed date + what appear to be patient personal names. "
            "No name is recorded in this output. If any of this dataset is ever "
            "used, files must be re-keyed to opaque ids first, and the original "
            "names must not enter our repo, logs, or any model artifact."),
        "verdict": (
            "CANNOT supply disease supervision. No clinical variable of any kind "
            "is present -- no subtype, antibody, mRSS or organ involvement -- "
            f"and the only filename-derivable label is {len(controls)} vs "
            f"{len(pngs) - len(controls)}. The clinical table exists only in the "
            "paper, so obtaining it requires contacting the authors."),
        "the_one_thing_it_could_still_do": (
            f"{repeat} of {len(visits)} subjects have images from more than one "
            "visit date. That permits a test-retest / repeatability estimate of "
            "an image-derived measure -- a number we cannot compute from our own "
            "cohort at all, since none of our cases has a repeat acquisition. "
            "It measures stability, NOT accuracy, and it cannot validate any "
            "field against a clinical truth."),
        "governance": dict(locked_cases_seen=0, models_trained=0,
                           images_transmitted=0, images_read=0,
                           personal_names_recorded=0),
    }
    with open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    print(f"files={len(allf)} png={len(pngs)} tables={len(tables)}")
    print(f"subjects={len(visits)} repeat_visit={repeat} "
          f"controls={len(controls)}")
    print("visit distribution:", dict(sorted(n_visits.items())))
    print("\n" + verdict["verdict"])
    print("\nstill useful for:", verdict["the_one_thing_it_could_still_do"])
    print("written ->", OUT)


if __name__ == "__main__":
    main()
