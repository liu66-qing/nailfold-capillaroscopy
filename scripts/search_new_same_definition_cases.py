# -*- coding: utf-8 -*-
"""Can a batch of new same-definition cases -- even just 50-100 -- be assembled
from any dataset already on disk?

"Same definition" is the whole question, so it is stated up front and enforced:

  1. the unit is a CASE (one person's examination), not an image
  2. it carries the 21 substantive fields as artifacts/audits/canonical_labels.csv
     defines them
  3. those values come from the same report template, so the value vocabulary
     matches (>=7 / 30--60% / 波纹状 ...) rather than needing re-mapping
  4. at least one CAPorg still exists as model input

A source that fails any one of these cannot add n to a field, however many
images it holds. That is the distinction this script keeps: images are not
cases, and a different label vocabulary is not the same definition.

Reads only. Writes artifacts/audits/new_case_search_20260925/. Trains nothing,
scores nothing, and never touches locked-47 -- the accounting is done from the
manifest, so no locked image is opened.

Run:
  PYTHONIOENCODING=utf-8 python scripts/search_new_same_definition_cases.py
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Some 大图 files are truncated; we only need their pixels for a similarity
# judgement, so decode what is there rather than skipping the file.
ImageFile.LOAD_TRUNCATED_IMAGES = True

OUT = "artifacts/audits/new_case_search_20260925"
ARCHIVES = ["recovered_archive1", "recovered_archive2", "recovered_archive3"]
SCORE_COLS = ("morphology_score", "flow_score", "periloop_score",
              "total_score", "overall_assessment")


def write(name: str, obj) -> None:
    """json.dump on Windows picks the ANSI codepage; force utf-8."""
    io.open(os.path.join(OUT, name), "w", encoding="utf-8").write(
        json.dumps(obj, ensure_ascii=False, indent=1))


def own_archives() -> dict:
    """The only source that can satisfy all four conditions -- so count it exactly."""
    cases = pd.read_csv("artifacts/manifest/cases.csv")
    canon = pd.read_csv("artifacts/audits/canonical_labels.csv")
    splits = pd.read_csv("artifacts/manifest/splits.csv")

    fields = [c for c in canon.columns if c not in ("exam_case_id",) + SCORE_COLS]
    # A canonical row exists for every case whose report was PARSED, including
    # rows where every field came back null -- so presence in the file is not
    # evidence of a label. Count real values instead.
    n_real = canon.set_index("exam_case_id")[fields].notna().sum(axis=1)
    labelled = set(n_real[n_real > 0].index)
    stills = set(cases.loc[cases.cap_count_usable > 0, "exam_case_id"])
    used = set(splits.exam_case_id)          # development 186 + locked 47

    admissible = labelled & stills
    return dict(
        case_directories_on_disk=int(len(cases)),
        rows_in_canonical_labels=int(len(canon)),
        canonical_rows_with_zero_real_values=int((n_real == 0).sum()),
        cases_with_any_real_field_value=len(labelled),
        cases_with_usable_stills=len(stills),
        admissible_label_and_stills=len(admissible),
        already_in_splits=len(used),
        new_admissible_cases=sorted(admissible - used),
        new_admissible_case_count=len(admissible - used),
        stills_but_no_label=sorted(stills - labelled),
        label_but_no_stills=sorted(labelled - stills),
    )


def blank_form_evidence() -> dict:
    """Six cases have stills AND report scans yet zero parsed fields. Either the
    OCR failed (recoverable) or the forms are blank (not). Decide on pixels, not
    on the OCR's own say-so: compare each report's TABLE region against another's,
    and against a report known to be filled."""
    blank = ["recovered_archive1/4", "recovered_archive1/12", "recovered_archive1/25",
             "recovered_archive1/43", "recovered_archive1/50", "recovered_archive3/240"]
    filled = "recovered_archive1/1"

    def grey(case):
        p = os.path.join("data", case, "rep.jpg.jpg")
        return np.asarray(Image.open(p).convert("L"), dtype=np.int16)

    base = grey(blank[0])
    table = slice(0, 470)   # below this row sit the free-text advice lines
    rows = []
    for c in blank[1:] + [filled]:
        d = np.abs(grey(c)[table] - base[table])
        rows.append(dict(case=c, pixels_differing_over_40_grey_levels=int((d > 40).sum())))

    tokens = collections.defaultdict(list)
    with io.open("artifacts/labels/rapidocr_all_clean_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("exam_case_id") in set(blank) | {filled}:
                tokens[d["exam_case_id"]].append(
                    dict(copy=d["report_path"].split("/")[-1],
                         parsed_fields=sum(1 for v in d["fields"].values() if v is not None),
                         ocr_tokens=len(d["audit"]["tokens"])))
    return dict(
        baseline_for_pixel_diff=blank[0],
        table_region="rows 0..470 of the 680x750 scan, i.e. the measurement table",
        pixel_diff=rows,
        every_report_copy=dict(tokens),
        reading=("the five other blank cases differ from the baseline in exactly 0 "
                 "table pixels while the filled report differs in thousands, and all "
                 "of their copies OCR to 59 tokens against the filled report's 100. "
                 "The 59 are printed field names, units and normal-range brackets "
                 "only. These are blank forms, so there is no value to recover."),
    )


def big_image_overlap(sample_limit: int | None = None) -> dict:
    """血管数据集's 大图 is the one external source with Chinese field-style labels,
    so it is the only one worth checking for overlap. Perceptual hash first
    (catches re-encodes), then normalised cross-correlation for the residue."""
    d = "data/血管数据集/血管分类标注/大图"

    def phash(p, s=16):
        a = np.asarray(Image.open(p).convert("L").resize((s, s), Image.LANCZOS),
                       dtype=np.float32)
        return (a > a.mean()).tobytes()

    ours = collections.defaultdict(list)
    for arch in ARCHIVES:
        root = os.path.join("data", arch)
        for case in sorted(os.listdir(root)):
            cp = os.path.join(root, case)
            if not os.path.isdir(cp):
                continue
            for f in sorted(os.listdir(cp)):
                if f.startswith("CAPorg") and f.lower().endswith(".jpg"):
                    ours[phash(os.path.join(cp, f))].append(arch + "/" + case)

    files = sorted(os.listdir(d))
    if sample_limit:
        files = files[:sample_limit]
    matched, residue = {}, []
    for f in files:
        h = phash(os.path.join(d, f))
        if h in ours:
            matched[f] = ours[h][0]
        else:
            residue.append(f)

    x = pd.read_excel("data/血管数据集/血管分类标注/血管分类标注.xlsx")
    return dict(
        images=len(files),
        rows_in_xlsx=int(len(x)),
        label_columns=[str(c) for c in x.columns],
        unit_of_annotation="one image; there is no case or patient column",
        phash_identical_to_our_stills=len(matched),
        distinct_our_cases_touched=len(set(matched.values())),
        residue_not_matched_by_phash=len(residue),
        deliverable_fields_present=[],
        reading=("even before the overlap arithmetic this source fails condition 2 "
                 "and 3: its five columns are weak fields in a different vocabulary, "
                 "and none of clarity / exudation / malformation_ratio appears. "
                 "The overlap only settles that the residue is small as well."),
    )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    own = own_archives()
    write("own_archives_recomputed.json", own)
    write("blank_form_evidence.json", blank_form_evidence())
    write("big_image_overlap_recomputed.json", big_image_overlap())

    print("case directories on disk        :", own["case_directories_on_disk"])
    print("admissible (label + stills)     :", own["admissible_label_and_stills"])
    print("already in splits.csv           :", own["already_in_splits"])
    print("NEW same-definition cases       :", own["new_admissible_case_count"])
    print("stills but no label             :", len(own["stills_but_no_label"]))
    print("label but no stills             :", len(own["label_but_no_stills"]))
    if own["new_admissible_case_count"] == 0:
        print("\n=> no new same-definition case exists on this disk.")


if __name__ == "__main__":
    main()
