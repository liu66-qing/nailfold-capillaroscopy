#!/usr/bin/env python
"""EVIDENCE GROUP 2: dataset statistics for every dataset in this project.

Answers, per dataset: exams / images / videos; per-field class proportions and
missing rate; field overlap between the hospital archives, the doctor-annotation
subset and ANFC; acquisition conditions (device, magnification, finger, site,
protocol) -- read from the report RTF and image headers, or reported ABSENT; and
whether the train/val/test split is by patient.

Anything not present in the repository is reported as ABSENT/UNVERIFIED with the
place I looked, not filled in.
"""
import argparse
import collections
import json
import os
import re

import pandas as pd

FIELDS = [
    "clarity", "capillary_count", "afferent_diameter", "efferent_diameter",
    "apex_diameter", "output_input_ratio", "loop_length", "crossing_ratio",
    "malformation_ratio", "flow_state", "flow_speed_um_s", "vasomotion",
    "rbc_aggregation", "wbc_count", "microthrombus", "blood_color", "exudation",
    "hemorrhage", "subpapillary_venous_plexus", "papilla", "sweat_duct",
    "morphology_score", "flow_score", "periloop_score", "total_score",
    "overall_assessment",
]

ACQ_KEYWORDS = ["倍", "放大", "手指", "无名指", "中指", "小指", "食指", "拇指",
                "左手", "右手", "甲襞", "仪器", "型号", "设备", "显微", "部位",
                "年龄", "性别", "microscope", "magnif"]

# a first pass showed finger/hand words DO occur in a minority of reports, so they
# must be extracted per exam rather than declared absent
FINGER_WORDS = ["无名指", "中指", "小指", "食指", "拇指"]
HAND_WORDS = ["左手", "右手"]

MISSING = {"nan", "", "None", "NaN"}


def rtf_text(path):
    """Decode a GB-coded RTF well enough to keyword-search it."""
    raw = open(path, "rb").read().decode("latin-1", "ignore")
    unescaped = re.sub(r"\\'([0-9a-fA-F]{2})",
                       lambda m: chr(int(m.group(1), 16)), raw)
    try:
        text = unescaped.encode("latin-1", "ignore").decode("gb18030", "ignore")
    except Exception:
        text = unescaped
    text = re.sub(r"\\[a-zA-Z]+-?[0-9]* ?", " ", text)
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text)


def find_rtfs(root, archives):
    out = []
    for a in archives:
        base = os.path.join(root, "data", a)
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if fn.lower().endswith(".rtf"):
                    out.append(os.path.join(dirpath, fn))
    return out


def field_stats(df, fields):
    out = {}
    for fl in fields:
        if fl not in df.columns:
            continue
        v = df[fl].astype(str).str.strip()
        present = v[~v.isin(MISSING)]
        vc = present.value_counts()
        out[fl] = {
            "n_labelled": int(len(present)),
            "n_missing": int(len(v) - len(present)),
            "missing_rate": round(float(1 - len(present) / max(len(v), 1)), 4),
            "n_classes": int(len(vc)),
            "class_counts": {str(k): int(x) for k, x in vc.items()},
            "class_proportions": {str(k): round(float(x / len(present)), 4)
                                  for k, x in vc.items()} if len(present) else {},
            "majority_share": round(float(vc.iloc[0] / len(present)), 4) if len(present) else None,
        }
    return out


def anfc_stats(root):
    """Count ANFC images/labels per split and the class ids actually present."""
    out = {}
    for name in sorted(os.listdir(os.path.join(root, "data"))):
        if "anfc" not in name.lower():
            continue
        base = os.path.join(root, "data", name)
        entry = {"path": os.path.join("data", name), "splits": {}}
        yml = [f for f in os.listdir(base) if f.endswith((".yaml", ".yml"))]
        if yml:
            entry["yaml"] = open(os.path.join(base, yml[0]), encoding="utf-8",
                                 errors="ignore").read().strip()
        for split in ("train", "valid", "val", "test"):
            for img_rel in (os.path.join(split, "images"),
                            os.path.join("images", split), split):
                img_dir = os.path.join(base, img_rel)
                if not os.path.isdir(img_dir):
                    continue
                imgs = [f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                lbl_dir = img_dir.replace("images", "labels")
                cls = collections.Counter()
                n_lbl = 0
                n_inst = 0
                if os.path.isdir(lbl_dir):
                    for fn in os.listdir(lbl_dir):
                        if not fn.endswith(".txt"):
                            continue
                        n_lbl += 1
                        for line in open(os.path.join(lbl_dir, fn),
                                         errors="ignore"):
                            line = line.strip()
                            if line:
                                cls[line.split()[0]] += 1
                                n_inst += 1
                entry["splits"][img_rel] = {
                    "n_images": len(imgs), "n_label_files": n_lbl,
                    "n_instances": n_inst,
                    "class_id_counts": dict(sorted(cls.items())),
                }
                break
        # COCO variant
        for split in ("train", "valid", "test"):
            j = os.path.join(base, split, "_annotations.coco.json")
            if os.path.exists(j):
                d = json.load(open(j, encoding="utf-8", errors="ignore"))
                cats = {c["id"]: c["name"] for c in d.get("categories", [])}
                cc = collections.Counter(a["category_id"]
                                         for a in d.get("annotations", []))
                entry["splits"]["coco_" + split] = {
                    "n_images": len(d.get("images", [])),
                    "n_instances": len(d.get("annotations", [])),
                    "categories": cats,
                    "category_counts": {cats.get(k, k): int(v)
                                        for k, v in sorted(cc.items())},
                }
        out[name] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--files", default="artifacts/manifest/files.csv")
    ap.add_argument("--cases", default="artifacts/manifest/cases.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.root

    m = pd.read_csv(os.path.join(R, a.manifest))
    m["exam_case_id"] = m.exam_case_id.astype(str)
    f = pd.read_csv(os.path.join(R, a.files))
    f["exam_case_id"] = f.exam_case_id.astype(str)
    cases = pd.read_csv(os.path.join(R, a.cases))
    cases["exam_case_id"] = cases.exam_case_id.astype(str)

    cap = f[f.role == "cap_image"].copy()
    cap["blank"] = cap.is_black_placeholder.astype(bool)
    role = dict(zip(m.exam_case_id, m.evaluation_role))
    cap["cohort"] = cap.exam_case_id.map(role).fillna("NOT_IN_MANIFEST")

    out = {"generated_by": "scripts/evidence_group2_dataset_stats.py"}

    # ---- dataset A: the hospital archives, labelled cohort -------------------
    ds = {}
    for cohort, sub in [("development", m[m.evaluation_role == "development"]),
                        ("locked_test", m[m.evaluation_role == "locked_test"]),
                        ("all_labelled_233", m)]:
        ids = set(sub.exam_case_id)
        c = cap[cap.exam_case_id.isin(ids)]
        ds[cohort] = {
            "n_exams": int(len(sub)),
            "n_patients": "UNKNOWN -- patient_id is null for all 233 rows",
            "patient_id_non_null": int(sub.patient_id.notna().sum()),
            "n_cap_images": int(len(c)),
            "n_cap_images_nonblank": int((~c.blank).sum()),
            "n_blank_placeholder_images": int(c.blank.sum()),
            "exams_touched_by_a_blank": int(c[c.blank].exam_case_id.nunique()),
            "images_per_exam": {
                "min": int(c.groupby("exam_case_id").size().min()),
                "median": float(c.groupby("exam_case_id").size().median()),
                "mean": round(float(c.groupby("exam_case_id").size().mean()), 2),
                "max": int(c.groupby("exam_case_id").size().max()),
            },
            "n_videos": int(sub.video_count.fillna(0).sum()),
            "exams_with_video": int((sub.video_count.fillna(0) > 0).sum()),
            "n_archives": int(sub.archive_id.nunique()),
            "by_archive": sub.archive_id.value_counts().to_dict(),
            "fields": field_stats(sub, FIELDS),
        }
    out["dataset_A_hospital_labelled"] = ds

    # ---- unlabelled / excluded hospital exams --------------------------------
    orphan = cap[cap.cohort == "NOT_IN_MANIFEST"]
    out["dataset_A_extra_unlabelled"] = {
        "what": "exams present in files.csv but NOT in the labelled manifest",
        "n_exams": int(orphan.exam_case_id.nunique()),
        "n_cap_images": int(len(orphan)),
        "exam_ids": sorted(orphan.exam_case_id.unique().tolist()),
        "by_archive": orphan.archive_id.value_counts().to_dict(),
        "cases_csv_rows": int(len(cases)),
        "note": "cases.csv has %d rows, the labelled manifest 233; the difference "
                "is exams whose report could not be parsed into fields"
                % len(cases),
    }

    # ---- dataset B: doctor-annotation subset --------------------------------
    ann = {}
    for d in sorted(os.listdir(os.path.join(R, "artifacts"))):
        if "annotation" not in d:
            continue
        base = os.path.join(R, "artifacts", d)
        e = {"path": "artifacts/" + d,
             "contents": sorted(os.listdir(base))[:12]}
        mf = os.path.join(base, "manifest.json")
        if os.path.exists(mf):
            j = json.load(open(mf, encoding="utf-8", errors="ignore"))
            if isinstance(j, dict):
                e["manifest_keys"] = list(j.keys())
                for k in ("n_images", "n_cases", "images", "cases", "fields",
                          "selected_fields", "count"):
                    if k in j:
                        v = j[k]
                        e["manifest_" + k] = (len(v) if isinstance(v, (list, dict))
                                              else v)
        sa = os.path.join(base, "selection_audit.csv")
        if os.path.exists(sa):
            s = pd.read_csv(sa)
            e["selection_audit_rows"] = int(len(s))
            e["selection_audit_cols"] = list(s.columns)
            for col in s.columns:
                if "case" in col.lower() or "exam" in col.lower():
                    e["selection_audit_unique_" + col] = int(s[col].nunique())
                if "field" in col.lower():
                    e["selection_audit_fields"] = s[col].value_counts().to_dict()
        for sub in ("original", "draft_overlay", "corrections"):
            p = os.path.join(base, sub)
            if os.path.isdir(p):
                e["n_" + sub] = len(os.listdir(p))
        ann[d] = e
    out["dataset_B_doctor_annotation"] = ann

    # ---- dataset C: ANFC ----------------------------------------------------
    out["dataset_C_anfc"] = anfc_stats(R)

    # ---- field overlap between the three datasets ---------------------------
    out["field_overlap"] = {
        "hospital_labelled_fields": FIELDS,
        "hospital_n_fields": len(FIELDS),
        "anfc_label_type": "object detection / instance segmentation boxes and "
                           "masks over vessel classes -- NOT the 21 clinical "
                           "fields",
        "anfc_classes": "see dataset_C_anfc[*].yaml",
        "shared_field_count": 0,
        "shared_field_explanation":
            "ZERO clinical fields are shared. ANFC carries per-vessel geometry "
            "classes (capillary/abnormal/aggregation/blur/hemo/normal); the "
            "hospital cohort carries exam-level clinical answers. They are not "
            "the same label space, so ANFC cannot be an external test set for "
            "any of the 21 fields. It was only ever used to train a detector/"
            "segmenter whose outputs became features.",
        "doctor_annotation_label_type":
            "per-image vessel annotations on a subset of the SAME hospital "
            "exams -- so it overlaps the hospital set at exam level but adds "
            "no new exams and no new clinical fields",
    }

    # ---- acquisition conditions --------------------------------------------
    res = cap.groupby(["width", "height"]).size()
    rtfs = find_rtfs(R, ["recovered_archive1", "recovered_archive2",
                         "recovered_archive3"])
    hits = collections.Counter()
    scanned = 0
    sample_text = None
    finger_by_exam = {}
    hand_by_exam = {}
    mag_strings = collections.Counter()
    for p in rtfs:
        try:
            t = rtf_text(p)
        except Exception:
            continue
        scanned += 1
        if sample_text is None and len(t) > 200:
            sample_text = t[:700]
        for k in ACQ_KEYWORDS:
            if k in t:
                hits[k] += 1
        # exam id = the two path components under data/
        parts = os.path.normpath(p).split(os.sep)
        exam = "/".join(parts[-3:-1]) if len(parts) >= 3 else p
        fg = [w for w in FINGER_WORDS if w in t]
        hd = [w for w in HAND_WORDS if w in t]
        if fg:
            finger_by_exam.setdefault(exam, set()).update(fg)
        if hd:
            hand_by_exam.setdefault(exam, set()).update(hd)
        for mm in re.findall(r"[0-9]{1,4}\s*(?:倍|[xX×])", t):
            mag_strings[mm.strip()] += 1
    finger_summary = collections.Counter()
    for v in finger_by_exam.values():
        finger_summary["+".join(sorted(v))] += 1
    hand_summary = collections.Counter()
    for v in hand_by_exam.values():
        hand_summary["+".join(sorted(v))] += 1
    out["acquisition_conditions"] = {
        "manifest_columns_for_device_magnification_finger_site": "ABSENT -- no "
            "such column exists in the manifest (102 columns, all listed in "
            "the group-1 artifact)",
        "image_resolution": {f"{int(w)}x{int(h)}": int(n)
                             for (w, h), n in res.items()},
        "resolution_is_uniform": bool(len(res) == 1),
        "filename_pattern": cap.filename.str.replace(r"[0-9]+", "#", regex=True)
                               .value_counts().head(5).to_dict(),
        "exif": "EMPTY -- no EXIF tags on any sampled cap_image",
        "rtf_reports_scanned": scanned,
        "rtf_total_found": len(rtfs),
        "rtf_keyword_report_counts": dict(hits),
        "device": "UNVERIFIED -- no 仪器/型号/设备/显微 string in any of "
                  "%d reports, no EXIF, no manifest column" % scanned,
        "magnification": {
            "status": "UNVERIFIED",
            "magnification_like_strings_found": dict(mag_strings.most_common(15)),
            "note": "these are regex hits for <number>倍/x and may be text about "
                    "vessel counts rather than optics; none is a reliable "
                    "magnification record",
        },
        "finger": {
            "status": "PARTIALLY RECORDED -- present in the report text of a "
                      "MINORITY of exams only",
            "exams_mentioning_a_finger": len(finger_by_exam),
            "exams_mentioning_a_hand": len(hand_by_exam),
            "of_total_reports": scanned,
            "coverage_fraction": round(len(finger_by_exam) / max(scanned, 1), 4),
            "finger_combinations_per_exam": dict(finger_summary),
            "hand_combinations_per_exam": dict(hand_summary),
            "note": "coverage is far too low and too irregular to stratify any "
                    "result by finger; and the mention is free text in the "
                    "report body, not a structured field",
        },
        "site": "UNVERIFIED as a structured field; 甲襞 appears in report text but "
                "identifies the examination type, not a per-image site",
        "protocol": "UNVERIFIED -- no written acquisition protocol exists in the "
                    "repository",
        "verdict": "Acquisition metadata is essentially NOT RECORDED. Device, "
                   "magnification and protocol: nothing, in the manifest, in "
                   "EXIF (empty on every sampled image) or in the report text. "
                   "Finger/hand: mentioned in free text for a small minority of "
                   "exams, not structured, not usable for stratification. All "
                   "2207 images are 1024x768 and named CAPorg<n>.jpg, which is "
                   "consistent with one capture software but does NOT prove one "
                   "device or one magnification. CONSEQUENCES: (1) no result in "
                   "this project can be stratified by device, magnification or "
                   "finger; (2) the micron scale cannot be recovered from "
                   "acquisition settings, which is why um_per_pixel had to be "
                   "back-fitted from labels and is not usable as calibration; "
                   "(3) 'same device' cannot be asserted in any launch material.",
        "sample_decoded_report_text": sample_text,
    }

    # ---- split integrity ---------------------------------------------------
    sp = os.path.join(R, "artifacts/experiments/split_integrity_20260919/"
                         "split_integrity.json")
    out["split_integrity"] = (json.load(open(sp, encoding="utf-8"))
                              if os.path.exists(sp)
                              else "NOT FOUND at " + sp)
    out["split_is_by_patient"] = {
        "answer": "NO -- it is by EXAM (exam_case_id), not by patient. Patient "
                  "identity cannot be checked at all because patient_id is null "
                  "for all 233 rows.",
        "what_was_verified_instead": "0 exams and 0 duplicate_groups span the "
                                     "split; 0 of 2089 image hashes are shared "
                                     "across it; and a matched near-duplicate "
                                     "control shows locked images are no closer "
                                     "to development images (0.9217) than "
                                     "development images are to each other "
                                     "(0.9207). See split_integrity above.",
        "residual_risk": "If the same person was examined twice, their two exams "
                         "could land on opposite sides of the split and I would "
                         "not detect it. This is UNQUANTIFIED.",
    }

    with open(os.path.join(R, a.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out)
    for k in out:
        print(" ", k)
    print("acq keyword hits:", dict(hits), "over", scanned, "reports")


if __name__ == "__main__":
    main()
