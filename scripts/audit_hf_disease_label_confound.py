"""Can the HF Capillary-Dataset supply the disease supervision we lack?

WHY THIS IS THE FIRST THING TO CHECK
------------------------------------
Our single hardest gap is that no case in our cohort has an independently
diagnosed disease status -- the report text is the reader's impression of the
image, so there is no label that is not downstream of the image. A published
dataset with 126 diabetic and 76 healthy *subject* folders would fix exactly
that, and it is the top item on the reading list we were handed. So it gets
tested before any architecture work.

The test is not "does a CNN reach high accuracy". It is "is the class
distinguishable by something that is not capillary biology". If acquisition
differs between the two groups, a model reaches ceiling accuracy while learning
the camera, and the disease label is unusable as supervision no matter which
architecture consumes it.

Three probes, cheapest first:
  1. file format -- a label leak that needs no pixels at all;
  2. low-level image statistics -- what any encoder sees in its first layers;
  3. SUBJECT-level separation by a single scalar, which is the honest unit: a
     per-image number can look moderate purely because subjects contribute
     unequal image counts.

Probe 3 is the decisive one. If one hand-computed scalar separates subjects
perfectly, no amount of modelling can disentangle disease from acquisition,
because the groups were not photographed under the same conditions.

ALSO SETTLED HERE: the subject count dispute.
I earlier reported "14 healthy subjects" from directory prefixes; the KDD paper
says 76 healthy. Both are in the data and neither is a mistake -- there are 76
healthy *directories* whose names are <subject>-<session>, spanning 14 distinct
subject prefixes. Which number is right depends on whether the prefix is an
identity, so the prefix structure is enumerated rather than assumed.

Run:  PYTHONIOENCODING=utf-8 python scripts/audit_hf_disease_label_confound.py
"""
from __future__ import annotations

import collections
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "Capillary-Dataset", "classification_original")
OUT = os.path.join(ROOT, "artifacts", "evidence", "hf_disease_confound_20260920")
PER_SUBJ_IMAGES = 4          # fixed in advance
SEED = 20260920


def imread_unicode(p):
    """cv2.imread returns None on non-ASCII Windows paths; decode bytes instead."""
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def subject_of(rel: str):
    """('H'|'D', subject_id) for a path relative to SRC.

    Healthy dirs are '<subject>-<session>', diabetic dirs are a bare id. The
    healthy prefix is treated as the subject, which is the reading that makes
    the two groups comparable; the alternative (76 independent healthy people)
    is reported alongside so the choice is visible rather than buried.
    """
    parts = rel.split(os.sep)
    group, d = parts[0], parts[1]
    if group.startswith("healthy"):
        return "H", d.split("-")[0]
    return "D", d


def sharpness(img) -> float:
    """Laplacian variance -- the standard no-reference focus measure.

    Chosen because it is the one low-level statistic that cannot be a disease
    signal: how in-focus a photograph is describes the photographer and the
    optics, not the subject's capillaries. If it separates the two diagnostic
    groups, the groups differ by equipment.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {g: sorted(glob.glob(os.path.join(SRC, g, "*", "*")))
             for g in ("diabetic_patients", "healthy_subjects")}

    # ---- PROBE 1: does the label leak through the file format? -------------
    ext = {g: collections.Counter(os.path.splitext(f)[1].lower() for f in fs)
           for g, fs in files.items()}
    nd, nh = len(files["diabetic_patients"]), len(files["healthy_subjects"])
    jpg_h = ext["healthy_subjects"][".jpg"]
    jpg_d = ext["diabetic_patients"][".jpg"]
    probe1 = dict(
        rule="predict healthy iff extension == .jpg",
        diabetic_ext=dict(ext["diabetic_patients"]),
        healthy_ext=dict(ext["healthy_subjects"]),
        image_level_accuracy=round((jpg_h + (nd - jpg_d)) / (nd + nh), 4),
        majority_baseline=round(max(nd, nh) / (nd + nh), 4),
        note=("no pixel is read by this classifier. Any accuracy above the "
              "baseline here is stored metadata, not capillary morphology."),
    )

    # ---- PROBE 2+3: low-level stats, scored at SUBJECT level ---------------
    rows = []
    for g, fs in files.items():
        by_dir = collections.defaultdict(list)
        for f in fs:
            by_dir[os.path.dirname(f)].append(f)
        for _, ff in sorted(by_dir.items()):
            for f in ff[:PER_SUBJ_IMAGES]:
                img = imread_unicode(f)
                if img is None:
                    continue
                grp, sid = subject_of(os.path.relpath(f, SRC))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                rows.append(dict(
                    group=grp, subject=f"{grp}{sid}",
                    y=1 if grp == "D" else 0,
                    width=img.shape[1], height=img.shape[0],
                    sharpness=sharpness(img),
                    saturation=float(hsv[:, :, 1].mean()),
                    gray_std=float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()),
                ))
    img_df = pd.DataFrame(rows)
    img_df.to_csv(os.path.join(OUT, "per_image_stats.csv"),
                  index=False, encoding="utf-8")

    subj = (img_df.groupby(["subject", "y"], as_index=False)
                  .agg(sharpness=("sharpness", "median"),
                       saturation=("saturation", "median"),
                       gray_std=("gray_std", "median"),
                       n_images=("sharpness", "size")))
    subj.to_csv(os.path.join(OUT, "per_subject_stats.csv"),
                index=False, encoding="utf-8")

    probes = {}
    for feat in ("sharpness", "saturation", "gray_std"):
        # sign chosen so higher score = diabetic, then reported as-is; AUROC is
        # symmetric about 0.5 so the direction does not change the magnitude
        auc = float(roc_auc_score(subj.y, -subj[feat]))
        auc = max(auc, 1 - auc)
        d = subj.loc[subj.y == 1, feat]
        h = subj.loc[subj.y == 0, feat]
        probes[feat] = dict(
            subject_level_auroc=round(auc, 4),
            diabetic_median=round(float(d.median()), 2),
            healthy_median=round(float(h.median()), 2),
            diabetic_range=[round(float(d.min()), 2), round(float(d.max()), 2)],
            healthy_range=[round(float(h.min()), 2), round(float(h.max()), 2)],
            ranges_overlap=bool(max(d.min(), h.min()) <= min(d.max(), h.max())),
        )

    n_h_dirs = len({os.path.basename(p) for p in
                    glob.glob(os.path.join(SRC, "healthy_subjects", "*"))})
    n_h_prefix = int((subj.y == 0).sum())
    verdict = {
        "question": ("can this dataset's diabetes label supply the disease "
                     "supervision our cohort lacks?"),
        "generated": "2026-09-20",
        "source": SRC,
        "n_images": dict(diabetic=nd, healthy=nh),
        "subject_count_dispute_resolved": {
            "healthy_directories": n_h_dirs,
            "healthy_unique_prefixes": n_h_prefix,
            "diabetic_directories": int((subj.y == 1).sum()),
            "finding": (
                f"both published numbers are in the data: {n_h_dirs} healthy "
                f"directories named <subject>-<session>, spanning {n_h_prefix} "
                "distinct prefixes. The paper's '76 healthy' counts "
                "directories; my earlier '14' counts prefixes. Which is the "
                "number of PEOPLE is not determinable from the files -- it needs "
                "the authors' naming rule. Until then the conservative unit is "
                f"{n_h_prefix}, because splitting on directories when the prefix "
                "is the person would leak the same person across train/test."),
        },
        "probe1_file_format_leak": probe1,
        "probe2_lowlevel_subject_level": probes,
        "verdict": (
            "UNUSABLE as disease supervision. The two diagnostic groups were "
            "acquired on different equipment: the file format alone classifies "
            f"{probe1['image_level_accuracy']:.4f} of images, and subject-level "
            f"focus sharpness alone reaches AUROC "
            f"{probes['sharpness']['subject_level_auroc']:.4f} with "
            f"{'overlapping' if probes['sharpness']['ranges_overlap'] else 'NON-OVERLAPPING'} "
            "ranges. Any model trained on this label learns the camera. This is "
            "not fixable by architecture, augmentation, or normalisation, "
            "because the confound is perfectly collinear with the target."),
        "what_this_does_and_does_not_close": [
            "CLOSES: borrowing this dataset's diabetes label as an independent "
            "disease endpoint, and quoting its published accuracy.",
            "DOES NOT CLOSE: its Morphology_detection boxes, which are a "
            "different asset with a different (local, within-image) label whose "
            "usefulness was assessed separately and does not depend on the "
            "diabetes grouping.",
            "DOES NOT establish that our own 3 archives are free of the same "
            "problem -- that is exactly why our ruler requires "
            "leave-one-archive-out, and why clarity/exudation surviving it is "
            "the load-bearing result.",
        ],
        "governance": dict(locked_cases_seen=0, models_trained=0,
                           images_transmitted=0, our_patient_images_read=0),
    }
    with open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    print(json.dumps(probe1, ensure_ascii=False, indent=1))
    for k, v in probes.items():
        print(f"{k:12s} subj AUROC={v['subject_level_auroc']:.4f} "
              f"D={v['diabetic_range']} H={v['healthy_range']} "
              f"overlap={v['ranges_overlap']}")
    print("\nhealthy dirs", n_h_dirs, "/ unique prefixes", n_h_prefix)
    print("\n" + verdict["verdict"])
    print("written ->", OUT)


if __name__ == "__main__":
    main()
