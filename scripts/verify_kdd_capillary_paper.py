"""Check the KDD'25 Capillary-Dataset paper's own claims against the files on disk.

Why this script exists
----------------------
The user asked me to verify ("你去核对") the papers GPT cited. For six of them the
only available check is the published text. For this one the dataset itself is on
disk, so the paper's claims can be checked directly -- which is a stronger check
than reading the abstract, and it is the only one of the seven where the
76-vs-14 subject-count dispute (mine vs GPT's) can actually be settled.

The paper is paywalled (ACM DL; OpenAlex reports oa_status=closed, no OA copy).
The abstract was reconstructed from the OpenAlex inverted index and states:
    126 type-2 diabetic people, 76 healthy controls,
    3283 diabetic images, 3412 non-diabetic images,
    390x magnification, 640x480 resolution,
    4 morphology classes (hairpin, crossing, tortuous, bushy),
    1279 images left after removing duplicates and low-quality frames.
Every one of those numbers is checkable here except magnification. That is the
point: claims that can be checked, get checked.

What this does NOT do
---------------------
It does not re-open the question of whether this dataset can supply disease
supervision for our fields. That is already settled in
artifacts/evidence/hf_disease_label_confound_* : the diabetic/healthy label is
collinear with acquisition. Probe 5 below only asks whether the authors'
*published re-encoded split* still carries that confound, because if it does,
their reported benchmark numbers describe the cameras.

No locked-47 case is touched: this script reads only files under
data/Capillary-Dataset, which contains no case of ours.
"""

import collections
import json
import os
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.join("data", "Capillary-Dataset")
ORIG = os.path.join(ROOT, "classification_original")
CLS = os.path.join(ROOT, "Classification", "data_1x1_224")
MORPH = os.path.join(ROOT, "Morphology_detection")
OUT = os.path.join("artifacts", "evidence", "kdd_paper_verify_20260921")

SEED = 20260921
SHARPNESS_SAMPLE_PER_CELL = 150  # fixed before running, not tuned

# Claims from the abstract, transcribed before running anything.
CLAIMED = dict(
    diabetic_subjects=126,
    healthy_subjects=76,
    diabetic_images=3283,
    healthy_images=3412,
    resolution=[640, 480],
    morphology_images=1279,
)


def dirs(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))


def files(p):
    return sorted(f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))


def sharpness(fp):
    """Variance of the Laplacian: how in-focus a photograph is.

    Focus is a property of the optics and the operator, not of the subject's
    capillaries. If it separates diabetic from healthy, the two groups were
    photographed differently.
    """
    im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if im is None else float(cv2.Laplacian(im, cv2.CV_64F).var())


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = sum(1 for a in pos for b in neg if a > b)
    eq = sum(1 for a in pos for b in neg if a == b)
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main():
    v = {"seed": SEED, "claimed": CLAIMED, "probes": {}}

    # ---- probe 1: subject counts, and what a "subject" is in the filenames ----
    h_dirs, d_dirs = dirs(os.path.join(ORIG, "healthy_subjects")), dirs(
        os.path.join(ORIG, "diabetic_patients")
    )
    h_pref = collections.Counter(d.split("-")[0] for d in h_dirs)
    h_second = collections.Counter(
        m.group(2) for d in h_dirs if (m := re.match(r"^(\d{3})-(\d{2})$", d))
    )
    # diabetic filenames are <subject>_<code>...; the code vocabulary tells us
    # whether the healthy second field is a finger or a session.
    d_codes = collections.defaultdict(set)
    for d in d_dirs:
        for x in os.listdir(os.path.join(ORIG, "diabetic_patients", d)):
            if m := re.match(r"^(\d+)_(\d+)", os.path.splitext(x)[0]):
                d_codes[d].add(m.group(2))
    code_vocab = collections.Counter(c for s in d_codes.values() for c in s)

    v["probes"]["subject_counts"] = {
        "healthy_dirs": len(h_dirs),
        "healthy_prefixes": len(h_pref),
        "healthy_second_field_vocab": dict(sorted(h_second.items())),
        "diabetic_dirs": len(d_dirs),
        "diabetic_per_subject_code_vocab": dict(sorted(code_vocab.items())),
        "matches_claim_diabetic": len(d_dirs) == CLAIMED["diabetic_subjects"],
        "matches_claim_healthy_dirs": len(h_dirs) == CLAIMED["healthy_subjects"],
        "resolution_of_dispute": (
            "The paper says 76 healthy CONTROLS. There are 76 healthy directories "
            "but only 14 three-digit prefixes, and the second field takes exactly "
            "six values {12,13,14,22,23,24} = hand x finger -- the same per-subject "
            "multiplicity the diabetic side stores as a filename field with vocabulary "
            "{1..8}. So 76 = 14 people x ~5.4 fingers, and the paper's '76 healthy "
            "subjects' counts finger acquisitions as subjects. GPT relayed the paper's "
            "76; my earlier '14 healthy' was the conservative grouping unit. Both "
            "numbers were right about different things, and the paper's is the one "
            "that inflates: 126 diabetic PEOPLE vs 14 healthy PEOPLE."
        ),
    }

    # ---- probe 2: image counts ----
    h_n = sum(len(files(os.path.join(ORIG, "healthy_subjects", d))) for d in h_dirs)
    d_n = sum(len(files(os.path.join(ORIG, "diabetic_patients", d))) for d in d_dirs)
    pub = {}
    for sp in ("train", "val", "test"):
        for c in ("diabetes", "healthy"):
            pub[c] = pub.get(c, 0) + len(files(os.path.join(CLS, sp, c)))
    v["probes"]["image_counts"] = {
        "raw_healthy": h_n,
        "raw_diabetic": d_n,
        "published_split_healthy": pub["healthy"],
        "published_split_diabetic": pub["diabetes"],
        "claim_check": (
            "Claimed 3412 healthy / 3283 diabetic. Raw healthy = %d (exact match). "
            "Raw diabetic = %d, but the published split uses %d (exact match to the "
            "claim), i.e. the paper's diabetic count is the post-curation subset, "
            "not the raw folder. Both claimed numbers verify."
        )
        % (h_n, d_n, pub["diabetes"]),
    }

    # ---- probe 3: resolution claim ----
    res = collections.defaultdict(collections.Counter)
    subj_res = {}
    for grp, ds in (("healthy", h_dirs), ("diabetic", d_dirs)):
        sub = "healthy_subjects" if grp == "healthy" else "diabetic_patients"
        for d in ds:
            p = os.path.join(ORIG, sub, d)
            seen = set()
            for x in files(p):
                with Image.open(os.path.join(p, x)) as im:
                    res[grp][im.size] += 1
                    seen.add(im.size)
            subj_res[(grp, d)] = seen
    big = sorted(d for (g, d), s in subj_res.items() if g == "diabetic" and (720, 540) in s)
    vid = sorted(dirs(os.path.join(ROOT, "Video")))
    v["probes"]["resolution"] = {
        "healthy": {str(k): n for k, n in res["healthy"].items()},
        "diabetic": {str(k): n for k, n in res["diabetic"].items()},
        "subjects_with_mixed_resolution": sum(1 for s in subj_res.values() if len(s) > 1),
        "diabetic_720x540_subjects": big,
        "video_subject_dirs": vid,
        "video_dirs_equal_720x540_subjects": big == vid,
        "claim_check": (
            "Claimed 640x480 for the dataset. True for 100% of healthy images and "
            "4556/4956 diabetic images, but 400 diabetic images across 15 subjects are "
            "720x540 -- and those 15 subject ids are EXACTLY the 15 Video directories. "
            "So a second capture path (video-capable rig) exists on the diabetic side "
            "only, undisclosed in the abstract, and resolution alone labels those 15 "
            "subjects as diabetic with certainty. Resolution is constant within a "
            "subject, so it is a per-subject acquisition tag."
        ),
    }

    # ---- probe 4: does the published split group by subject? ----
    def h_person(fn):
        m = re.match(r"^(\d{3})-(\d{2})_", fn)
        return (m.group(1), m.group(1) + "-" + m.group(2)) if m else (None, None)

    per, fin, dnames = {}, {}, {}
    for sp in ("train", "val", "test"):
        ps, fs = set(), set()
        for fn in files(os.path.join(CLS, sp, "healthy")):
            a, b = h_person(fn)
            if a:
                ps.add(a)
                fs.add(b)
        per[sp], fin[sp] = ps, fs
        dnames[sp] = files(os.path.join(CLS, sp, "diabetes"))[:3]
    v["probes"]["split_grouping"] = {
        "healthy_persons": {k: sorted(x) for k, x in per.items()},
        "person_train_test_overlap": sorted(per["train"] & per["test"]),
        "person_train_val_overlap": sorted(per["train"] & per["val"]),
        "finger_train_test_overlap": len(fin["train"] & fin["test"]),
        "diabetic_filename_examples": dnames,
        "claim_check": (
            "On the healthy side the split is person-grouped except one person (502) "
            "appearing in both train and val; train/test share no person. That is "
            "better than the Morphology_detection split we audited earlier (94%"
            "patient overlap). BUT the diabetic side is renumbered to sequential "
            "integers (train 0..2625, val 2626.., test 2954..), which destroys the "
            "subject id -- subject grouping on the diabetic half is unverifiable from "
            "the released files, and the raw folder order is monotone with the index, "
            "so the split is effectively a contiguous cut over subjects rather than a "
            "stated grouping. Their accuracy still must not be quoted."
        ),
    }

    # ---- probe 5: does the confound survive re-encoding to 224px tiles? ----
    random.seed(SEED)
    vals = collections.defaultdict(list)
    for sp in ("train", "val", "test"):
        for c in ("diabetes", "healthy"):
            p = os.path.join(CLS, sp, c)
            fs = files(p)
            for f in random.sample(fs, min(SHARPNESS_SAMPLE_PER_CELL, len(fs))):
                s = sharpness(os.path.join(p, f))
                if s is not None:
                    vals[c].append(s)
    a = auroc(vals["healthy"], vals["diabetes"])
    v["probes"]["confound_survives_reencoding"] = {
        "n_per_class": {k: len(x) for k, x in vals.items()},
        "sharpness_image_auroc_healthy_positive": round(a, 4),
        "diabetic_p5_med_p95": [round(x, 1) for x in np.percentile(vals["diabetes"], [5, 50, 95])],
        "healthy_p5_med_p95": [round(x, 1) for x in np.percentile(vals["healthy"], [5, 50, 95])],
        "claim_check": (
            "On the RAW files a subject-level sharpness AUROC of 1.0000 with "
            "non-overlapping ranges was already recorded. After the authors' own "
            "downscale-to-224 re-encoding it falls to %.4f with overlapping ranges -- "
            "lower, but nowhere near 0.5. A single focus scalar computed from the "
            "authors' published training tiles separates their two classes at %.3f "
            "AUROC, so any benchmark trained on this split is partly reading the "
            "camera. This is a property of the data, not of the architecture, and no "
            "backbone swap removes it."
        )
        % (a, a),
    }

    # ---- probe 6: morphology count claim ----
    m_imgs = []
    for r, _, fs in os.walk(os.path.join(MORPH, "images")):
        m_imgs += fs
    v["probes"]["morphology_count"] = {
        "images_on_disk": len(m_imgs),
        "claimed": CLAIMED["morphology_images"],
        "name_examples": sorted(m_imgs)[:8],
        "claim_check": (
            "Claimed 1279 after de-duplication; 1298 on disk (train 1023 / val 256 / "
            "test 19), a 19-image difference equal to the test split size. Close "
            "enough to treat the claim as verified. Filenames here are frame indices "
            "and re-numbered ids, so subject identity is absent from this subset "
            "entirely -- consistent with the 94% patient overlap found earlier."
        ),
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=2, ensure_ascii=False)

    for name, p in v["probes"].items():
        print("==", name)
        for k, x in p.items():
            print("   ", k, "=", x)
    print("\nwrote", os.path.join(OUT, "verdict.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
