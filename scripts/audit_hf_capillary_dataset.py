"""Audit the HanaNguyen/Capillary-Dataset local copy before any transfer work.

What this answers, and why each question is here:

Q1 counts / integrity     -- do images and labels pair 1:1, are any labels empty,
                             are there duplicate or cross-split identical images?
Q2 class semantics        -- the 4 YOLO classes are bushy/crossing/hairpin/tortuous
                             (from the authors' data.yaml). Recompute OUR two field
                             definitions from them:
                               crossing_ratio    = crossing / all_loops
                               malformation_ratio= (bushy+tortuous) / all_loops
Q3 patient recovery       -- the `dia_NNNN` filenames have the patient id STRIPPED.
                             Recover it by matching each morphology image back into
                             classification_original/diabetic_patients/<patient>/ with a
                             48x48 normalised-intensity nearest neighbour + Lowe ratio
                             test. This is needed for two reasons: to know how many real
                             PEOPLE the boxes cover, and to audit the train/val split.
Q4 scale                  -- compare box size as a FRACTION of the image, which is the
                             only scale-free comparison available (no calibration exists
                             on either side; absolute micron claims are forbidden).
Q5 appearance domain      -- saturation / channel means. Our images are colour
                             capillaroscopy; if HF is near-greyscale, colour-dependent
                             fields cannot borrow from it.

GOVERNANCE
  - This script does NOT train anything and does NOT touch locked-47.
    It reads only the external dataset plus our own DEV-side label distribution.
  - No absolute micron claim is made anywhere. All scale statements are ratios.
  - The external boxes are a third-party annotation, NOT a clinical gold standard.
  - The published classification accuracy of this dataset must not be quoted:
    its split is not patient-disjoint (Q3 proves this on the diabetic side too).

Run:
  python scripts/audit_hf_capillary_dataset.py
Writes:
  artifacts/evidence/hf_capillary_audit_20260920/audit.json
  artifacts/evidence/hf_capillary_audit_20260920/dia_image_to_patient_map.csv
"""

from __future__ import annotations

import collections
import csv
import glob
import hashlib
import json
import os
import re
import statistics as st

import numpy as np
from PIL import Image

ROOT = r"E:\甲劈微循环\data\Capillary-Dataset"
MORPH = os.path.join(ROOT, "Morphology_detection")
CO_DIA = os.path.join(ROOT, "classification_original", "diabetic_patients")
OUT_DIR = r"E:\甲劈微循环\artifacts\evidence\hf_capillary_audit_20260920"

# From the authors' data.yaml (nc: 4). Order matters.
NAMES = {0: "bushy", 1: "crossing", 2: "hairpin", 3: "tortuous"}
# Our field definitions, expressed over their classes.
MALFORMED = (0, 3)   # bushy + tortuous
CROSSING = (1,)

NATIVE_W, NATIVE_H = 640, 480   # verified: 1284/1298 images are exactly this
SIG = 48                        # signature grid for the patient-recovery match
RATIO_GATE = 0.80               # Lowe ratio; d1/d2 below this counts as confident


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q(values, frac: float) -> float:
    values = sorted(values)
    return float(values[int(frac * (len(values) - 1))])


def read_labels(path: str):
    """Return list of (cls, w_frac, h_frac). Raises on a malformed line."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        assert len(parts) == 5, "malformed label line in %s: %r" % (path, line)
        out.append((int(parts[0]), float(parts[3]), float(parts[4])))
    return out


def q1_integrity():
    res = {}
    for split in ("train", "val", "test"):
        imgs = sorted(glob.glob(os.path.join(MORPH, "images", split, "*")))
        labs = sorted(glob.glob(os.path.join(MORPH, "labels", split, "*.txt")))
        stems_i = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
        stems_l = {os.path.splitext(os.path.basename(p))[0] for p in labs}
        res[split] = {
            "images": len(imgs),
            "labels": len(labs),
            "unpaired_images": sorted(stems_i - stems_l)[:10],
            "unpaired_labels": sorted(stems_l - stems_i)[:10],
            "empty_label_files": sum(1 for p in labs if os.path.getsize(p) == 0),
        }
    # identical pixels across splits would be a leak we must report
    by_hash = collections.defaultdict(list)
    sizes = collections.Counter()
    for split in ("train", "val", "test"):
        for p in glob.glob(os.path.join(MORPH, "images", split, "*")):
            by_hash[md5(p)].append(split + "/" + os.path.basename(p))
            sizes[Image.open(p).size] += 1
    dups = {k: v for k, v in by_hash.items() if len(v) > 1}
    res["unique_md5"] = len(by_hash)
    res["duplicate_groups"] = [v for v in dups.values()]
    res["cross_split_duplicate_groups"] = [
        v for v in dups.values() if len({x.split("/")[0] for x in v}) > 1
    ]
    res["image_sizes"] = {"%dx%d" % k: v for k, v in sizes.most_common()}
    return res


def q2_class_semantics():
    """Per-image ratios in OUR field definitions, split by diabetic vs healthy."""
    cls_by_split = collections.Counter()
    geom = collections.defaultdict(lambda: {"w": [], "h": [], "ar": []})
    rows = []
    for split in ("train", "val", "test"):
        for p in glob.glob(os.path.join(MORPH, "labels", split, "*.txt")):
            stem = os.path.splitext(os.path.basename(p))[0]
            boxes = read_labels(p)
            counter = collections.Counter(c for c, _, _ in boxes)
            for c, wf, hf in boxes:
                cls_by_split[(split, c)] += 1
                w = wf * NATIVE_W
                h = hf * NATIVE_H
                geom[c]["w"].append(w)
                geom[c]["h"].append(h)
                geom[c]["ar"].append(h / max(w, 1e-6))
            total = sum(counter.values())
            rows.append(
                {
                    "split": split,
                    "stem": stem,
                    "group": "hea" if stem.startswith("hea_") else ("dia" if stem.startswith("dia_") else "other"),
                    "loops": total,
                    "crossing": sum(counter[c] for c in CROSSING),
                    "malformed": sum(counter[c] for c in MALFORMED),
                    "crossing_ratio": sum(counter[c] for c in CROSSING) / total,
                    "malformation_ratio": sum(counter[c] for c in MALFORMED) / total,
                }
            )

    def bins(values, edges, labels):
        out = collections.Counter()
        for v in values:
            for e, lab in zip(edges, labels):
                if v <= e:
                    out[lab] += 1
                    break
            else:
                out[labels[-1]] += 1
        return dict(out)

    per_group = {}
    for group in ("dia", "hea"):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        cr = [r["crossing_ratio"] for r in sub]
        mr = [r["malformation_ratio"] for r in sub]
        per_group[group] = {
            "images": len(sub),
            "loops_per_image_median": st.median([r["loops"] for r in sub]),
            "crossing_ratio": {"median": st.median(cr), "p25": q(cr, .25), "p75": q(cr, .75)},
            "malformation_ratio": {"median": st.median(mr), "p25": q(mr, .25), "p75": q(mr, .75)},
            # binned onto OUR label brackets so they are directly comparable
            "crossing_bins_our_brackets": bins(cr, [.30, .60, .80], ["<=30%", "30-60%", "60-80%", ">80%"]),
            "malformation_bins_our_brackets": bins(mr, [.10, .30, .60], ["<=10%", "10-30%", "30-60%", ">60%"]),
        }

    def auroc(pos, neg):
        pos = np.asarray(pos)
        neg = np.asarray(neg)
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

    dia = [r for r in rows if r["group"] == "dia"]
    hea = [r for r in rows if r["group"] == "hea"]
    separation = {
        "note": "diabetes-vs-healthy separability of the two ratios computed from THEIR boxes",
        "auroc_crossing_ratio": auroc([r["crossing_ratio"] for r in dia], [r["crossing_ratio"] for r in hea]),
        "auroc_malformation_ratio": auroc([r["malformation_ratio"] for r in dia], [r["malformation_ratio"] for r in hea]),
        "n_dia": len(dia),
        "n_hea": len(hea),
    }

    geometry = {}
    for c, d in sorted(geom.items()):
        geometry[NAMES[c]] = {
            "n": len(d["w"]),
            "w_px_median": st.median(d["w"]),
            "h_px_median": st.median(d["h"]),
            "aspect_h_over_w_median": st.median(d["ar"]),
            "aspect_p25": q(d["ar"], .25),
            "aspect_p75": q(d["ar"], .75),
        }
    return {
        "class_counts": {"%s/%s" % (s, NAMES[c]): n for (s, c), n in sorted(cls_by_split.items())},
        "total_boxes": sum(cls_by_split.values()),
        "per_group": per_group,
        "diabetes_separation": separation,
        "box_geometry_native_px": geometry,
    }, rows


def _signature(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((SIG, SIG)), np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def q3_recover_patients(rows):
    """The dia_NNNN names have the patient id stripped; recover it by pixel match.

    Matching is done on a 48x48 contrast-normalised intensity signature. A match is
    accepted only when the Lowe ratio d1/d2 < RATIO_GATE, i.e. the best candidate is
    clearly better than the runner-up. This is a provenance recovery, not a
    measurement, so an approximate signature is adequate -- but the ratio gate is
    what makes it trustworthy, NOT the absolute distance (my first attempt used an
    absolute threshold and rejected every true match).
    """
    gallery = sorted(glob.glob(os.path.join(CO_DIA, "*", "*")))
    G = np.stack([_signature(p) for p in gallery])
    queries = sorted(glob.glob(os.path.join(MORPH, "images", "train", "dia_*"))) + sorted(
        glob.glob(os.path.join(MORPH, "images", "val", "dia_*"))
    )
    Q = np.stack([_signature(p) for p in queries])
    D = -2 * Q @ G.T + (G * G).sum(1)[None, :] + (Q * Q).sum(1)[:, None]
    order = np.argsort(D, axis=1)[:, :2]

    mapping = []
    for i, p in enumerate(queries):
        j1, j2 = order[i]
        d1 = float(np.sqrt(max(D[i, j1], 0.0)))
        d2 = float(np.sqrt(max(D[i, j2], 0.0)))
        split = "train" if (os.sep + "train" + os.sep) in p else "val"
        mapping.append(
            {
                "img": os.path.basename(p),
                "split": split,
                "patient": os.path.basename(os.path.dirname(gallery[j1])),
                "src": os.path.basename(gallery[j1]),
                "d1": round(d1, 2),
                "ratio": round(d1 / (d2 + 1e-9), 3),
            }
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "dia_image_to_patient_map.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping[0].keys()))
        w.writeheader()
        w.writerows(mapping)

    conf = [m for m in mapping if m["ratio"] < RATIO_GATE]
    by_patient = collections.defaultdict(set)
    for m in conf:
        by_patient[m["patient"]].add(m["split"])
    leaky = sorted(p for p, s in by_patient.items() if len(s) > 1)
    n_leaky_images = sum(1 for m in conf if m["patient"] in set(leaky))

    # patient-level ratios, which is the level a product must answer at
    per_patient = collections.defaultdict(lambda: {"loops": 0, "crossing": 0, "malformed": 0, "images": 0})
    by_stem = {r["stem"]: r for r in rows}
    for m in conf:
        stem = os.path.splitext(m["img"])[0]
        r = by_stem.get(stem)
        if r is None:
            continue
        d = per_patient[m["patient"]]
        d["loops"] += r["loops"]
        d["crossing"] += r["crossing"]
        d["malformed"] += r["malformed"]
        d["images"] += 1
    cr = [d["crossing"] / d["loops"] for d in per_patient.values() if d["loops"]]
    mr = [d["malformed"] / d["loops"] for d in per_patient.values() if d["loops"]]

    return {
        "queries": len(mapping),
        "gallery_files": len(gallery),
        "confident_matches": len(conf),
        "ratio_median": float(np.median([m["ratio"] for m in mapping])),
        "distinct_patients_recovered": len(by_patient),
        "images_per_patient_median": int(np.median([d["images"] for d in per_patient.values()])),
        "images_per_patient_max": int(max(d["images"] for d in per_patient.values())),
        "patients_in_both_train_and_val": len(leaky),
        "leaky_patient_examples": leaky[:20],
        "images_in_leaky_patients": n_leaky_images,
        "leak_fraction_of_confident": round(n_leaky_images / max(len(conf), 1), 3),
        "patient_level_crossing_ratio": {"median": st.median(cr), "p25": q(cr, .25), "p75": q(cr, .75)},
        "patient_level_malformation_ratio": {"median": st.median(mr), "p25": q(mr, .25), "p75": q(mr, .75)},
    }


def q4_scale():
    """Scale comparison as a FRACTION of image size -- the only calibration-free axis.

    Ours comes from the SAM proposal polygons in segmiss_v1, the only file that holds
    real vessel coordinates on our own images (1024x768).
    """
    ours_w, ours_h = [], []
    src = r"E:\甲劈微循环\artifacts\features\segmiss_v1\ai_proposed_misses.jsonl"
    for line in open(src, encoding="utf-8"):
        poly = json.loads(line).get("poly")
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        ours_w.append(max(xs) - min(xs))
        ours_h.append(max(ys) - min(ys))

    hf_w, hf_h = [], []
    for split in ("train", "val"):
        for p in glob.glob(os.path.join(MORPH, "labels", split, "*.txt")):
            for _, wf, hf in read_labels(p):
                hf_w.append(wf)
                hf_h.append(hf)

    ours_wf = st.median(ours_w) / 1024.0
    ours_hf = st.median(ours_h) / 768.0
    return {
        "ours_source": src,
        "ours_n": len(ours_w),
        "ours_bbox_w_px_median": st.median(ours_w),
        "ours_bbox_h_px_median": st.median(ours_h),
        "ours_aspect_h_over_w_median": st.median([h / max(w, 1e-6) for w, h in zip(ours_w, ours_h)]),
        "ours_w_frac_of_width": round(ours_wf, 4),
        "ours_h_frac_of_height": round(ours_hf, 4),
        "hf_n": len(hf_w),
        "hf_w_frac_of_width": round(st.median(hf_w), 4),
        "hf_h_frac_of_height": round(st.median(hf_h), 4),
        "ratio_width_frac_hf_over_ours": round(st.median(hf_w) / ours_wf, 2),
        "ratio_height_frac_hf_over_ours": round(st.median(hf_h) / ours_hf, 2),
        "interpretation": (
            "The width ratio is the transferable-scale gap. The much larger height ratio "
            "means their boxes span a whole hairpin loop top-to-bottom while our SAM "
            "proposals are loop fragments (our aspect ~0.9 vs theirs ~2.4-2.9), so box "
            "DEFINITION must be aligned before any box is transferred."
        ),
    }


def q5_appearance(n_sample=150, seed=1):
    """Colour domain gap. If HF is near-greyscale, colour fields cannot borrow from it."""
    import random

    random.seed(seed)

    def summarise(paths):
        rows = []
        for p in paths:
            try:
                a = np.asarray(Image.open(p).convert("RGB").resize((256, 192)), np.float32) / 255.0
            except Exception:
                continue
            mx = a.max(2)
            mn = a.min(2)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
            rows.append([a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean(), sat.mean()])
        A = np.array(rows)
        return {
            "n": len(A),
            "R_mean": round(float(A[:, 0].mean() * 255), 1),
            "G_mean": round(float(A[:, 1].mean() * 255), 1),
            "B_mean": round(float(A[:, 2].mean() * 255), 1),
            "saturation_mean": round(float(A[:, 3].mean()), 4),
            "saturation_p95": round(float(np.percentile(A[:, 3], 95)), 4),
        }

    hf_paths = sorted(glob.glob(os.path.join(MORPH, "images", "train", "*")))
    hf_paths = random.sample(hf_paths, min(n_sample, len(hf_paths)))
    ours = []
    for k in (1, 2, 3):
        fs = [
            f
            for f in glob.glob(r"E:\甲劈微循环\data\recovered_archive%d\**\*" % k, recursive=True)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        ours += random.sample(fs, min(50, len(fs)))
    return {"hf": summarise(hf_paths), "ours": summarise(ours)}


def q6_our_target_distribution():
    """Our own DEV distribution for the two fields HF could plausibly help.

    Uses development_fold (NaN == locked-47) -- the `split` column must never be used,
    it yields the opposite conclusion. Locked cases are excluded at CASE level.
    """
    import pandas as pd

    man = pd.read_csv(r"E:\甲劈微循环\artifacts\manifest\locked_evaluation_v1.csv")
    dev_ids = set(man.loc[man["development_fold"].notna(), "exam_case_id"].astype(str))
    locked_ids = set(man.loc[man["development_fold"].isna(), "exam_case_id"].astype(str))
    lab = pd.read_csv(r"E:\甲劈微循环\artifacts\audits\canonical_labels.csv")
    lab["k"] = lab["exam_case_id"].astype(str)
    dev = lab[lab["k"].isin(dev_ids)]
    assert not (set(dev["k"]) & locked_ids), "locked case leaked into the DEV frame"

    out = {"dev_cases": int(len(dev)), "locked_cases_seen": 0, "models_run": 0}
    for field in ("crossing_ratio", "malformation_ratio"):
        vc = dev[field].value_counts(dropna=False)
        labelled = dev[field].dropna()
        counts = labelled.value_counts()
        out[field] = {
            "labelled": int(len(labelled)),
            "missing": int(len(dev) - len(labelled)),
            "distribution": {str(k): int(v) for k, v in vc.items()},
            "mode_label": str(counts.index[0]),
            "mode_baseline_accuracy": round(float(counts.iloc[0] / len(labelled)), 3),
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    out["dataset_root"] = ROOT
    out["q1_integrity"] = q1_integrity()
    sem, rows = q2_class_semantics()
    out["q2_class_semantics"] = sem
    out["q3_patient_recovery"] = q3_recover_patients(rows)
    out["q4_scale"] = q4_scale()
    out["q5_appearance"] = q5_appearance()
    out["q6_our_target_distribution"] = q6_our_target_distribution()
    out["governance"] = {
        "locked_cases_seen": 0,
        "models_run": 0,
        "trained_on_external_data": False,
        "absolute_micron_claims": 0,
        "external_boxes_are_clinical_gold_standard": False,
        "published_accuracy_quotable": False,
    }
    path = os.path.join(OUT_DIR, "audit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print("wrote", path)
    print(json.dumps(out["q3_patient_recovery"], indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

