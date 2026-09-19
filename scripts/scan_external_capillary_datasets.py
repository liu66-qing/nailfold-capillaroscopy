"""What do the two external capillaroscopy datasets actually supply?

Dataset A (downloaded, local): Mendeley 8wrjdknb5k v1, "The number of nail fold
capillaries and nail fold bleedings reflects the clinical manifestations of
systemic sclerosis". Scanned from disk.

Dataset B (NOT downloaded): HuggingFace HanaNguyen/Capillary-Dataset. Scanned
from the HF metadata API plus the authors' data.yaml on GitHub -- file tree,
subject folders and split membership only. No image bytes are fetched, so this
costs nothing and leaks nothing.

The question is not "is there more data" -- it is which of OUR fields each
dataset could supervise, on how many PEOPLE, and whether the supervision is
machine-readable at all. A dataset whose labels exist only as coloured pixels
burned into the image is not a label file; recovering it is our work, and the
overlay damages the pixels any appearance field would have to read.

Writes one JSON. Reads nothing of ours: no manifest, no locked case, no model.

Usage:
  python scripts/scan_external_capillary_datasets.py \
      --out artifacts/evidence/external_datasets_20260919/external.json
"""
import argparse
import collections
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage

MENDELEY_DIR = ("data/The number of nail fold capillaries/The number of nail "
                "fold capillaries and nail fold bleedings reflects the "
                "clinical manifestations of systemic sclerosis")
MENDELEY_API = ("https://data.mendeley.com/public-api/datasets/"
                "8wrjdknb5k/files?folder_id=root&version=1")
HF_API = "https://huggingface.co/api/datasets/HanaNguyen/Capillary-Dataset"
HF_YAML = ("https://raw.githubusercontent.com/urgonguyen/Capillarydataset/"
           "main/improved_yolov8_morphology/data.yaml")

# exact-value colour keys. Annotation software writes saturated primaries;
# a capillaroscopy photograph essentially never contains exact 255/0/0.
COLOR_KEYS = {"red": (255, 0, 0), "yellow": (255, 255, 0),
              "white": (255, 255, 255), "green": (0, 255, 0),
              "blue": (0, 0, 255), "cyan": (0, 255, 255),
              "magenta": (255, 0, 255)}
MIN_COMPONENT_PX = 8  # below this, JPEG-style ringing and stray pixels dominate


def fetch(url, tries=3):
    """Both hosts drop header-less urllib requests, so send a UA and retry."""
    last = None
    for _k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "curl/8.0", "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=90) as fh:
                return fh.read()
        except Exception as e:  # transient RemoteDisconnected is common here
            last = e
    raise last


def get_json(url):
    return json.loads(fetch(url).decode("utf-8", "replace"))


def scan_mendeley(root):
    """Count files, group by patient/exam from the filename, find the overlay."""
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(".png"))
    empty = [f for f in files if os.path.getsize(os.path.join(root, f)) == 0]
    usable = [f for f in files if f not in set(empty)]

    # filenames are YYMMDD + romanised patient name + image index, e.g.
    # '160422akiyama koichi1.png'. date+name = one exam; name = one person.
    exam_of, person_of = {}, {}
    for f in files:
        stem = re.sub(r"\.png$", "", f, flags=re.I)
        stem = re.sub(r"\d+$", "", stem)          # drop the image index
        exam_of[f] = stem
        person_of[f] = re.sub(r"^\d{6}", "", stem)  # drop the date
    exams = sorted(set(exam_of.values()))
    persons = sorted(set(person_of.values()))
    per_exam = collections.Counter(exam_of.values())
    exams_per_person = collections.Counter(
        re.sub(r"^\d{6}", "", e) for e in exams)

    # the overlay: exact-valued saturated colours, as connected components
    marker = {k: {"files_with_markers": 0, "components": 0, "areas": [],
                  "bbox_fills": []} for k in COLOR_KEYS}
    sizes = collections.Counter()
    for f in usable:
        a = np.array(Image.open(os.path.join(root, f)).convert("RGB"))
        sizes[a.shape[:2]] += 1
        for name, (R, G, B) in COLOR_KEYS.items():
            m = ((a[..., 0] == R) & (a[..., 1] == G) & (a[..., 2] == B))
            if not m.any():
                continue
            lb, n = ndimage.label(m)
            hit = 0
            for i, sl in enumerate(ndimage.find_objects(lb)):
                s = (lb[sl] == i + 1)
                area = int(s.sum())
                if area < MIN_COMPONENT_PX:
                    continue
                hit += 1
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                marker[name]["areas"].append(area)
                marker[name]["bbox_fills"].append(area / max(h * w, 1))
            if hit:
                marker[name]["files_with_markers"] += 1
                marker[name]["components"] += hit
    return files, empty, usable, exams, persons, per_exam, \
        exams_per_person, marker, sizes


def summarise_marker(m):
    if not m["areas"]:
        return {"files_with_markers": 0, "components": 0}
    return {
        "files_with_markers": m["files_with_markers"],
        "components_total": m["components"],
        "median_component_area_px": float(np.median(m["areas"])),
        "p90_component_area_px": float(np.percentile(m["areas"], 90)),
        # a filled dot fills its own bounding box; an open outline or a
        # freehand stroke leaves most of the box empty
        "median_bbox_fill": round(float(np.median(m["bbox_fills"])), 3),
    }


def scan_hf():
    """File tree only. Subject folders and split membership from paths."""
    meta = get_json(HF_API)
    sib = [s["rfilename"] for s in meta.get("siblings", [])]

    dia_imgs = collections.Counter()
    hea_imgs = collections.Counter()
    for s in sib:
        p = s.split("/")
        if s.startswith("classification_original/diabetic_patients/"):
            dia_imgs[p[2]] += 1
        elif s.startswith("classification_original/healthy_subjects/"):
            hea_imgs[p[2]] += 1

    # healthy folders are <person>-<session>; diabetic folders are one per
    # patient. So the healthy arm has far fewer PEOPLE than folders.
    hea_person = collections.Counter()
    for k, v in hea_imgs.items():
        hea_person[k.split("-")[0]] += v

    # the published split: does one person's frames appear in two splits?
    # healthy filenames keep the '<person>-<session>_<idx>' stem, so this is
    # checkable without downloading anything.
    sess_splits = collections.defaultdict(set)
    dia_per_split = collections.Counter()
    for s in sib:
        if not s.startswith("Classification/data_1x1_224/"):
            continue
        p = s.split("/")
        split, cls, fn = p[2], p[3], p[4]
        if cls == "healthy":
            mm = re.match(r"^(\d+-\d+)_", fn)
            if mm:
                sess_splits[mm.group(1)].add(split)
        else:
            # diabetic frames were renumbered 0..N with no patient id left in
            # the filename, so their split cannot be audited from the tree
            dia_per_split[split] += 1
    person_splits = collections.defaultdict(set)
    for k, v in sess_splits.items():
        person_splits[k.split("-")[0]] |= v

    det = collections.Counter()
    det_prefix = collections.Counter()
    for s in sib:
        if s.startswith("Morphology_detection/"):
            det["/".join(s.split("/")[1:-1])] += 1
            if "/labels/" in s:
                det_prefix[re.sub(r"[0-9].*", "", s.split("/")[-1])] += 1

    yml = fetch(HF_YAML).decode("utf-8", "replace")
    cls_names = re.search(r"names:\s*\[(.*?)\]", yml)
    names = [c.strip().strip("'\"") for c in cls_names.group(1).split(",")] \
        if cls_names else []
    return (meta, sib, dia_imgs, hea_imgs, hea_person, sess_splits,
            person_splits, dia_per_split, det, det_prefix, names)


# Which of OUR 22 fields could each dataset supervise, and at what verdict do
# those fields currently sit? Verdicts are quoted from the committed runs, not
# re-derived here.
OUR_FIELD_REACH = {
    "mendeley_8wrjdknb5k": {
        "capillary_count": "FAIL dev +0.050 CI[-0.011,+0.110]; LOAO passes "
                           "nowhere. Mendeley counts NFCs -> same field.",
        "hemorrhage": "FAIL collapses to one class, AUROC 0.331. Mendeley "
                      "counts NFBs (nail fold bleedings) -> same field.",
        "clarity": "PASS LOAO 3/3 -- but NO label in this dataset.",
        "exudation": "PASS LOAO 3/3 -- but NO label in this dataset.",
    },
    "hf_capillary_dataset": {
        "crossing_malformation": "'crossing'/'tortuous'/'bushy' map here. Our "
                                 "segmenter's correlation sign is INVERTED "
                                 "(rho=-0.257); more boxes of the same kind "
                                 "does not fix a sign error.",
        "disease_endpoint_diabetes": "the ONLY external per-person disease "
                                     "label either dataset carries. Our own "
                                     "reports carry none (139/139 disease "
                                     "words are image-inferred).",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendeley-root", default=MENDELEY_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-net", action="store_true")
    a = ap.parse_args()

    (files, empty, usable, exams, persons, per_exam, exams_per_person,
     marker, sizes) = scan_mendeley(a.mendeley_root)

    remote = None
    if not a.skip_net:
        try:
            rf = get_json(MENDELEY_API)
            names = {(f.get("filename") or f.get("name")): f.get("size")
                     for f in rf}
            remote = {
                "files_listed_on_mendeley": len(names),
                "files_present_locally": len(files),
                "zero_byte_on_mendeley": sum(1 for v in names.values()
                                             if v == 0),
                "non_png_on_mendeley": [n for n in names
                                        if not n.lower().endswith(".png")],
                "note": "zero-byte files are zero-byte AT SOURCE, not a "
                        "broken download.",
            }
        except Exception as e:  # network is optional for the local half
            remote = {"error": "%s: %s" % (type(e).__name__, e)}

    mend = {
        "root": a.mendeley_root,
        "doi": "10.17632/8wrjdknb5k.1",
        "licence": "CC BY 4.0",
        "png_files": len(files),
        "zero_byte_files": len(empty),
        "zero_byte_names": empty,
        "images_readable": len(usable),
        "exams_date_plus_name": len(exams),
        "distinct_persons_by_name": len(persons),
        "persons_with_more_than_one_exam": sum(
            1 for v in exams_per_person.values() if v > 1),
        "images_per_exam_histogram": dict(
            sorted(collections.Counter(per_exam.values()).items())),
        "distinct_image_sizes": len(sizes),
        "most_common_size": list(max(sizes.items(),
                                     key=lambda kv: kv[1])[0]) if sizes else [],
        "machine_readable_label_files": 0,
        "label_file_extensions_present": [],
        "overlay_markers_burned_into_pixels": {
            k: summarise_marker(v) for k, v in marker.items()},
        "remote_file_list_check": remote,
    }
    print_mendeley(mend, persons, exams_per_person)
    hf = None
    if not a.skip_net:
        try:
            hf = build_hf()
        except Exception as e:
            hf = {"error": "%s: %s" % (type(e).__name__, e)}

    out = {
        "question": "how much do these two external datasets help, measured in "
                    "PEOPLE and in which of our fields they could supervise?",
        "our_cohort_for_comparison": {
            "development_cases": 186, "locked_cases": 47,
            "cases_needed_for_an_independent_disease_endpoint": "150-200 by "
                "our own ruler (at n=186, malformation_ratio +0.080, "
                "microthrombus +0.077 and capillary_count +0.050 all still "
                "have CIs including 0)",
        },
        "mendeley_8wrjdknb5k": mend,
        "hf_capillary_dataset": hf,
        "which_of_our_fields_each_could_reach": OUR_FIELD_REACH,
        "images_opened_from_our_cohort": 0,
        "locked_cases_seen": 0,
        "models_run": 0,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote %s" % a.out)


def print_mendeley(m, persons, exams_per_person):
    print("=== Mendeley 8wrjdknb5k ===")
    print("png files            %d (zero-byte at source: %d)"
          % (m["png_files"], m["zero_byte_files"]))
    print("exams (date+name)    %d" % m["exams_date_plus_name"])
    print("DISTINCT PERSONS     %d   (%d of them have >1 exam)"
          % (m["distinct_persons_by_name"],
             m["persons_with_more_than_one_exam"]))
    print("label files          %d  <-- annotation is burned into the pixels"
          % m["machine_readable_label_files"])
    for k, v in m["overlay_markers_burned_into_pixels"].items():
        if v.get("components_total"):
            print("  %-8s files=%3d comps=%5d median_area=%6.0f fill=%.2f"
                  % (k, v["files_with_markers"], v["components_total"],
                     v["median_component_area_px"], v["median_bbox_fill"]))


def build_hf():
    (meta, sib, dia_imgs, hea_imgs, hea_person, sess_splits, person_splits,
     dia_per_split, det, det_prefix, names) = scan_hf()
    leak_sess = {k: sorted(v) for k, v in sess_splits.items() if len(v) > 1}
    leak_person = {k: sorted(v) for k, v in person_splits.items() if len(v) > 1}
    hf = {
        "repo": "HanaNguyen/Capillary-Dataset",
        "licence": meta.get("cardData", {}).get("license"),
        "downloaded": False,
        "total_bytes": meta.get("usedStorage"),
        "files_listed": len(sib),
        "classification_original": {
            "diabetic_patient_folders": len(dia_imgs),
            "diabetic_images": sum(dia_imgs.values()),
            "healthy_person_session_folders": len(hea_imgs),
            "healthy_DISTINCT_PERSONS": len(hea_person),
            "healthy_images": sum(hea_imgs.values()),
            "total_original_images": sum(dia_imgs.values())
                                     + sum(hea_imgs.values()),
            "people_total": len(dia_imgs) + len(hea_person),
            "note": "healthy folders are <person>-<session>; %d folders are "
                    "only %d people. Diabetic folders are one per patient."
                    % (len(hea_imgs), len(hea_person)),
        },
        "published_split_audit_data_1x1_224": {
            "healthy_sessions_appearing_in_more_than_one_split": leak_sess,
            "healthy_PERSONS_appearing_in_more_than_one_split": leak_person,
            "diabetic_frames_per_split": dict(dia_per_split),
            "diabetic_split_auditable": False,
            "why": "diabetic frames were renumbered 0..N, dropping the patient "
                   "id, so their split cannot be checked from the file tree. "
                   "The healthy arm, which CAN be checked, already leaks %d "
                   "people across splits -- so the published accuracy is not "
                   "a patient-disjoint number and must not be quoted as one."
                   % len(leak_person),
        },
        "morphology_detection": {
            "counts_by_folder": dict(det),
            "label_filename_prefixes": dict(det_prefix),
            "class_names_from_authors_data_yaml": names,
            "yaml_url": HF_YAML,
            "train_folder_is_augmented": "data.yaml train: images/train_aug",
        },
        "concat_variants_are_not_new_images": {
            "note": "data_concat_2x2 / 3x3 / 4x1 / 4x4 / 1x9 are mosaics of "
                    "the SAME crops as data_1x1_224. 48027 Classification "
                    "files come from 8368 originals; tiling does not add "
                    "people, and the 2x2/4x1 variants (16338 each) are "
                    "combinatorial re-pairings of the same frames.",
        },
    }
    print("\n=== HF HanaNguyen/Capillary-Dataset (metadata only) ===")
    c = hf["classification_original"]
    print("original images      %d" % c["total_original_images"])
    print("diabetic patients    %d" % c["diabetic_patient_folders"])
    print("healthy PERSONS      %d  (in %d person-session folders)"
          % (c["healthy_DISTINCT_PERSONS"], c["healthy_person_session_folders"]))
    print("PEOPLE TOTAL         %d" % c["people_total"])
    print("detection classes    %s" % names)
    print("split leak (healthy) persons=%s" % leak_person)
    return hf


if __name__ == "__main__":
    main()
