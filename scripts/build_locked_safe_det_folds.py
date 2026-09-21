"""
Build locked-safe, group-disjoint detection folds for the 3-class vessel detector.

Why this script exists
----------------------
`data/yolo_det_3class/fold{0..4}` cannot be used as it stands:

1. Its `images/` directories are EMPTY (0 files against 9280/2320 labels) --
   the images only ever existed on the server, so nothing trains locally.
2. `dataset.yaml` points at `/root/nailfold/data/yolo_det_3class/fold{N}`.
3. Every fold contains known-locked original_ids (20-29 in train, 3-12 in
   val). Training on those would consume the locked-47 holdout.

What it does differently
------------------------
- Drops every original_id flagged EXACT_RECOVERED_LOCKED, and additionally
  every id listed in the extra-exclusion file if one is present (that file is
  produced by the near-duplicate leak check; md5 matching alone cannot see a
  re-encoded copy of a locked image).
- Keeps each original_id's 20 augmentations TOGETHER in one fold. They are
  geometric variants of the same nailfold, so splitting them would leak.
  Governance for this dataset explicitly forbids a file-level random split.
- Stages images to an ASCII path. cv2 on Windows returns None from
  `imread` on the non-ASCII `血管数据集` path (confirmed, not assumed), and
  ultralytics reads through cv2, so a pure symlink back into the Chinese path
  is not enough -- the staged tree must itself be ASCII.
- Uses hardlinks where possible so 0.49 GB is not duplicated five times.

The fold assignment is recomputed here rather than reused, because the old
assignment was made over a pool that still contained the locked ids.
"""

import os
import csv
import json
import random
import hashlib
import argparse
import collections

import pandas as pd

ROOT = r"E:\甲劈微循环"
SRC_IMG = os.path.join(ROOT, "data", "血管数据集", "分类数据集", "image")
ALL_LABELS = os.path.join(ROOT, "data", "yolo_det_3class", "all_labels")
MAPPING = os.path.join(
    ROOT, "artifacts", "audits", "vascular_dataset_governance_20260830",
    "source_case_mapping.csv")
# written by scripts/leakcheck_unmapped_vs_locked.py if near-duplicates are found.
#
# Two lists exist. The 24-id one is what the committed 5-fold run actually used;
# the 45-id one is the full-resolution-confirmed superset (the 24 are a strict
# subset) and hits 17 of the 47 locked cases. --exclude-list selects which, so
# the committed run stays reproducible instead of being silently changed.
#
# The expanded file is TAB-SEPARATED (id, locked_case, filename, full-res MAE,
# null-floor distance), so it must be parsed on the first field. Pointing the old
# whole-line parser at it would have matched nothing and excluded nothing while
# appearing to work -- exactly the kind of silent no-op that makes a leakage
# guard worthless.
EXCLUDE_DIR = os.path.join(
    ROOT, "artifacts", "evidence", "leakcheck_unmapped_20260920")
EXCLUDE_FILES = {
    "committed24": os.path.join(EXCLUDE_DIR, "exclude_original_ids.txt"),
    "expanded45": os.path.join(EXCLUDE_DIR, "exclude_original_ids_expanded.txt"),
}

STAGE = os.path.join(ROOT, "data", "det_stage_lockedsafe")
OUT_DIR = os.path.join(ROOT, "artifacts", "evidence", "det_folds_20260920")
N_FOLDS = 5
SEED = 20260920
NAMES = {0: "vessel", 1: "malformed_vessel", 2: "cross_vessel"}


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        import shutil
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split and exclusions without staging files")
    ap.add_argument("--exclude-list", choices=sorted(EXCLUDE_FILES),
                    default="committed24",
                    help="committed24 reproduces the committed 5-fold run; "
                         "expanded45 is the full-res-confirmed superset and is "
                         "required before any locked-47 use")
    args = ap.parse_args()

    mp = pd.read_csv(MAPPING)
    mp["original_id"] = mp["original_id"].astype(str)
    locked_ids = set(mp.loc[mp["source_mapping_status"] ==
                            "EXACT_RECOVERED_LOCKED", "original_id"])

    exclude_path = EXCLUDE_FILES[args.exclude_list]
    extra = set()
    if os.path.exists(exclude_path):
        with open(exclude_path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                extra.add(ln.split("\t")[0].split()[0])
    # a leakage guard that silently matches nothing is worse than none at all
    if not extra:
        raise AssertionError(
            f"parsed 0 ids from {exclude_path}; refusing to stage an "
            "unguarded dataset")
    print(f"exclusion list {args.exclude_list}: {len(extra)} original_ids "
          f"from {os.path.basename(exclude_path)}")

    # inventory: which original_ids actually have images AND labels on disk
    have = collections.defaultdict(list)
    for fn in os.listdir(SRC_IMG):
        if not fn.lower().endswith(".jpg"):
            continue
        stem = fn[:-4]
        if "_" not in stem:
            continue
        oid = stem.split("_")[0]
        if os.path.exists(os.path.join(ALL_LABELS, stem + ".txt")):
            have[oid].append(stem)

    all_ids = sorted(have, key=lambda s: int(s))
    excluded = (locked_ids | extra) & set(all_ids)
    keep_ids = [i for i in all_ids if i not in excluded]

    # class counts on the kept pool, so the training target is known up front
    cls = collections.Counter()
    n_imgs = 0
    empty_label_files = 0
    for oid in keep_ids:
        for stem in have[oid]:
            n_imgs += 1
            with open(os.path.join(ALL_LABELS, stem + ".txt"),
                      encoding="utf-8") as fh:
                k = 0
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        cls[int(float(ln.split()[0]))] += 1
                        k += 1
                if k == 0:
                    empty_label_files += 1

    rng = random.Random(SEED)
    shuffled = keep_ids[:]
    rng.shuffle(shuffled)
    fold_of = {oid: i % N_FOLDS for i, oid in enumerate(shuffled)}

    report = {
        "seed": SEED,
        "n_original_ids_on_disk": len(all_ids),
        "n_excluded_total": len(excluded),
        "n_excluded_known_locked": len(locked_ids & set(all_ids)),
        "n_excluded_near_duplicate": len(extra & set(all_ids)),
        "exclude_list_used": args.exclude_list,
        "exclude_list_path": exclude_path,
        "n_ids_in_exclude_list": len(extra),
        "excluded_original_ids": sorted(excluded, key=lambda s: int(s)),
        "n_original_ids_kept": len(keep_ids),
        "n_images_kept": n_imgs,
        "n_empty_label_files": empty_label_files,
        "box_counts_kept": {NAMES[k]: v for k, v in sorted(cls.items())},
        "augs_per_original": dict(collections.Counter(
            len(have[o]) for o in keep_ids)),
        "folds": {},
        "governance": {
            "locked_original_ids_in_any_fold": 0,
            "group_unit": "original_id (all augmentations kept together)",
            "file_level_random_split": False,
            "images_transmitted": 0,
        },
    }
    for f in range(N_FOLDS):
        v = [o for o in keep_ids if fold_of[o] == f]
        t = [o for o in keep_ids if fold_of[o] != f]
        assert not (set(v) & set(t))
        assert not ((set(v) | set(t)) & excluded)
        report["folds"][f"fold{f}"] = {
            "train_original_ids": len(t), "val_original_ids": len(v),
            "train_images": sum(len(have[o]) for o in t),
            "val_images": sum(len(have[o]) for o in v),
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.dry_run:
        report["staged"] = False
        print(json.dumps(report, indent=2, ensure_ascii=False))
        with open(os.path.join(OUT_DIR, "folds_dryrun.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        return

    modes = collections.Counter()
    for f in range(N_FOLDS):
        for sp, ids in (("train", [o for o in keep_ids if fold_of[o] != f]),
                        ("val", [o for o in keep_ids if fold_of[o] == f])):
            ip = os.path.join(STAGE, f"fold{f}", sp, "images")
            lp = os.path.join(STAGE, f"fold{f}", sp, "labels")
            os.makedirs(ip, exist_ok=True)
            os.makedirs(lp, exist_ok=True)
            for oid in ids:
                for stem in have[oid]:
                    modes[link_or_copy(os.path.join(SRC_IMG, stem + ".jpg"),
                                       os.path.join(ip, stem + ".jpg"))] += 1
                    modes[link_or_copy(os.path.join(ALL_LABELS, stem + ".txt"),
                                       os.path.join(lp, stem + ".txt"))] += 1
        yml = os.path.join(STAGE, f"fold{f}", "dataset.yaml")
        with open(yml, "w", encoding="utf-8") as fh:
            fh.write("# locked-safe 3-class nailfold vessel detection\n")
            fh.write("# built by scripts/build_locked_safe_det_folds.py\n")
            fh.write(f"path: {os.path.join(STAGE, f'fold{f}')}\n")
            fh.write("train: train/images\nval: val/images\n\nnames:\n")
            for k, v in NAMES.items():
                fh.write(f"  {k}: {v}\n")

    # a machine-readable record of which original_id went to which fold, so a
    # later evaluation can prove a case was never trained on
    with open(os.path.join(OUT_DIR, "fold_assignment.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["original_id", "fold", "n_augmentations"])
        for oid in keep_ids:
            w.writerow([oid, fold_of[oid], len(have[oid])])

    report["staged"] = True
    report["stage_root"] = STAGE
    report["link_modes"] = dict(modes)
    with open(os.path.join(OUT_DIR, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
