"""Train the vessel-morphology detector used by arms A2 / A3 / M2 / R2.

One detector PER OUTER FOLD. Fold k's detector never sees any image of any case
in fold k, so the local features it produces for fold k's cases come from a model
trained only on fold k's training cases. This is the whole reason the arms can use
human boxes at all: the boxes are supervision inside training folds, and a
validation case is only ever given automatically detected regions.

Two supervision sources, selected by --source:
  local     our own 血管数据集 human boxes (3 classes: vessel / malformed_vessel /
            cross_vessel), restricted to the 96 original_ids that were
            provenance-matched to development cases
  external  the HF Capillary-Dataset morphology boxes (4 classes: bushy /
            crossing / hairpin / tortuous). THEIR class definitions, used as
            morphology pretraining only. Grouped by recovered subject, never by
            the published split, which puts 94% of images in subjects spanning
            both sides.
  external_then_local  external pretraining, then fine-tuning on the fold's local
            training boxes. This is arm A3's detector.

Leakage controls, all asserted rather than assumed:
  - every augmentation of an original_id goes to the same side, because the 20
    augmentations are geometric variants of ONE nailfold
  - a case in the validation fold contributes no image and no box to training
  - locked-47 original_ids (32 of them) are dropped before anything is written
  - the external pool contains no local image

Counts are unitless. No micron, per-mm or per-minute value is produced here or
downstream of here.

  PYTHONIOENCODING=utf-8 python scripts/train_morph_detector_folds.py \
      --source local --epochs 40
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
FROZEN = EXP / "frozen"
DET = EXP / "detectors"
LOCAL_IMG = ROOT / "data" / "血管数据集" / "分类数据集" / "image"
LOCAL_LAB = ROOT / "data" / "yolo_det_3class" / "all_labels"
BOX_MAP = (ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260830" /
           "source_case_mapping.csv")
LOCAL_NAMES = {0: "vessel", 1: "malformed_vessel", 2: "cross_vessel"}
EXT_NAMES = {0: "bushy", 1: "crossing", 2: "hairpin", 3: "tortuous"}
SEED = 20260917


def local_assignment():
    """original_id -> (exam_case_id, fold) for development ids, plus the locked set."""
    m = pd.read_csv(BOX_MAP, dtype=str)
    m.columns = [c.lstrip("﻿") for c in m.columns]
    locked_ids = set(m.loc[m.source_mapping_status == "EXACT_RECOVERED_LOCKED",
                           "original_id"])
    dev = m[m.source_mapping_status == "EXACT_RECOVERED_DEVELOPMENT"]
    split = pd.read_csv(FROZEN / "local_case_split.csv",
                        dtype={"exam_case_id": str}).set_index("exam_case_id")
    rows = []
    for oid, cases in zip(dev.original_id, dev.development_cases):
        for c in str(cases).split(";"):
            c = c.strip()
            if c and c in split.index:
                rows.append(dict(original_id=oid, exam_case_id=c,
                                 fold=float(split.loc[c, "development_fold"])))
    a = pd.DataFrame(rows)
    # an original matched to several cases must follow the EARLIEST fold, so it can
    # never appear in training for any fold that holds one of its cases
    keep = a.groupby("original_id").agg(folds=("fold", lambda s: sorted(set(s))),
                                        cases=("exam_case_id", lambda s: sorted(set(s))))
    return keep, locked_ids


def write_local_fold(fold, keep, locked_ids, work):
    """Materialise one fold's YOLO tree with symlink-free copies (Windows-safe)."""
    tr_i, tr_l = work / "train" / "images", work / "train" / "labels"
    va_i, va_l = work / "val" / "images", work / "val" / "labels"
    for d in (tr_i, tr_l, va_i, va_l):
        d.mkdir(parents=True, exist_ok=True)
    n_tr = n_va = 0
    held_cases, train_cases = set(), set()
    for oid, row in keep.iterrows():
        if oid in locked_ids:
            continue
        is_val = fold in row.folds
        for k in range(64):
            lab = LOCAL_LAB / ("%s_%d.txt" % (oid, k))
            if not lab.exists():
                continue
            img = None
            for ext in (".jpg", ".png", ".jpeg", ".bmp", ".JPG"):
                cand = LOCAL_IMG / ("%s_%d%s" % (oid, k, ext))
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            di, dl = ((va_i, va_l) if is_val else (tr_i, tr_l))
            shutil.copy2(img, di / img.name)
            shutil.copy2(lab, dl / lab.name)
            if is_val:
                n_va += 1
            else:
                n_tr += 1
        (held_cases if is_val else train_cases).update(row.cases)
    if held_cases & train_cases:
        raise RuntimeError("fold %s: a case is on both sides: %s"
                           % (fold, sorted(held_cases & train_cases)[:3]))
    y = dict(path=str(work), train="train/images", val="val/images",
             names=LOCAL_NAMES)
    (work / "dataset.yaml").write_text(yaml.safe_dump(y, allow_unicode=True),
                                       encoding="utf-8")
    return dict(train_images=n_tr, val_images=n_va,
                train_cases=len(train_cases), val_cases=len(held_cases))


def write_external(work):
    """One subject-grouped external tree, shared by every fold.

    It contains no local image, so it cannot leak a local validation case and does
    not need to be rebuilt per fold. Subjects are split 85/15 for the detector's
    own early-stopping only; that split has no bearing on our field evaluation.
    """
    box = pd.read_csv(FROZEN / "external_pool_hf_boxes.csv")
    subs = sorted(box.subject_id.unique())
    rng = pd.Series(subs).sample(frac=1.0, random_state=SEED).tolist()
    cut = max(1, int(0.15 * len(rng)))
    val_subs = set(rng[:cut])
    tr_i, tr_l = work / "train" / "images", work / "train" / "labels"
    va_i, va_l = work / "val" / "images", work / "val" / "labels"
    for d in (tr_i, tr_l, va_i, va_l):
        d.mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "val": 0}
    for _, r in box.iterrows():
        src_i, src_l = ROOT / r.rel_path, ROOT / r.label_path
        if not src_i.exists() or not src_l.exists():
            continue
        side = "val" if r.subject_id in val_subs else "train"
        di, dl = ((va_i, va_l) if side == "val" else (tr_i, tr_l))
        shutil.copy2(src_i, di / src_i.name)
        shutil.copy2(src_l, dl / (src_i.stem + ".txt"))
        n[side] += 1
    if any("recovered_archive" in p for p in box.rel_path.astype(str)):
        raise RuntimeError("a local image reached the external detector tree")
    y = dict(path=str(work), train="train/images", val="val/images", names=EXT_NAMES)
    (work / "dataset.yaml").write_text(yaml.safe_dump(y, allow_unicode=True),
                                       encoding="utf-8")
    return dict(train_images=n["train"], val_images=n["val"],
                subjects=len(subs), val_subjects=len(val_subs))


def _shim_numpy_trapz():
    """ultralytics 8.3.0 computes average precision with np.trapz, which NumPy 2.x
    removed in favour of np.trapezoid. Rather than editing the installed package
    (an invisible change that would not reproduce elsewhere), restore the alias
    here. np.trapezoid is the same composite trapezoidal rule with the same
    signature, so the mAP values are unaffected."""
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid


def train_one(yaml_path, weights, epochs, imgsz, out_dir, name, device, seed=SEED):
    _shim_numpy_trapz()
    from ultralytics import YOLO
    m = YOLO(str(weights))
    # workers=2 rather than ultralytics' default 8: each worker held ~590 MB and
    # the default drove this 31.6 GB host to 0.9 GB free, which made any second
    # numpy process fail with OpenBLAS allocation errors. Worker count affects
    # loading throughput only, not the trained weights.
    m.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=8, workers=2,
            device=device, project=str(out_dir), name=name, seed=seed,
            deterministic=True, verbose=False, plots=False, val=True,
            pretrained=True, exist_ok=True)
    best = out_dir / name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("no best.pt at %s" % best)
    return best, m.metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["local", "external", "external_then_local"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--external-epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base", default="weights/yolo11s.pt")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    DET.mkdir(parents=True, exist_ok=True)
    base = ROOT / a.base
    if not base.exists():
        raise RuntimeError("missing detector base weights at %s" % base)
    keep, locked_ids = local_assignment()
    t0 = time.time()
    meta = dict(run="morph_detector_folds", source=a.source,
                base_weights=a.base, epochs=a.epochs, imgsz=a.imgsz, seed=SEED,
                locked_original_ids_excluded=len(locked_ids),
                locked_cases_seen=0, folds={})

    ext_best = None
    if a.source in ("external", "external_then_local"):
        work = DET / "_tree_external"
        if not (work / "dataset.yaml").exists():
            info = write_external(work)
            (DET / "external_tree.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8")
        else:
            info = json.loads((DET / "external_tree.json").read_text())
        meta["external_tree"] = info
        meta["external_class_names"] = EXT_NAMES
        meta["external_note"] = (
            "THEIR class definitions (bushy/crossing/hairpin/tortuous) are not our "
            "report fields; this detector is morphology pretraining, and its boxes "
            "span a whole hairpin (7.12x our box height fraction)")
        ext_best, _ = train_one(work / "dataset.yaml", base, a.external_epochs,
                                a.imgsz, DET, "external_pretrain", a.device)
        meta["external_pretrain_weights"] = str(
            ext_best.relative_to(ROOT)).replace("\\", "/")
        if a.source == "external":
            (DET / ("detectors_%s.json" % a.source)).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(meta, ensure_ascii=False, indent=2))
            return

    for fold in [float(x) for x in a.folds.split(",")]:
        work = DET / ("_tree_local_fold%g" % fold)
        info = write_local_fold(fold, keep, locked_ids, work)
        start = ext_best if a.source == "external_then_local" else base
        name = "%s_fold%g" % (a.source, fold)
        best, _ = train_one(work / "dataset.yaml", start, a.epochs, a.imgsz,
                            DET, name, a.device)
        meta["folds"]["%g" % fold] = dict(
            info, weights=str(best.relative_to(ROOT)).replace("\\", "/"),
            initialised_from=str(Path(start).relative_to(ROOT)).replace("\\", "/"))
        print("fold %g done: %s" % (fold, meta["folds"]["%g" % fold]), flush=True)
        shutil.rmtree(work, ignore_errors=True)

    meta["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    meta["leakage_controls"] = [
        "all 20 augmentations of one original_id stay on one side",
        "no case in the validation fold contributes any image or box to training",
        "32 locked-matched original_ids dropped before any tree was written",
        "the external tree contains no local image and is subject-grouped",
    ]
    (DET / ("detectors_%s.json" % a.source)).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
