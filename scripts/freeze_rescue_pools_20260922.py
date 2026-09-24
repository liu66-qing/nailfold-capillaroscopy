"""Step 1 of the rescue round: freeze the training pools and the case-level split.

Nothing is trained here. This writes the manifests every later arm must load, so
no arm can quietly change its own pool, split, or exclusion list after seeing a
result.

Frozen artefacts
  local_case_split.csv          186 development cases, development_fold, archive
  locked_exclusion.csv          the 47 locked ids, for assert-only use
  local_image_provenance.csv    every development image -> case, fold, archive
  external_pool_hf.csv          HF unlabelled nailfold images, subject-grouped
  external_pool_hf_boxes.csv    HF morphology boxes, subject-grouped
  external_pool_mendeley.csv    Mendeley images, marker flag, opaque subject id
  local_boxes_cases.csv         our human boxes mapped to development cases
  unauditable_external.csv      external samples with no recoverable subject id

Governance carried in from earlier rounds rather than re-derived:
  - HF subject recovery (90 subjects; 94% of images sit in subjects that span the
    published train/val split) from hf_capillary_audit_20260920.
  - Mendeley overlay measurement (yellow = device caliper strokes, red = contour
    point sequences) from mendeley_overlay_20260919. Marker-bearing images are
    flagged and BANNED from clarity / blood_color / exudation training.
  - Our own box provenance (96 of 582 original_ids matched to development cases,
    32 to locked) from vascular_dataset_governance_20260830.

Hard rules encoded here
  - locked-47 never enters any pool; the run aborts if one appears anywhere.
  - Mendeley filenames embed apparent personal names, so subjects are re-keyed to
    opaque salted ids and NO name is written to any output.
  - HF diabetes labels are external-endpoint metadata only, never a disease label
    for our patients.
  - The published HF train/val split is not used; grouping is by subject.

  PYTHONIOENCODING=utf-8 python scripts/freeze_rescue_pools_20260922.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "experiments" / "rescue_external_20260922" / "frozen"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
DEV_INDEX = ROOT / ".tmp_probe" / "feat" / "dinov2" / "index.csv"
HF = ROOT / "data" / "Capillary-Dataset"
MEND_AUDIT = ROOT / "artifacts" / "evidence" / "mendeley_overlay_20260919"
MEND = (ROOT / "data" / "The number of nail fold capillaries" /
        "The number of nail fold capillaries and nail fold bleedings reflects "
        "the clinical manifestations of systemic sclerosis")
HF_AUDIT = ROOT / "artifacts" / "evidence" / "hf_capillary_audit_20260920"
BOX_MAP = (ROOT / "artifacts" / "audits" / "vascular_dataset_governance_20260830" /
           "source_case_mapping.csv")
BOX_LABELS = ROOT / "data" / "yolo_det_3class" / "all_labels"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".BMP", ".JPG", ".PNG"}
SALT = "rescue_external_20260922"
MARKER_COLS = ("n_red_dots", "n_yellow_strokes")


def opaque(s: str) -> str:
    return hashlib.sha256((SALT + "|" + s).encode("utf-8")).hexdigest()[:12]


def local_split():
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str})
    dev = man[man.development_fold.notna()].copy()
    locked = man[man.development_fold.isna()].copy()
    if len(dev) != 186 or len(locked) != 47:
        raise RuntimeError("expected 186/47, got %d/%d" % (len(dev), len(locked)))
    split = dev[["exam_case_id", "development_fold", "archive"]].copy()
    ix = pd.read_csv(DEV_INDEX, dtype={"exam_case_id": str})
    leak = set(ix.exam_case_id) & set(locked.exam_case_id)
    if leak:
        raise RuntimeError("locked case in the development image index: %s"
                           % sorted(leak)[:5])
    prov = ix[["exam_case_id", "image_path"]].merge(split, on="exam_case_id",
                                                    how="left")
    if prov.development_fold.isna().any():
        raise RuntimeError("image without a fold: %s"
                           % prov[prov.development_fold.isna()].head().to_dict("records"))
    if prov.image_path.astype(str).str.contains(r"rep[_0-9]").any():
        raise RuntimeError("report scan present in the development image index")
    prov["stem"] = prov.image_path.map(lambda p: Path(p).stem)
    return split, locked[["exam_case_id"]], prov


def hf_pool():
    """Unlabelled nailfold images plus morphology boxes, grouped by subject."""
    rows = []
    for grp, sub in (("dia", "diabetic_patients"), ("hea", "healthy_subjects")):
        base = HF / "classification_original" / sub
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix in IMG_EXT:
                    rows.append(dict(
                        source="hf_classification_original", group=grp,
                        subject_dir=d.name, subject_id="hf_%s_%s" % (grp, d.name),
                        rel_path=str(f.relative_to(ROOT)).replace("\\", "/"),
                        external_endpoint=("diabetic" if grp == "dia" else "healthy"),
                        auditable_subject=True))
    unl = pd.DataFrame(rows)

    pmap = pd.read_csv(HF_AUDIT / "dia_image_to_patient_map.csv", dtype=str)
    pcol = [c for c in pmap.columns if "patient" in c.lower() or "subject" in c.lower()]
    icol = [c for c in pmap.columns
            if c.lower() in ("img", "image", "file", "filename", "query")
            or "image" in c.lower() or "file" in c.lower() or "query" in c.lower()]
    if not pcol or not icol:
        raise RuntimeError("unexpected columns in the HF subject map: %s"
                           % pmap.columns.tolist())
    pmap = pmap.rename(columns={icol[0]: "box_image", pcol[0]: "subject_dir"})
    pmap["box_image"] = pmap.box_image.map(lambda s: Path(str(s)).name)

    brows = []
    for sp in ("train", "val", "test"):
        idir = HF / "Morphology_detection" / "images" / sp
        ldir = HF / "Morphology_detection" / "labels" / sp
        for f in sorted(idir.iterdir()):
            if not f.is_file() or f.suffix not in IMG_EXT:
                continue
            lab = ldir / (f.stem + ".txt")
            n, cls = 0, {}
            if lab.exists():
                for line in lab.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        n += 1
                        cls[parts[0]] = cls.get(parts[0], 0) + 1
            brows.append(dict(
                source="hf_morphology_detection", published_split=sp,
                group=("dia" if f.name.startswith("dia") else "hea"),
                image=f.name,
                rel_path=str(f.relative_to(ROOT)).replace("\\", "/"),
                label_path=str(lab.relative_to(ROOT)).replace("\\", "/"),
                n_boxes=n, **{("cls_%s" % k): v for k, v in sorted(cls.items())}))
    box = pd.DataFrame(brows).fillna(0)
    box = box.merge(pmap[["box_image", "subject_dir"]].drop_duplicates("box_image"),
                    left_on="image", right_on="box_image", how="left")
    sid = []
    for s, i in zip(box.subject_dir, box.image):
        if isinstance(s, str) and s == s:
            sid.append("hf_dia_%s" % s)
        elif i.startswith("hea_") and "-" in i:
            # hea_<room>-<subject>_<frame>.jpg : the middle token is the subject
            sid.append("hf_hea_%s" % i.split("_")[1])
        else:
            sid.append(None)
    box["subject_id"] = sid
    box["auditable_subject"] = box.subject_id.notna()
    box["subject_id"] = [s if s else "hf_unmatched_%s" % opaque(i)
                         for s, i in zip(box.subject_id, box.image)]
    box["published_split_used_for_grouping"] = False
    return unl, box.drop(columns=["box_image"])


def mendeley_pool():
    per = pd.read_csv(MEND_AUDIT / "per_image.csv")
    per["subject_id"] = ["mend_%s" % opaque(str(p)) for p in per.person]
    per["exam_id"] = ["mendx_%s" % opaque(str(e)) for e in per.exam]
    per["has_marker"] = per[list(MARKER_COLS)].fillna(0).sum(axis=1) > 0
    per["rel_path"] = [str((MEND / f).relative_to(ROOT)).replace("\\", "/")
                       for f in per.file]
    # The Mendeley filenames embed apparent personal names, so the path itself is
    # identifying and must not be written to a committed artefact. The frozen pool
    # carries an opaque image_id only; the id -> path resolution is written to a
    # separate local-only file that training reads and git never sees.
    per["image_id"] = ["mendi_%s" % opaque(str(f)) for f in per.file]
    pathmap = per[["image_id", "rel_path"]].copy()
    out = per[["image_id", "subject_id", "exam_id", "w", "h", "n_red_dots",
               "n_yellow_strokes", "marker_px_fraction", "has_marker",
               "is_control", "n_panels"]].copy()
    out["appearance_training_allowed"] = ~out.has_marker
    out["appearance_ban_reason"] = [
        "device caliper strokes / contour points present: colour and sharpness are "
        "altered, so clarity, blood_color and exudation must not train on it"
        if m else "" for m in out.has_marker]
    out["is_new_patient_for_us"] = False
    out["count_definition_aligned_with_ours"] = False
    # every text column must be an opaque id or a fixed sentence written here; a
    # name can only arrive via a path or a raw filename, and neither survives above
    for col in ("image_id", "subject_id", "exam_id"):
        if not out[col].astype(str).str.fullmatch(r"(mendi|mend|mendx)_[0-9a-f]{12}").all():
            raise RuntimeError("column %s is not fully opaque" % col)
    if any(c in out.columns for c in ("file", "person", "exam", "rel_path")):
        raise RuntimeError("an identifying column survived into the frozen pool")
    return out, pathmap


def local_boxes():
    m = pd.read_csv(BOX_MAP, dtype=str)
    m.columns = [c.lstrip("﻿") for c in m.columns]
    dev = m[m.source_mapping_status == "EXACT_RECOVERED_DEVELOPMENT"]
    lk = m[m.source_mapping_status == "EXACT_RECOVERED_LOCKED"]
    rows = []
    for oid, cases in zip(dev.original_id, dev.development_cases):
        augs, nb, cls = [], 0, {}
        for k in range(64):
            p = BOX_LABELS / ("%s_%d.txt" % (oid, k))
            if not p.exists():
                continue
            augs.append(p.name)
            for ln in p.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 5:
                    nb += 1
                    cls[parts[0]] = cls.get(parts[0], 0) + 1
        for c in str(cases).split(";"):
            c = c.strip()
            if c:
                rows.append(dict(original_id=oid, exam_case_id=c,
                                 n_augmentations=len(augs), n_boxes_total=nb,
                                 **{("cls_%s" % k): v for k, v in sorted(cls.items())}))
    return pd.DataFrame(rows).fillna(0), sorted(set(lk.original_id))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split, locked, prov = local_split()
    locked_ids = set(locked.exam_case_id)
    unl, box = hf_pool()
    mend, mend_paths = mendeley_pool()
    lbox, locked_box_originals = local_boxes()

    if set(lbox.exam_case_id) & locked_ids:
        raise RuntimeError("a locked case reached the local box pool")
    for name, df in (("hf_unlabelled", unl), ("hf_boxes", box),
                     ("mendeley_pathmap", mend_paths)):
        if df.rel_path.astype(str).str.contains("recovered_archive").any():
            raise RuntimeError("an internal archive path reached the %s pool" % name)

    split.to_csv(OUT / "local_case_split.csv", index=False, encoding="utf-8-sig")
    locked.to_csv(OUT / "locked_exclusion.csv", index=False, encoding="utf-8-sig")
    prov.to_csv(OUT / "local_image_provenance.csv", index=False, encoding="utf-8-sig")
    unl.to_csv(OUT / "external_pool_hf.csv", index=False, encoding="utf-8-sig")
    box.to_csv(OUT / "external_pool_hf_boxes.csv", index=False, encoding="utf-8-sig")
    mend.to_csv(OUT / "external_pool_mendeley.csv", index=False, encoding="utf-8-sig")
    # local-only: resolves opaque image ids back to paths that contain apparent
    # personal names. Named *.local.csv so .gitignore keeps it out of history.
    mend_paths.to_csv(OUT / "mendeley_pathmap.local.csv", index=False,
                      encoding="utf-8-sig")
    lbox.to_csv(OUT / "local_boxes_cases.csv", index=False, encoding="utf-8-sig")
    un = box.loc[~box.auditable_subject, ["image", "rel_path", "subject_id"]].copy()
    un["reason"] = "no subject recoverable by the earlier image-matching pass"
    un.to_csv(OUT / "unauditable_external.csv", index=False, encoding="utf-8-sig")

    meta = dict(
        run="rescue_external_pools_frozen",
        frozen_at_utc=pd.Timestamp.utcnow().isoformat(),
        compute=dict(
            local="RTX 5070, torch 2.11+cu128, timm; encoder adaptation and all "
                  "feature extraction run here",
            server="ssh -p 14170 connect.westc.seetacloud.com, /root/miniconda3/"
                   "envs/nfc/bin/python, ultralytics 8.3.0 + yolo11s/m present but "
                   "NO GPU (nvidia-smi: No devices found), so it can host CPU-only "
                   "steps and storage, not detector training"),
        local_data=dict(dev_cases=int(len(split)), dev_images=int(len(prov)),
                        folds=sorted(split.development_fold.unique().tolist()),
                        archives=split.archive.value_counts().to_dict(),
                        locked_cases_excluded=int(len(locked)), locked_cases_seen=0),
        hf_unlabelled=dict(images=int(len(unl)),
                           subjects=int(unl.subject_id.nunique()),
                           by_group=unl.group.value_counts().to_dict(),
                           use="unlabelled domain adaptation only",
                           diabetes_label_use="external endpoint metadata only; "
                                              "never a disease label for our patients"),
        hf_boxes=dict(images=int(len(box)), boxes=int(box.n_boxes.sum()),
                      subjects=int(box.subject_id.nunique()),
                      auditable_subject_images=int(box.auditable_subject.sum()),
                      published_split_used=False,
                      grouping="by recovered subject; the published train/val split "
                               "puts 94% of images in subjects spanning both sides",
                      class_semantics="bushy/crossing/hairpin/tortuous are THEIR "
                                      "definitions, not our report fields; used as "
                                      "morphology pretraining signal only",
                      box_definition_gap="their boxes span a whole hairpin (height "
                                         "fraction 7.12x ours), so counts are not "
                                         "transferable without alignment"),
        mendeley=dict(images=int(len(mend)), subjects=int(mend.subject_id.nunique()),
                      exams=int(mend.exam_id.nunique()),
                      marker_bearing=int(mend.has_marker.sum()),
                      appearance_training_allowed=int(
                          mend.appearance_training_allowed.sum()),
                      subjects_rekeyed_to_opaque_ids=True, names_written=0,
                      is_new_patient_for_us=False),
        local_boxes=dict(original_ids=int(lbox.original_id.nunique()),
                         cases=int(lbox.exam_case_id.nunique()),
                         boxes=int(lbox.n_boxes_total.sum()),
                         locked_original_ids_excluded=len(locked_box_originals),
                         use="supervision inside TRAINING folds only; a validation "
                             "case is never given a human box as input"),
        rules=[
            "locked-47 is absent from every pool and was never read",
            "external data may enter training-fold pretraining / adaptation only",
            "Mendeley marker-bearing images are banned from clarity, blood_color "
            "and exudation training",
            "no absolute micron, per-mm or per-minute value is produced anywhere",
            "external endpoint accuracy is never our field capability",
        ])
    (OUT / "frozen_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
