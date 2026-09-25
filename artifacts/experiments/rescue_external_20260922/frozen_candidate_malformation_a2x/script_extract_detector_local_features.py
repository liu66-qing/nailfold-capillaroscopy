"""Case-level local morphology features from a per-fold detector.

For every development case, each of its images is run through the detector that
did NOT train on that case (fold k's images use fold k's detector), the boxes are
counted, and per-image statistics are aggregated to the case.

What this produces, per case
  vessel / malformed_vessel / cross_vessel counts, their ratios, and the case-level
  median, mean, p25 and p75 of each per-image quantity, plus box geometry in
  IMAGE-RELATIVE units only (fraction of frame width/height, aspect, and the
  normalised bounding-box diagonal as a size proxy).

Units. Every number here is either a count, a ratio, or a fraction of the frame.
No micron, per-mm or per-minute value is computed or emitted -- device calibration
is UNCALIBRATED, and a count over an undefined denominator is not a density.

Leakage. The fold each case belongs to decides which detector reads it, so no
case is ever read by a detector that saw it in training. The run asserts this
mapping and refuses to proceed if a case has no detector or if a locked id
appears anywhere.

  PYTHONIOENCODING=utf-8 python scripts/extract_detector_local_features.py \
      --detectors detectors_local.json --tag local
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
FROZEN = EXP / "frozen"
DET = EXP / "detectors"
OUT = EXP / "local_features"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
CLS = {0: "vessel", 1: "malformed", 2: "cross"}
QUANTS = ("median", "mean", "p25", "p75")


def per_image_stats(res, conf: float, cls_map: dict = None):
    """Counts and frame-relative geometry for one image.

    cls_map decides what the class indices MEAN. The local detector emits
    0/1/2 = vessel/malformed/cross, but the external-pretrained detector emits
    0/1/2/3 = bushy/crossing/hairpin/tortuous. Filing the external detector's
    class 0 under "n_vessel" would put its bushy count in the vessel column and
    drop class 3 entirely, and the resulting table would look perfectly normal.
    So the caller must pass the map that matches the checkpoint.
    """
    cls_map = CLS if cls_map is None else cls_map
    names = list(cls_map.values())
    b = res.boxes
    if b is None or len(b) == 0:
        out = dict(n_total=0, **{"n_" + v: 0 for v in names})
        out.update({"ratio_" + v: np.nan for v in names})
        out.update(box_w_frac=np.nan, box_h_frac=np.nan,
                   box_aspect=np.nan, box_diag_frac=np.nan)
        return out
    keep = b.conf.cpu().numpy() >= conf
    cls = b.cls.cpu().numpy()[keep].astype(int)
    xywhn = b.xywhn.cpu().numpy()[keep]
    n = len(cls)
    out = dict(n_total=n)
    for k, v in cls_map.items():
        out["n_" + v] = int((cls == k).sum())
    for v in names:
        out["ratio_" + v] = (out["n_" + v] / n) if n else np.nan
    if n:
        w, h = xywhn[:, 2], xywhn[:, 3]
        out["box_w_frac"] = float(np.median(w))
        out["box_h_frac"] = float(np.median(h))
        out["box_aspect"] = float(np.median(h / np.clip(w, 1e-6, None)))
        out["box_diag_frac"] = float(np.median(np.sqrt(w ** 2 + h ** 2)))
    else:
        out.update(box_w_frac=np.nan, box_h_frac=np.nan, box_aspect=np.nan,
                   box_diag_frac=np.nan)
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Case-level aggregation. Ratios are recomputed from pooled counts as well as
    averaged per image, because the two answer different questions and neither is
    obviously right for a clinical ratio read off a whole nailfold."""
    num = [c for c in df.columns if c not in ("exam_case_id", "image_path")]
    g = df.groupby("exam_case_id")
    parts = []
    for q in QUANTS:
        if q == "median":
            a = g[num].median()
        elif q == "mean":
            a = g[num].mean()
        elif q == "p25":
            a = g[num].quantile(0.25)
        else:
            a = g[num].quantile(0.75)
        a.columns = ["%s_%s" % (c, q) for c in a.columns]
        parts.append(a)
    # Derive the class names from the columns actually present rather than naming
    # them here, so an external checkpoint's 4 classes are pooled too instead of
    # being silently dropped by a hard-coded vessel/malformed/cross list.
    ncols = [c for c in ("n_total",) if c in df.columns] + \
            [c for c in df.columns if c.startswith("n_") and c != "n_total"]
    tot = g[ncols].sum()
    tot.columns = ["pooled_" + c for c in tot.columns]
    for c in ncols:
        if c == "n_total":
            continue
        tot["pooled_ratio_" + c[2:]] = \
            tot["pooled_" + c] / tot.pooled_n_total.replace(0, np.nan)
    tot["n_images"] = g.size()
    out = pd.concat(parts + [tot], axis=1)

    # A ratio over zero boxes is undefined, not zero, so per_image_stats writes NaN
    # and that is right at the image level. At the CASE level those NaNs have to be
    # resolved here rather than left for a downstream fillna, because the two cases
    # they cover are different things:
    #
    #   counts     -> 0 is the measurement. The detector looked and found nothing.
    #   ratios and -> still undefined for a case whose every image was empty. Filling
    #   geometry      0 would assert "0% malformed", which is a claim the data does
    #                 not support. Those stay NaN and carry an explicit flag column.
    #
    # The flag is what lets the evaluator distinguish "no vessels detected" from "a
    # missing row", instead of both arriving as a zero.
    # Count columns are exactly the ones whose name starts n_ or pooled_n_ -- no
    # class name appears here, so this stays correct for a 4-class external
    # checkpoint. n_images is a count too and is already non-null.
    cnt_cols = [c for c in out.columns
                if c.startswith("pooled_n_") or c.startswith("n_")]
    out[cnt_cols] = out[cnt_cols].fillna(0)
    out["no_detection_in_any_image"] = (out.get("pooled_n_total", 0) == 0).astype(int)
    out["frac_images_with_no_detection"] = (
        df.assign(empty=(df.n_total == 0).astype(float))
          .groupby("exam_case_id").empty.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", required=True,
                    help="detectors_*.json written by train_morph_detector_folds")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="fixed and declared up front; not tuned per field")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    meta_in = json.loads((DET / a.detectors).read_text(encoding="utf-8"))
    prov = pd.read_csv(FROZEN / "local_image_provenance.csv",
                       dtype={"exam_case_id": str})
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str})
    locked = set(man.loc[man.development_fold.isna(), "exam_case_id"])
    if set(prov.exam_case_id) & locked:
        raise RuntimeError("locked case in the provenance table")
    folds = sorted(prov.development_fold.unique())
    # The external-only detector is ONE checkpoint, not five: it is trained purely
    # on external images, reads no local image, label or fold, and therefore cannot
    # leak into any local validation fold. So a single checkpoint legitimately
    # serves every fold here -- the same argument already applied to the
    # external-only domain adaptation in the whole-image ladder. Any manifest that
    # DID see local images must still carry one detector per fold, so this fallback
    # is allowed only for source == "external", and the source is checked, not the
    # emptiness of the folds dict.
    single = None
    if meta_in["source"] == "external":
        single = meta_in.get("external_pretrain_weights")
        if not single:
            raise RuntimeError("external manifest has no external_pretrain_weights")
        if meta_in["folds"]:
            raise RuntimeError("external manifest unexpectedly carries per-fold "
                               "detectors: %s" % sorted(meta_in["folds"]))
    else:
        missing = [f for f in folds if ("%g" % f) not in meta_in["folds"]]
        if missing:
            raise RuntimeError("no detector for folds %s" % missing)

    # Which class map applies is a property of the checkpoint, so take it from the
    # manifest rather than assuming. An external-only checkpoint predicts THEIR four
    # classes; external_then_local was fine-tuned on our boxes and predicts ours.
    if meta_in["source"] == "external":
        cls_map = {int(k): v for k, v in meta_in["external_class_names"].items()}
    else:
        cls_map = dict(CLS)
    print("class map for %s: %s" % (meta_in["source"], cls_map), flush=True)

    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in folds:
        w = ROOT / (single if single else meta_in["folds"]["%g" % f]["weights"])
        model = YOLO(str(w))
        # The checkpoint must predict the same NUMBER of classes on the same
        # indices as the map, or the columns are mislabelled in a way no
        # downstream check would catch -- an external checkpoint's class 0 landing
        # in n_vessel produces a table that looks entirely normal.
        #
        # Compared on indices, not on strings: the trainer's LOCAL_NAMES says
        # malformed_vessel / cross_vessel and this module's CLS says malformed /
        # cross. Same indices, same meaning, two spellings; a name equality check
        # rejects a correct pairing.
        got = {int(k): str(v) for k, v in (getattr(model, "names", None) or {}).items()}
        if got and sorted(got) != sorted(cls_map):
            raise RuntimeError("checkpoint %s predicts classes %s, manifest map is "
                               "%s" % (w.name, got, cls_map))
        sub = prov[prov.development_fold == f]
        paths = [str(ROOT / "data" / p) for p in sub.image_path]
        for i in range(0, len(paths), 16):
            chunk = paths[i:i + 16]
            res = model.predict(chunk, imgsz=a.imgsz, conf=a.conf, device=a.device,
                                verbose=False)
            for r, p, cid in zip(res, chunk, sub.exam_case_id.iloc[i:i + 16]):
                rows.append(dict(exam_case_id=cid,
                                 image_path=Path(p).name,
                                 **per_image_stats(r, a.conf, cls_map)))
        print("fold %g: %d images through %s"
              % (f, len(paths), Path(w).parent.parent.name), flush=True)
        del model
        torch.cuda.empty_cache()

    per_img = pd.DataFrame(rows)
    case = aggregate(per_img).reset_index()
    if case.exam_case_id.duplicated().any():
        raise RuntimeError("case-level table is not unique")
    if set(case.exam_case_id) & locked:
        raise RuntimeError("a locked case reached the feature table")
    per_img.to_csv(out / "per_image.csv", index=False, encoding="utf-8-sig")
    case.to_csv(out / "case_features.csv", index=False, encoding="utf-8-sig")
    meta = dict(
        run="detector_local_features", tag=a.tag,
        detector_manifest=a.detectors,
        detector_source=meta_in["source"],
        conf_threshold=a.conf, imgsz=a.imgsz,
        cases=int(case.exam_case_id.nunique()), images=int(len(per_img)),
        feature_columns=[c for c in case.columns if c != "exam_case_id"],
        fold_to_detector={("%g" % f): (single if single
                                       else meta_in["folds"]["%g" % f]["weights"])
                          for f in folds},
        class_map={str(k): v for k, v in cls_map.items()},
        one_checkpoint_for_all_folds=bool(single),
        one_checkpoint_justification=(
            None if not single else
            "this detector trained only on external images -- zero local images, "
            "labels or folds -- so it cannot leak into any local validation fold "
            "and one checkpoint legitimately serves every fold"),
        locked_cases_seen=0,
        units=dict(counts="unitless box counts",
                   ratios="unitless fractions of detected boxes",
                   geometry="fractions of the frame, never micrometres"),
        forbidden_outputs=["micrometres", "per-mm density", "per-minute rates"],
        limitations=[
            "counts depend on the detector's confidence threshold, fixed at %.2f "
            "before any field result was seen and not tuned per field" % a.conf,
            "a box count over an undefined field of view is not a density; "
            "capillary_count is evaluated against the report's printed band only",
            "each case is read by the detector of a fold that excluded it, so these "
            "features are out-of-fold by construction",
        ])
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
