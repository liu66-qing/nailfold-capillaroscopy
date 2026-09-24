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


def per_image_stats(res, conf: float):
    """Counts and frame-relative geometry for one image."""
    b = res.boxes
    if b is None or len(b) == 0:
        return dict(n_total=0, n_vessel=0, n_malformed=0, n_cross=0,
                    ratio_malformed=np.nan, ratio_cross=np.nan,
                    box_w_frac=np.nan, box_h_frac=np.nan,
                    box_aspect=np.nan, box_diag_frac=np.nan)
    keep = b.conf.cpu().numpy() >= conf
    cls = b.cls.cpu().numpy()[keep].astype(int)
    xywhn = b.xywhn.cpu().numpy()[keep]
    n = len(cls)
    cnt = {v: int((cls == k).sum()) for k, v in CLS.items()}
    out = dict(n_total=n, n_vessel=cnt.get("vessel", 0),
               n_malformed=cnt.get("malformed", 0), n_cross=cnt.get("cross", 0))
    out["ratio_malformed"] = (out["n_malformed"] / n) if n else np.nan
    out["ratio_cross"] = (out["n_cross"] / n) if n else np.nan
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
    tot = g[["n_total", "n_vessel", "n_malformed", "n_cross"]].sum()
    tot.columns = ["pooled_" + c for c in tot.columns]
    tot["pooled_ratio_malformed"] = tot.pooled_n_malformed / tot.pooled_n_total.replace(0, np.nan)
    tot["pooled_ratio_cross"] = tot.pooled_n_cross / tot.pooled_n_total.replace(0, np.nan)
    tot["n_images"] = g.size()
    return pd.concat(parts + [tot], axis=1)


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
    missing = [f for f in folds if ("%g" % f) not in meta_in["folds"]]
    if missing:
        raise RuntimeError("no detector for folds %s" % missing)

    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in folds:
        w = ROOT / meta_in["folds"]["%g" % f]["weights"]
        model = YOLO(str(w))
        sub = prov[prov.development_fold == f]
        paths = [str(ROOT / "data" / p) for p in sub.image_path]
        for i in range(0, len(paths), 16):
            chunk = paths[i:i + 16]
            res = model.predict(chunk, imgsz=a.imgsz, conf=a.conf, device=a.device,
                                verbose=False)
            for r, p, cid in zip(res, chunk, sub.exam_case_id.iloc[i:i + 16]):
                rows.append(dict(exam_case_id=cid,
                                 image_path=Path(p).name,
                                 **per_image_stats(r, a.conf)))
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
        fold_to_detector={("%g" % f): meta_in["folds"]["%g" % f]["weights"]
                          for f in folds},
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
