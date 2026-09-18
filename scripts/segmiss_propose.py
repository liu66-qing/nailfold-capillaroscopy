#!/usr/bin/env python
"""Phase 1+2: class-agnostic mask proposals, then isolate YOLO's misses.

Circularity guard
-----------------
Proposals come from MobileSAM (class-agnostic, never trained on this project)
or, as fallback, classical vesselness (frangi/sato). Neither sees the failing
YOLO segmenter's output. The YOLO conf=0.05 detections are used ONLY as a
subtractive mask to decide which proposals are "missed", never to generate them.

locked-47
---------
The image index is filtered to development cases via development_fold, and the
intersection with locked_test is asserted empty at CASE level before any
inference runs. locked_cases_seen is recorded as 0.

No absolute scale
-----------------
All geometry in pixels / scale-free ratios. calibration_factors.json unused.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- vessel sanity

def vessel_like(img_bgr, mask, img_w, img_h):
    """Colour / size / shape sanity for a candidate mask.

    Nailfold capillaries are small, elongated, reddish structures. This filter
    is deliberately loose: it must NOT encode the canonical-hairpin prior that
    the YOLO model already over-learned, only reject background, glare, skin
    field and whole-image blobs.
    """
    area = float(mask.sum())
    if area < 60 or area > 0.06 * img_w * img_h:
        return None

    ys, xs = np.nonzero(mask)
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    if bw < 5 or bh < 5:
        return None
    # reject full-width / full-height slabs (skin field, illumination bands)
    if bw > 0.75 * img_w or bh > 0.75 * img_h:
        return None

    # redness: capillaries carry blood, so R should exceed G/B inside the mask
    m = mask.astype(bool)
    b, g, r = [img_bgr[..., i][m].astype(np.float32) for i in range(3)]
    if r.size == 0:
        return None
    redness = float(r.mean() - 0.5 * (g.mean() + b.mean()))
    # darker than the surrounding skin is also acceptable (dim/venous limbs)
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), 1).astype(bool) & ~m
    if ring.sum() < 10:
        return None
    inside = img_bgr[m].astype(np.float32).mean()
    outside = img_bgr[ring].astype(np.float32).mean()
    contrast = float(outside - inside)
    if redness < 2.0 and contrast < 4.0:
        return None
    return {"area": area, "bw": bw, "bh": bh,
            "redness": redness, "contrast": contrast}


# --------------------------------------------------------------- shape features
# identical descriptors to scripts/extract_instance_geometry.py so the miss
# features are directly comparable to the existing c05 geometry

def poly_shape(poly):
    if poly is None or len(poly) < 3:
        return None
    x, y = poly[:, 0].astype(float), poly[:, 1].astype(float)
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    d = np.diff(np.vstack([poly, poly[:1]]).astype(float), axis=0)
    perim = float(np.sqrt((d ** 2).sum(1)).sum())
    if area <= 0 or perim <= 0:
        return None
    pts = np.stack([x - x.mean(), y - y.mean()])
    cov = np.cov(pts)
    ev = np.linalg.eigvalsh(cov) if cov.shape == (2, 2) else np.array([1.0, 1.0])
    ev = np.clip(ev, 1e-9, None)
    elong = float(np.sqrt(ev[1] / ev[0]))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(poly.astype(float))
        hull_area = float(hull.volume)
        solidity = area / hull_area if hull_area > 0 else np.nan
        hd = np.diff(np.vstack([poly[hull.vertices],
                               poly[hull.vertices][:1]]).astype(float), axis=0)
        hull_perim = float(np.sqrt((hd ** 2).sum(1)).sum())
        convexity = hull_perim / perim if perim > 0 else np.nan
    except Exception:
        solidity, convexity = np.nan, np.nan
    bw = float(x.max() - x.min())
    bh = float(y.max() - y.min())
    return {"area": float(area), "perimeter": perim,
            "circularity": float(4 * np.pi * area / (perim ** 2)),
            "elongation": elong,
            "solidity": float(solidity) if solidity == solidity else np.nan,
            "convexity": float(convexity) if convexity == convexity else np.nan,
            "aspect": bw / bh if bh > 0 else np.nan,
            "extent": float(area / (bw * bh)) if bw > 0 and bh > 0 else np.nan,
            "cx": float(x.mean()), "cy": float(y.mean()), "bw": bw, "bh": bh}


def mask_to_poly(mask):
    c, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    big = max(c, key=cv2.contourArea)
    return big.reshape(-1, 2) if len(big) >= 3 else None


# ------------------------------------------------------------------- proposers

def propose_sam(sam, path, img_bgr, imgsz):
    """MobileSAM 'segment everything'. Class-agnostic: no vessel prior at all."""
    res = sam(path, imgsz=imgsz, verbose=False, device=DEVICE, retina_masks=True)[0]
    out = []
    if res.masks is None or res.masks.data is None:
        return out
    H, W = img_bgr.shape[:2]
    for m in res.masks.data.cpu().numpy():
        mm = m.astype(np.uint8)
        if mm.shape != (H, W):
            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(mm.astype(bool))
    return out


def propose_vesselness(img_bgr):
    """Fallback: frangi vesselness + connected components. Also class-agnostic."""
    from skimage.filters import frangi
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    v = frangi(1.0 - g, sigmas=range(1, 8, 2), black_ridges=False)
    if not np.isfinite(v).any():
        return []
    thr = float(np.nanpercentile(v, 99.0))
    bw = (v >= max(thr, 1e-12)).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(bw, 8)
    return [(lab == i) for i in range(1, n)]


# --------------------------------------------------------------- YOLO occupancy

def yolo_occupancy(model, path, conf, imgsz, shape):
    """Union mask of everything the failing segmenter already finds at conf=0.05.

    Used ONLY subtractively. max_det capped at 120: 300 at low conf OOMs.
    """
    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    res = model.predict(path, conf=conf, imgsz=imgsz, verbose=False,
                        device=DEVICE, max_det=120, retina_masks=True)[0]
    n = 0
    if res.masks is not None and res.masks.xy is not None:
        for poly in res.masks.xy:
            p = np.asarray(poly)
            if len(p) >= 3:
                cv2.fillPoly(occ, [p.astype(np.int32)], 1)
                n += 1
    return occ.astype(bool), n


SHAPE_KEYS = ["circularity", "elongation", "solidity", "convexity",
              "aspect", "extent", "area", "perimeter"]


def miss_row(misses, n_yolo, img_w, img_h):
    """Features describing ONLY the suspected misses.

    miss_density is the headline Phase 3 quantity: recovered vessels per
    already-detected vessel. If the miss hypothesis is right this must rise
    with malformation level.
    """
    row = {"n_miss": len(misses), "n_yolo": n_yolo,
           "img_w": img_w, "img_h": img_h,
           "miss_density": len(misses) / max(n_yolo, 1),
           "miss_share": len(misses) / max(n_yolo + len(misses), 1)}
    if not misses:
        for k in SHAPE_KEYS:
            row[f"miss_{k}_mean"] = np.nan
        row["miss_area_share"] = 0.0
        row["miss_frac_low_circ"] = np.nan
        return row
    for k in SHAPE_KEYS:
        v = np.array([m[k] for m in misses], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        row[f"miss_{k}_mean"] = float(v.mean())
        row[f"miss_{k}_std"] = float(v.std())
        row[f"miss_{k}_p90"] = float(np.percentile(v, 90))
    circ = np.array([m["circularity"] for m in misses], dtype=float)
    sol = np.array([m["solidity"] for m in misses], dtype=float)
    with np.errstate(invalid="ignore"):
        row["miss_frac_low_circ"] = float(np.nanmean(circ < 0.30))
        row["miss_frac_low_solidity"] = float(np.nanmean(sol < 0.80))
    ar = np.array([m["area"] for m in misses], dtype=float)
    row["miss_area_share"] = float(ar.sum() / (img_w * img_h))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/root/autodl-tmp/nailfold/artifacts/features/dinov2/index.csv")
    ap.add_argument("--manifest", default="/root/autodl-tmp/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv")
    ap.add_argument("--data-root", default="/root/autodl-tmp/nailfold/data")
    ap.add_argument("--weights-dir", default="/root/autodl-tmp/nailfold/artifacts/models")
    ap.add_argument("--sam", default="/root/autodl-tmp/segmiss_v1/mobile_sam.pt")
    ap.add_argument("--proposer", default="sam", choices=["sam", "vesselness"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iou-miss", type=float, default=0.10,
                    help="a proposal counts as MISSED if its overlap with the "
                         "YOLO union mask is below this fraction of its own area")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--save-polys", type=int, default=1)
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    os.makedirs(args.out, exist_ok=True)

    man = pd.read_csv(args.manifest)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man["evaluation_role"].astype(str)
                         .str.contains("locked", case=False), "exam_case_id"])
    dev_fold = (man.dropna(subset=["development_fold"])
                   .set_index("exam_case_id")["development_fold"].astype(int))
    assert not (set(dev_fold.index) & locked), "locked case carries a development_fold"

    idx = pd.read_csv(args.index)
    idx["exam_case_id"] = idx["exam_case_id"].astype(str)
    idx = idx[idx["exam_case_id"].isin(dev_fold.index)].reset_index(drop=True)
    assert not (set(idx["exam_case_id"]) & locked), "locked case reached the index"
    if args.limit:
        idx = idx.head(args.limit)
    idx["fold"] = idx["exam_case_id"].map(dev_fold).astype(int)
    print(f"images={len(idx)} cases={idx.exam_case_id.nunique()} locked_seen=0",
          flush=True)

    from ultralytics import YOLO
    sam = None
    if args.proposer == "sam":
        from ultralytics import SAM
        sam = SAM(args.sam)

    rows, polys, fails = [], [], []
    for k, grp in idx.groupby("fold", sort=True):
        wp = os.path.join(args.weights_dir, f"seg_vessel_fold{k}", "weights", "best.pt")
        model = YOLO(wp)
        print(f"[fold {k}] {len(grp)} images", flush=True)
        for n, (_, r) in enumerate(grp.iterrows(), 1):
            p = os.path.join(args.data_root, str(r["image_path"]).replace("\\", "/"))
            if not os.path.exists(p):
                fails.append({"image_path": r["image_path"], "why": "missing"})
                continue
            try:
                img = cv2.imread(p)
                if img is None:
                    fails.append({"image_path": r["image_path"], "why": "unreadable"})
                    continue
                H, W = img.shape[:2]
                occ, n_yolo = yolo_occupancy(model, p, args.conf, args.imgsz, (H, W))
                cands = (propose_sam(sam, p, img, args.imgsz)
                         if args.proposer == "sam" else propose_vesselness(img))
                misses = []
                for mk in cands:
                    a = float(mk.sum())
                    if a <= 0:
                        continue
                    # (b) NO overlap with any existing YOLO detection
                    if float((mk & occ).sum()) / a >= args.iou_miss:
                        continue
                    # (a) vessel-like sanity
                    if vessel_like(img, mk, W, H) is None:
                        continue
                    poly = mask_to_poly(mk)
                    sh = poly_shape(poly) if poly is not None else None
                    if sh is None:
                        continue
                    misses.append(sh)
                    if args.save_polys:
                        polys.append({"image_path": r["image_path"],
                                      "exam_case_id": r["exam_case_id"],
                                      "fold": int(k),
                                      "poly": poly.astype(int).tolist()})
            except Exception as e:
                fails.append({"image_path": r["image_path"], "why": repr(e)[:200]})
                continue
            row = miss_row(misses, n_yolo, W, H)
            row["exam_case_id"] = r["exam_case_id"]
            row["image_path"] = r["image_path"]
            row["fold"] = int(k)
            rows.append(row)
            if n % 100 == 0:
                print(f"  [fold {k}] {n}/{len(grp)}", flush=True)

    img_df = pd.DataFrame(rows)
    img_df.to_csv(os.path.join(args.out, "miss_image.csv"), index=False)
    num = img_df.select_dtypes(include=[np.number]).columns.drop(["fold"], errors="ignore")
    g = img_df.groupby("exam_case_id")
    case = g[list(num)].mean()
    case["n_images"] = g.size()
    case["fold"] = case.index.map(dev_fold)
    case.to_csv(os.path.join(args.out, "miss_case.csv"))
    if args.save_polys:
        with open(os.path.join(args.out, "ai_proposed_misses.jsonl"), "w") as f:
            for d in polys:
                f.write(json.dumps(d) + "\n")

    meta = {"proposer": args.proposer, "ran_locally": True,
            "third_party_api_used": False,
            "yolo_conf_for_occupancy": args.conf, "iou_miss": args.iou_miss,
            "imgsz": args.imgsz, "device": args.device,
            "n_images": int(len(img_df)), "n_cases": int(case.shape[0]),
            "locked_cases_seen": 0,
            "miss_per_image_mean": float(img_df.n_miss.mean()) if len(img_df) else 0.0,
            "yolo_per_image_mean": float(img_df.n_yolo.mean()) if len(img_df) else 0.0,
            "total_misses": int(img_df.n_miss.sum()) if len(img_df) else 0,
            "n_failures": len(fails), "failures": fails[:50],
            "circularity_guard": ("proposals from class-agnostic %s; YOLO masks "
                                  "used only subtractively" % args.proposer),
            "limitations": [
                "AI proposals are NOT gold standard; stored separately from human annotations",
                "development set only (186 cases); NOT product capability",
                "pixel-scale geometry only; calibration_factors.json unused",
            ]}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    DEVICE = 1
    main()
