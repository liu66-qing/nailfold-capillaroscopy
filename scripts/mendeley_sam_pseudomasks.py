#!/usr/bin/env python
"""Mendeley SSc red dots -> SAM2.1-b prompts -> single-vessel pseudo-mask crops.

Rules are fixed in artifacts/experiments/mendeley_sam_seg_20260927/PREREG.md
before running. Output mirrors the vascular-dataset crop format (one vessel
per crop, 10% margin) so the three segmenter arms differ only in the data
added:

  <out>/inpaint/{images,labels}   overlay pixels inpainted (arm S1)
  <out>/raw/{images,labels}       same crops, overlay left in place (arm S1n)

SAM always sees the inpainted image: the red dots must not become the object.
Mendeley filenames carry real names, so crops are written as m<i>_<k>.jpg and
the mapping stays in <out>/crop_map.local.csv (gitignored *.local.csv).

  python scripts/mendeley_sam_pseudomasks.py --out E:/nailfold_tmp/mendeley_sam
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEND = os.path.join(ROOT, "data", "The number of nail fold capillaries",
                    "The number of nail fold capillaries and nail fold bleedings "
                    "reflects the clinical manifestations of systemic sclerosis")
DOTS = os.path.join(ROOT, "artifacts/evidence/mendeley_overlay_20260919/red_dots.csv")
SAM_W = os.path.join(ROOT, "weights/sam2/sam2.1_b.pt")
SEG_ANN = os.path.join(ROOT, "artifacts/derived/vascular_dataset_governance_20260830/"
                       "train_ready/segmentation/annotations")
LINK_PX = 30.0
SCALE = 81.9 / 71.0          # our vessel NN spacing / Mendeley cluster NN spacing
MIN_COVER = 0.5
MAX_BBOX_FRAC = 0.15
MARGIN = 0.10
HALF, UP = 80, 4
MIN_CONTRAST = 0.66   # p10 of human polygon contrast, vascular dataset (PREREG rev 1)


def overlay_mask(bgr):
    B, G, R = [bgr[..., i].astype(np.int16) for i in range(3)]
    red = (R >= 190) & (G <= 90) & (B <= 90)
    yel = (R >= 190) & (G >= 190) & (B <= 110)
    return cv2.dilate((red | yel).astype(np.uint8), np.ones((5, 5), np.uint8))


def base_area_range():
    """p2..p98 of the vascular-dataset polygon areas (their pixel scale)."""
    areas = []
    for fn in os.listdir(SEG_ANN):
        d = json.load(open(os.path.join(SEG_ANN, fn), encoding="utf-8"))
        for s in d["shapes"]:
            p = np.asarray(s["points"], float)
            if len(p) >= 3:
                areas.append(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1))
                                       - np.dot(p[:, 1], np.roll(p[:, 0], 1))))
    return float(np.percentile(areas, 2)), float(np.percentile(areas, 98))


def contrast(img, m):
    """(ring mean - mask mean) / ring std: positive when the mask is darker than
    its 4 px surround, which is what a capillary on this background looks like."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
    if ring.sum() < 5 or m.sum() < 5:
        return np.nan
    return (g[ring].mean() - g[m].mean()) / (g[ring].std() + 1e-6)


def local_mask(sam, img, pts):
    H, W = img.shape[:2]
    cx, cy = pts.mean(0)
    x0 = int(np.clip(cx - HALF, 0, max(0, W - 2 * HALF)))
    y0 = int(np.clip(cy - HALF, 0, max(0, H - 2 * HALF)))
    crop = img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    big = cv2.resize(crop, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    pp = ((pts - [x0, y0]) * UP).tolist()
    r = sam(big, points=[pp], labels=[[1] * len(pp)], verbose=False)[0]
    if r.masks is None or not len(r.masks.data):
        return None
    small = cv2.resize(r.masks.data[0].cpu().numpy().astype(np.uint8),
                       (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = np.zeros((H, W), bool)
    m[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = small.astype(bool)
    return m


def mask_to_poly(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from ultralytics import SAM
    sam = SAM(SAM_W)

    lo, hi = base_area_range()
    for arm in ("inpaint", "raw"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, arm, sub), exist_ok=True)
    dots = pd.read_csv(DOTS)
    dots = dots[~dots.is_control]
    files = sorted(dots.file.unique())
    if a.limit:
        files = files[:a.limit]

    rows, stats = [], dict(clusters=0, kept=0, rej_cover=0, rej_area=0,
                           rej_bbox=0, rej_empty=0, rej_overlap=0,
                           rej_contrast=0)
    for i, fn in enumerate(files):
        raw = cv2.imdecode(np.fromfile(os.path.join(MEND, fn), np.uint8), 1)
        ov = overlay_mask(raw)
        clean = cv2.inpaint(raw, ov, 3, cv2.INPAINT_TELEA)
        H, W = raw.shape[:2]
        g = dots[dots.file == fn][["x", "y"]].to_numpy()
        lab = (fcluster(linkage(g, "single"), LINK_PX, "distance")
               if len(g) > 1 else np.array([1]))
        prompts = [g[lab == k] for k in np.unique(lab)]
        stats["clusters"] += len(prompts)
        # Whole-image prompting segments background blobs (PREREG revision 1),
        # so SAM sees a 160x160 window around the cluster, upsampled 4x.
        cand = []
        for k, p in enumerate(prompts):
            m = local_mask(sam, clean, p)
            if m is None or not m.any():
                stats["rej_empty"] += 1
                continue
            if contrast(clean, m) < MIN_CONTRAST:
                stats["rej_contrast"] += 1
                continue
            xi = np.clip(p[:, 0].round().astype(int), 0, W - 1)
            yi = np.clip(p[:, 1].round().astype(int), 0, H - 1)
            if m[yi, xi].mean() < MIN_COVER:
                stats["rej_cover"] += 1
                continue
            area = m.sum() * SCALE ** 2
            if not lo <= area <= hi:
                stats["rej_area"] += 1
                continue
            ys, xs = np.nonzero(m)
            if (np.ptp(xs) + 1) * (np.ptp(ys) + 1) > MAX_BBOX_FRAC * H * W:
                stats["rej_bbox"] += 1
                continue
            cand.append((int(m.sum()), k, m))
        # overlap: smaller mask wins
        cand.sort(key=lambda t: t[0])
        taken = np.zeros((H, W), bool)
        for area, k, m in cand:
            if (m & taken).sum() > 0.3 * area:
                stats["rej_overlap"] += 1
                continue
            taken |= m
            poly = mask_to_poly(m)
            if poly is None:
                stats["rej_empty"] += 1
                continue
            x0, y0, w, h = cv2.boundingRect(poly)
            mx, my = int(np.ceil(w * MARGIN)), int(np.ceil(h * MARGIN))
            X0, Y0 = max(0, x0 - mx), max(0, y0 - my)
            X1, Y1 = min(W, x0 + w + mx), min(H, y0 + h + my)
            name = "m%03d_%03d" % (i, k)
            sz = (max(1, round((X1 - X0) * SCALE)), max(1, round((Y1 - Y0) * SCALE)))
            pp = (poly - [X0, Y0]) * SCALE
            line = "0 " + " ".join("%.6f %.6f" % (x / sz[0], y / sz[1]) for x, y in
                                   np.clip(pp, 0, [sz[0], sz[1]]))
            for arm, src in (("inpaint", clean), ("raw", raw)):
                crop = cv2.resize(src[Y0:Y1, X0:X1], sz, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(a.out, arm, "images", name + ".jpg"), crop)
                open(os.path.join(a.out, arm, "labels", name + ".txt"), "w").write(line + "\n")
            rows.append(dict(crop=name, file=fn, cluster=k, n_dots=len(prompts[k]),
                             contrast=float(contrast(clean, m)),
                             area_px=area, w=sz[0], h=sz[1],
                             overlay_frac=float(ov[Y0:Y1, X0:X1].mean())))
            stats["kept"] += 1
        if i % 50 == 0:
            print(i, len(files), stats, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "crop_map.local.csv"), index=False)
    summary = dict(stats, files=len(files), area_range_ours=[lo, hi], scale=SCALE,
                   crop_w_median=float(df.w.median()), crop_h_median=float(df.h.median()),
                   dots_per_kept_median=float(df.n_dots.median()),
                   overlay_frac_in_crop_median=float(df.overlay_frac.median()))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
