#!/usr/bin/env python
"""PREREG revision 2: red-dot-verified pseudo-labels on Mendeley SSc images.

1. Inpaint the red/yellow overlay, upscale by SCALE to our pixel scale.
2. Run S0 (base-only segmenter), conf 0.05.
3. A red-dot cluster is "recalled" if any S0 instance mask covers >= 1 of its
   dots. Cluster recall is reported per image and overall; this is the direct
   measure of whether the segmenter misses SSc vessels.
4. Pseudo-labels per image:
     S0 instances covering a cluster (dot-verified detections)
   + S0 instances with conf >= 0.25 (kept so found vessels aren't taught as
     background)
   + for uncovered clusters, the SAM local mask (sam_fill) if its contrast is
     >= 0.66 (human polygon p10).
5. Training images: 1024x1024 tiles of the upscaled image (native scale, same
   as the base canvases; whole images would be downscaled ~2x at imgsz 1024),
   arm "inpaint" and arm "raw" (overlay left in place, same labels).

  python scripts/mendeley_pseudo_from_s0.py --s0 E:/nailfold_tmp/seg_runs/S0/weights/best.pt \
      --out E:/nailfold_tmp/mendeley_pseudo
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import mendeley_sam_pseudomasks as M  # noqa: E402


TILE = 1024


def tile_origins(n):
    return [0] if n <= TILE else [0, n - TILE]


def yolo_line(poly, w, h):
    p = np.asarray(poly, float)
    p[:, 0] = np.clip(p[:, 0] / w, 0, 1); p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
    return "0 " + " ".join("%.6f %.6f" % tuple(q) for q in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import SAM, YOLO
    seg, sam = YOLO(a.s0), SAM(M.SAM_W)
    for arm in ("inpaint", "raw"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, arm, sub), exist_ok=True)
    dots = pd.read_csv(M.DOTS); dots = dots[~dots.is_control]
    files = sorted(dots.file.unique())
    if a.limit:
        files = files[:a.limit]
    per, tot = [], dict(clusters=0, recalled=0, sam_fill=0, sam_reject=0,
                        inst_dot=0, inst_conf=0)
    for i, fn in enumerate(files):
        raw = cv2.imdecode(np.fromfile(os.path.join(M.MEND, fn), np.uint8), 1)
        ov = M.overlay_mask(raw)
        clean = cv2.inpaint(raw, ov, 3, cv2.INPAINT_TELEA)
        H0, W0 = raw.shape[:2]
        W, H = round(W0 * M.SCALE), round(H0 * M.SCALE)
        up_c = cv2.resize(clean, (W, H), interpolation=cv2.INTER_CUBIC)
        up_r = cv2.resize(raw, (W, H), interpolation=cv2.INTER_CUBIC)
        g = dots[dots.file == fn][["x", "y"]].to_numpy()
        lab = fcluster(linkage(g, "single"), M.LINK_PX, "distance") if len(g) > 1 else np.array([1])
        clusters = [g[lab == k] * M.SCALE for k in np.unique(lab)]
        r = seg.predict(up_c, conf=0.05, imgsz=2048, verbose=False, device=a.device)[0]
        polys, confs = [], []
        if r.masks is not None:
            polys = [np.asarray(p) for p in r.masks.xy]
            confs = r.boxes.conf.cpu().numpy().tolist()
        masks = []
        for p in polys:
            m = np.zeros((H, W), np.uint8)
            if len(p) >= 3:
                cv2.fillPoly(m, [p.astype(np.int32)], 1)
            masks.append(m.astype(bool))
        keep, covered = set(), []
        for c in clusters:
            xi = np.clip(c[:, 0].astype(int), 0, W - 1); yi = np.clip(c[:, 1].astype(int), 0, H - 1)
            hit = [j for j, m in enumerate(masks) if m[yi, xi].any()]
            covered.append(bool(hit)); keep.update(hit)
        tot["inst_dot"] += len(keep)
        for j, cf in enumerate(confs):
            if cf >= 0.25 and j not in keep:
                keep.add(j); tot["inst_conf"] += 1
        inst = [masks[j] for j in sorted(keep)]
        n_fill = 0
        for c, cov in zip(clusters, covered):
            if cov:
                continue
            m = M.local_mask(sam, up_c, c)
            if m is None or not m.any() or not (M.contrast(up_c, m) >= M.MIN_CONTRAST):
                tot["sam_reject"] += 1; continue
            inst.append(m); n_fill += 1
        tot["sam_fill"] += n_fill
        tot["clusters"] += len(clusters); tot["recalled"] += int(sum(covered))
        n_tiles = 0
        for ty in tile_origins(H):
            for tx in tile_origins(W):
                lines = []
                for m in inst:
                    t = m[ty:ty + TILE, tx:tx + TILE]
                    if t.sum() < 20:
                        continue
                    p = M.mask_to_poly(t)
                    if p is not None:
                        lines.append(yolo_line(p, t.shape[1], t.shape[0]))
                name = "mend_%03d_%d_%d" % (i, ty, tx)
                for arm, im in (("inpaint", up_c), ("raw", up_r)):
                    cv2.imencode(".jpg", im[ty:ty + TILE, tx:tx + TILE])[1].tofile(
                        os.path.join(a.out, arm, "images", name + ".jpg"))
                    open(os.path.join(a.out, arm, "labels", name + ".txt"), "w").write("\n".join(lines) + "\n")
                n_tiles += 1
        name = "mend_%03d" % i
        per.append(dict(name=name, file=fn, clusters=len(clusters), recalled=int(sum(covered)),
                        n_s0=len(polys), sam_fill=n_fill, n_labels=len(inst), tiles=n_tiles))
        if i % 50 == 0:
            print(i, len(files), tot, flush=True)
    df = pd.DataFrame(per)
    df.to_csv(os.path.join(a.out, "per_image.local.csv"), index=False)
    img_recall = (df.recalled / df.clusters)
    summary = dict(tot, images=len(df), cluster_recall=round(tot["recalled"] / tot["clusters"], 4),
                   image_recall_pct=[round(float(img_recall.quantile(q)), 3) for q in (.1, .25, .5, .75, .9)],
                   s0_inst_per_image_median=float(df.n_s0.median()),
                   labels_per_image_median=float(df.n_labels.median()))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
