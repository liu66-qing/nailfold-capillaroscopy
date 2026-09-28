#!/usr/bin/env python
"""Build the shared base dataset for the segmenter arms (PREREG revision 2).

  vascular-dataset single-vessel crops, pasted at NATIVE scale onto 1024x1024
    canvases (background = median border colour of each crop) so the segmenter
    sees vessels at the size they have in our 1024x768 full-field images
  + ANFC-coco train 257 images, Roboflow's 640x640 stretch undone to 1024x768,
    all six classes merged into one "vessel" class

Validation = crops whose file number % 10 == 0, and ANFC-coco valid (whose 42
subjects all appear in train -- reported, not hidden).

  python scripts/mendeley_seg_build_base.py --out E:/nailfold_tmp/seg_base
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = os.path.join(ROOT, "artifacts/derived/vascular_dataset_governance_20260830/"
                   "train_ready/segmentation")
ANFC = os.path.join(ROOT, "data/ANFC-THU.v1i.coco")
CANVAS = 1024
GAP = 6


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), 1)


def imwrite(p, im):
    cv2.imencode(".jpg", im)[1].tofile(p)


def yolo_line(poly, w, h):
    p = np.asarray(poly, float)
    p[:, 0] = np.clip(p[:, 0] / w, 0, 1)
    p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
    return "0 " + " ".join("%.6f %.6f" % tuple(q) for q in p)


def load_crops():
    out = []
    for fn in sorted(os.listdir(os.path.join(SEG, "annotations"))):
        d = json.load(open(os.path.join(SEG, "annotations", fn), encoding="utf-8"))
        im = imread(os.path.join(SEG, "images", fn[:-5] + ".jpg"))
        if im is None:
            continue
        polys = [np.asarray(s["points"], float) for s in d["shapes"] if len(s["points"]) >= 3]
        if polys:
            out.append((int(fn[:-5]), im, polys))
    return out


def pack(crops, out_dir, prefix, rng):
    """Shelf-pack crops onto canvases; each crop keeps its own polygons."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    order = rng.permutation(len(crops))
    n_canvas, i = 0, 0
    while i < len(order):
        canvas = np.zeros((CANVAS, CANVAS, 3), np.uint8)
        filled = np.zeros((CANVAS, CANVAS), bool)
        lines, x, y, shelf = [], GAP, GAP, 0
        placed = []
        while i < len(order):
            _, im, polys = crops[order[i]]
            h, w = im.shape[:2]
            if x + w + GAP > CANVAS:
                x, y, shelf = GAP, y + shelf + GAP, 0
            if y + h + GAP > CANVAS:
                break
            canvas[y:y + h, x:x + w] = im
            filled[y:y + h, x:x + w] = True
            placed.append(im)
            for p in polys:
                lines.append(yolo_line(p + [x, y], CANVAS, CANVAS))
            x += w + GAP
            shelf = max(shelf, h)
            i += 1
        border = np.concatenate([np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
                                 for c in placed])
        canvas[~filled] = np.median(border, 0).astype(np.uint8)
        name = "%s_%03d" % (prefix, n_canvas)
        imwrite(os.path.join(out_dir, "images", name + ".jpg"), canvas)
        open(os.path.join(out_dir, "labels", name + ".txt"), "w").write("\n".join(lines) + "\n")
        n_canvas += 1
    return n_canvas


def anfc(split, out_dir):
    d = json.load(open(os.path.join(ANFC, split, "_annotations.coco.json")))
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    n = 0
    for im in d["images"]:
        img = imread(os.path.join(ANFC, split, im["file_name"]))
        img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_CUBIC)
        lines = []
        for a in by.get(im["id"], []):
            for s in a.get("segmentation") or []:
                if isinstance(s, list) and len(s) >= 6:
                    p = np.asarray(s, float).reshape(-1, 2)
                    lines.append(yolo_line(p, im["width"], im["height"]))
        name = "anfc_" + im["file_name"].split("_jpg")[0]
        imwrite(os.path.join(out_dir, "images", name + ".jpg"), img)
        open(os.path.join(out_dir, "labels", name + ".txt"), "w").write("\n".join(lines) + "\n")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20260927)
    crops = load_crops()
    tr = [c for c in crops if c[0] % 10 != 0]
    va = [c for c in crops if c[0] % 10 == 0]
    info = dict(crops_train=len(tr), crops_val=len(va))
    info["canvas_train"] = pack(tr, os.path.join(a.out, "train"), "vd", rng)
    info["canvas_val"] = pack(va, os.path.join(a.out, "val"), "vd", rng)
    info["anfc_train"] = anfc("train", os.path.join(a.out, "train"))
    info["anfc_val"] = anfc("valid", os.path.join(a.out, "val"))
    json.dump(info, open(os.path.join(a.out, "build.json"), "w"), indent=2)
    print(info)


if __name__ == "__main__":
    main()
