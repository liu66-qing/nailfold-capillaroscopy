"""Resolve what the Mendeley overlay markers ACTUALLY are, so we can decide which
of our fields they can supervise.

Three questions I have not yet answered correctly:

Q1. Are the yellow components hollow RECTANGLES (LabelImg bounding boxes) or
    diagonal polylines (measurement/boundary strokes)? I earlier eyeballed an
    ASCII render and said "polyline", but a hollow rectangle of 174x73 px has
    fill = perimeter/area = 0.039, which is exactly the 0.04 I measured. The two
    hypotheses are NOT distinguishable by fill. They ARE distinguishable by the
    row/column projection profile: a hollow rectangle has two rows whose sum
    equals the box width and all other rows sum to 2; a diagonal stroke has all
    rows summing to ~1-2. This matters enormously: if they are boxes, this
    dataset ships the paper's LabelImg annotations burned into pixels.

Q2. Is red NFC (capillaries) or NFB (bleedings)? Not answerable from colour, but
    it IS answerable from count statistics. The paper's own normal range for
    capillary density is 7-9/mm and SSc nailfold bleedings are counted in single
    digits per nail. A per-image distribution with median 28 and only 3 zeros
    behaves like a dense structure, not like a rare event. Compute the full
    distribution and the zero-inflation, and compare the two hypotheses
    explicitly instead of leaving it "unresolved".

Q3. Do the two marker classes co-locate? If yellow are boxes and red dots fall
    INSIDE them, they annotate the same objects (box+centre). If red dots fall
    OUTSIDE the yellow boxes, they are different object classes.

Reads only the external Mendeley PNGs. Touches none of our cohort.
"""
import json
import os
import sys
import collections

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = ("data/The number of nail fold capillaries/The number of nail fold "
        "capillaries and nail fold bleedings reflects the clinical "
        "manifestations of systemic sclerosis")
OUT = "artifacts/evidence/mendeley_semantics_20260920"

RED = lambda R, G, B: (R >= 190) & (G <= 90) & (B <= 90)
YELLOW = lambda R, G, B: (R >= 190) & (G >= 190) & (B <= 110)
MIN_PX = 8


def load(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    return a[:, :, 0], a[:, :, 1], a[:, :, 2]


def components(mask, min_px=MIN_PX):
    lb, n = ndimage.label(mask)
    out = []
    for idx, sl in enumerate(ndimage.find_objects(lb)):
        if sl is None:
            continue
        lab = idx + 1
        sub = lb[sl] == lab
        if sub.sum() < min_px:
            continue
        out.append((sl, sub))
    return out


def rectangle_score(sub):
    """Does this component look like a hollow axis-aligned rectangle?

    For a 1-px hollow rectangle of height H and width W:
      - exactly 2 rows have sum == W (top and bottom edges)
      - the other H-2 rows have sum == 2 (the two side edges)
    For a diagonal stroke, no row reaches W and most rows sum to 1-2.
    Returns (frac_rows_full, frac_rows_two, frac_cols_full, frac_cols_two).
    """
    H, W = sub.shape
    if H < 5 or W < 5:
        return None
    rs = sub.sum(axis=1)
    cs = sub.sum(axis=0)
    # allow antialiasing slack: "full" = covers >=80% of the span
    rows_full = int((rs >= 0.8 * W).sum())
    cols_full = int((cs >= 0.8 * H).sum())
    rows_thin = int(((rs >= 1) & (rs <= 4)).sum())
    cols_thin = int(((cs >= 1) & (cs <= 4)).sum())
    return dict(H=H, W=W, rows_full=rows_full, cols_full=cols_full,
                frac_rows_thin=rows_thin / H, frac_cols_thin=cols_thin / W,
                max_row=int(rs.max()), max_col=int(cs.max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(".png"))
    files = [f for f in files if os.path.getsize(os.path.join(ROOT, f)) > 0]
    assert files, "no readable png found"

    # --- Q1 on a sample big enough to be decisive, all yellow components ---
    rect_votes = collections.Counter()
    rect_rows = []
    # --- Q3 containment ---
    inside = 0
    outside = 0
    red_counts = []
    yellow_counts = []

    sample = files[::4]  # every 4th image: ~144 images, all their components
    for k, fn in enumerate(sample):
        R, G, B = load(os.path.join(ROOT, fn))
        ym = YELLOW(R, G, B)
        rm = RED(R, G, B)
        ycomp = components(ym)
        rcomp = components(rm)
        yellow_counts.append(len(ycomp))
        red_counts.append(len(rcomp))

        boxes = []
        for sl, sub in ycomp:
            sc = rectangle_score(sub)
            if sc is None:
                continue
            # a hollow rectangle must have BOTH a full row and a full column
            is_rect = (sc["rows_full"] >= 2 and sc["cols_full"] >= 2
                       and sc["frac_rows_thin"] > 0.5)
            rect_votes["rect" if is_rect else "stroke"] += 1
            sc["file"] = fn
            sc["is_rect"] = bool(is_rect)
            rect_rows.append(sc)
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))

        # Q3: are red centroids inside any yellow box?
        for sl, sub in rcomp:
            cy = (sl[0].start + sl[0].stop) / 2.0
            cx = (sl[1].start + sl[1].stop) / 2.0
            hit = any(y0 <= cy <= y1 and x0 <= cx <= x1
                      for y0, y1, x0, x1 in boxes)
            if hit:
                inside += 1
            else:
                outside += 1

    red_counts = np.array(red_counts)
    yellow_counts = np.array(yellow_counts)

    res = {
        "images_sampled": len(sample),
        "Q1_yellow_shape": {
            "verdict_counts": dict(rect_votes),
            "frac_rectangles": (rect_votes["rect"] /
                                max(1, sum(rect_votes.values()))),
            "reading": ("hollow LabelImg bounding boxes"
                        if rect_votes["rect"] > rect_votes["stroke"]
                        else "open strokes / polylines, not boxes"),
            "examples": rect_rows[:12],
        },
        "Q2_red_is_nfc_or_nfb": {
            "per_image_red": {
                "median": float(np.median(red_counts)),
                "mean": float(red_counts.mean()),
                "p05": float(np.percentile(red_counts, 5)),
                "p95": float(np.percentile(red_counts, 95)),
                "max": int(red_counts.max()),
                "frac_images_zero": float((red_counts == 0).mean()),
                "frac_images_le5": float((red_counts <= 5).mean()),
            },
            "per_image_yellow": {
                "median": float(np.median(yellow_counts)),
                "max": int(yellow_counts.max()),
                "frac_images_zero": float((yellow_counts == 0).mean()),
                "frac_images_le5": float((yellow_counts <= 5).mean()),
            },
            "argument": (
                "NFB in SSc are counted in single digits per nail and are "
                "absent in many nails, so an NFB channel must be strongly "
                "zero-inflated. A channel with median 28 and almost no zeros "
                "is a dense structure, i.e. capillaries."),
        },
        "Q3_red_inside_yellow": {
            "red_centroids_inside_a_yellow_box": inside,
            "red_centroids_outside": outside,
            "frac_inside": inside / max(1, inside + outside),
            "reading": ("same objects annotated twice (box + centre)"
                        if inside / max(1, inside + outside) > 0.5
                        else "different object classes"),
        },
        "images_opened_from_our_cohort": 0,
        "locked_cases_seen": 0,
        "models_run": 0,
    }
    with open(os.path.join(OUT, "semantics.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "Q1_yellow_shape"},
                     indent=2, ensure_ascii=False))
    q = res["Q1_yellow_shape"]
    print("Q1 verdict:", q["reading"], q["verdict_counts"],
          "frac_rect=%.3f" % q["frac_rectangles"])


if __name__ == "__main__":
    sys.exit(main())
