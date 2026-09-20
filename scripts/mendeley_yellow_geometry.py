"""What are the yellow strokes, geometrically? And is the 41%% red-inside-yellow
figure real or chance?

Q1 already ruled out hollow rectangles (0 of 712 components had both a full row
and a full column). So the yellow marks are open curves. Three candidates remain:
  (a) nail-fold boundary lines  -> long, spanning much of the image width, few
      per image, roughly horizontal
  (b) per-capillary measurement segments (diameter/length calipers) -> short,
      many per image, oriented along the loop axis (vertical-ish)
  (c) freehand region outlines -> long, closed-ish, high curvature

Discriminators computed here per component:
  span_x/W, span_y/H, arc length (pixel count), end-to-end distance,
  straightness = end_to_end / arc_len, dominant orientation via PCA,
  and whether the two endpoints are far apart (open) or adjacent (closed loop).

For Q3 I add the null model I owe: shuffle the red centroids uniformly inside
the image content panel and recompute the containment fraction. If the observed
41%% matches the null, red and yellow are independent; if it is far above, they
co-locate.
"""
import json
import os
import sys

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
RNG = np.random.default_rng(20260920)


def load(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    return a[:, :, 0], a[:, :, 1], a[:, :, 2]


def comps(mask, min_px=MIN_PX):
    lb, n = ndimage.label(mask)
    out = []
    for idx, sl in enumerate(ndimage.find_objects(lb)):
        if sl is None:
            continue
        sub = lb[sl] == (idx + 1)
        if sub.sum() < min_px:
            continue
        out.append((sl, sub))
    return out


def stroke_geom(sl, sub, H, W):
    ys, xs = np.nonzero(sub)
    ys = ys + sl[0].start
    xs = xs + sl[1].start
    arc = len(ys)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    # principal axis angle, 0 = horizontal, 90 = vertical
    ang = float(np.degrees(np.arctan2(abs(vt[0][1]), abs(vt[0][0]))))
    proj = (pts - c) @ vt[0]
    end_to_end = float(proj.max() - proj.min())
    elong = float(s[0] / max(s[1], 1e-6))
    return dict(arc=arc, span_x=int(np.ptp(xs) + 1), span_y=int(np.ptp(ys) + 1),
                span_x_frac=float((np.ptp(xs) + 1) / W),
                span_y_frac=float((np.ptp(ys) + 1) / H),
                end_to_end=end_to_end,
                straightness=float(end_to_end / max(arc, 1)),
                angle_deg=ang, elongation=elong)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(".png"))
    files = [f for f in files if os.path.getsize(os.path.join(ROOT, f)) > 0]
    sample = files[::4]

    rows = []
    obs_in, obs_tot = 0, 0
    null_in, null_tot = 0, 0
    for fn in sample:
        R, G, B = load(os.path.join(ROOT, fn))
        H, W = R.shape
        ym, rm = YELLOW(R, G, B), RED(R, G, B)
        yc, rc = comps(ym), comps(rm)
        boxes = []
        for sl, sub in yc:
            g = stroke_geom(sl, sub, H, W)
            g["file"] = fn
            rows.append(g)
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
        if not boxes or not rc:
            continue
        cys = np.array([(s[0].start + s[0].stop) / 2 for s, _ in rc])
        cxs = np.array([(s[1].start + s[1].stop) / 2 for s, _ in rc])

        def frac_in(cy, cx):
            return any(y0 <= cy <= y1 and x0 <= cx <= x1
                       for y0, y1, x0, x1 in boxes)

        for cy, cx in zip(cys, cxs):
            obs_tot += 1
            obs_in += frac_in(cy, cx)
        # null: same number of points, uniform over the red points' own
        # bounding region (not the whole letterboxed image)
        y0, y1 = cys.min(), cys.max()
        x0, x1 = cxs.min(), cxs.max()
        for _ in range(len(cys)):
            null_tot += 1
            null_in += frac_in(RNG.uniform(y0, y1), RNG.uniform(x0, x1))

    a = {k: np.array([r[k] for r in rows], float)
         for k in ("arc", "span_x_frac", "span_y_frac", "straightness",
                   "angle_deg", "elongation", "span_x", "span_y")}
    q = lambda v: dict(p25=float(np.percentile(v, 25)),
                       median=float(np.median(v)),
                       p75=float(np.percentile(v, 75)),
                       max=float(v.max()))
    res = {
        "n_components": len(rows),
        "images_sampled": len(sample),
        "geometry": {k: q(v) for k, v in a.items()},
        "orientation": {
            "frac_within_20deg_of_horizontal": float((a["angle_deg"] <= 20).mean()),
            "frac_within_20deg_of_vertical": float((a["angle_deg"] >= 70).mean()),
        },
        "straightness_reading": (
            "near 1.0 = a straight open segment traced once; near 0.5 = the "
            "trace doubles back (drawn 2 px wide or closed); much below 0.4 = "
            "closed or strongly curved outline"),
        "Q3_containment_vs_null": {
            "observed_frac_inside": obs_in / max(1, obs_tot),
            "null_frac_inside": null_in / max(1, null_tot),
            "n_red": obs_tot,
            "reading": None,
        },
        "images_opened_from_our_cohort": 0,
        "locked_cases_seen": 0,
        "models_run": 0,
    }
    o = res["Q3_containment_vs_null"]["observed_frac_inside"]
    nl = res["Q3_containment_vs_null"]["null_frac_inside"]
    res["Q3_containment_vs_null"]["reading"] = (
        "co-located with the yellow marks (observed >> chance)"
        if o > nl * 1.5 else
        "independent of the yellow marks (observed ~ chance)")
    with open(os.path.join(OUT, "yellow_geometry.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:2600])


if __name__ == "__main__":
    sys.exit(main())
