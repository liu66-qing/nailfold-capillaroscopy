"""Decode what the Mendeley overlay actually is, and recover it as coordinates.

Earlier I scanned with EXACT colour equality (255,0,0) and reported 13834 red
components. That undercounted: a hue histogram over saturated pixels shows two
peaks only -- red (0-20 and 350-360 deg) and yellow (50-70 deg) -- and nothing
else, but the markers are antialiased, so exact equality clips their edges and
splits some of them. With tolerant thresholds the counts are 18289 red and 2894
yellow. The white pixels I earlier listed as a third marker colour are NOT a
marker colour: their components render as solid filled blobs with no hue, i.e.
saturated highlights in the photograph itself. They are dropped here.

What the red markers are, established by measurement rather than by eye:
  - 576 images carry them; only 4 do not. Median 28 per image, max 115.
  - Mean nearest-neighbour spacing ~15 px and median x-gap ~15 px, with
    y_std/H ~0.14 -- they lie in a band, spread over 50-80% of the image width.
    That is the geometry of one row of capillary loops along a nailfold.
  - Luminance 14 px BELOW a marker is 15.2 lower than 14 px ABOVE it, so the
    dark structure hangs below the marked point: the marker sits on the loop
    TIP, which is what an NFC count marks.
  - Red count does NOT correlate with yellow count (r = -0.011), so they are
    not two encodings of the same thing.

What the yellow strokes are: 1-11 per image (median 5), fill ratio 0.04,
aspect 2.4, and ASCII rendering shows single-pixel-wide diagonal polylines.
They are boundary/measurement lines, not counts.

IMPORTANT unresolved item, stated rather than guessed: the paper describes BOTH
NFC (capillary) and NFB (bleeding) annotation, but only ONE dot colour exists.
So a single red channel cannot carry both. Either the NFB annotation is not in
this release, or red means one of the two and the other is absent. This script
does NOT assign red to capillaries in its output field names; it emits
`n_red_dots` and records the evidence above. Calling it capillary_count
supervision requires resolving this, and the resolution is not in the pixels.

Also measured, and a hard constraint on any appearance-field use: two images are
0 bytes AT SOURCE, two files are named `control <name>.png` (no date, no index)
and one of those has zero red dots -- they are the healthy controls, not
patients, and must not join a patient cohort. The overlay covers pixels in
576/576 images, so this dataset contains NO clean frame.

Usage:
  python scripts/decode_mendeley_overlay.py \
      --out-dir artifacts/evidence/mendeley_overlay_20260919
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

DEFAULT_ROOT = ("data/The number of nail fold capillaries/The number of nail "
                "fold capillaries and nail fold bleedings reflects the "
                "clinical manifestations of systemic sclerosis")
# tolerant thresholds; exact equality clips antialiased marker edges
RED = lambda R, G, B: (R >= 190) & (G <= 90) & (B <= 90)          # noqa: E731
YELLOW = lambda R, G, B: (R >= 190) & (G >= 190) & (B <= 110)     # noqa: E731
MIN_PX = 8


def components(mask, min_px=MIN_PX):
    """Centroids and areas of connected components at or above min_px."""
    lb, n = ndimage.label(mask)
    if n == 0:
        return [], []
    sz = ndimage.sum(mask, lb, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sz) if s >= min_px]
    if not keep:
        return [], []
    cen = ndimage.center_of_mass(mask, lb, keep)
    return cen, [float(sz[k - 1]) for k in keep]


def panels(lum, frac=0.02):
    """Bounding boxes of content regions. Many images are 2-panel composites:
    a large capillaroscopy field plus a small 340x240 inset, and the inset
    carries its own markers, so a per-image count mixes two fields of view."""
    content = ndimage.binary_closing(lum > 28, np.ones((9, 9)))
    lb, n = ndimage.label(content)
    if n == 0:
        return []
    sz = ndimage.sum(content, lb, range(1, n + 1))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lb)):
        if sz[i] <= frac * lum.size:
            continue
        out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop,
                    float(sz[i])))
    out.sort(key=lambda t: -(t[2] - t[0]) * (t[3] - t[1]))
    return out


def tip_evidence(lum, mask_all, centroids, offset=14, half=5):
    """Is the dark structure BELOW each marker (loop body under a marked tip)?

    Returns mean luminance in clean patches above and below the centroids,
    skipping any patch that overlaps another marker.
    """
    md = ndimage.binary_dilation(mask_all, iterations=3)
    up, dn = [], []
    for cy, cx in centroids:
        cy, cx = int(cy), int(cx)
        for dy, acc in ((-offset, up), (offset, dn)):
            y0, y1 = cy + dy - half, cy + dy + half + 1
            x0, x1 = cx - half, cx + half + 1
            if y0 < 0 or x0 < 0 or y1 > lum.shape[0] or x1 > lum.shape[1]:
                continue
            w, mm = lum[y0:y1, x0:x1], md[y0:y1, x0:x1]
            if mm.mean() > 0.3 or not (~mm).any():
                continue
            acc.append(float(w[~mm].mean()))
    return up, dn


def parse_name(fn):
    """`160422akiyama koichi1.png` -> exam `160422akiyama koichi`, person
    `akiyama koichi`, index 1. `control <name>.png` has no date and no index;
    it is a healthy control, flagged and kept out of the patient cohort."""
    stem = re.sub(r"\.png$", "", fn, flags=re.I)
    if stem.lower().startswith("control"):
        return {"exam": stem, "person": stem, "image_index": None,
                "exam_date": None, "is_control": True}
    idx = re.search(r"(\d+)$", stem)
    body = re.sub(r"\d+$", "", stem)
    date = re.match(r"^(\d{6})", body)
    return {"exam": body, "person": re.sub(r"^\d{6}", "", body),
            "image_index": int(idx.group(1)) if idx else None,
            "exam_date": date.group(1) if date else None,
            "is_control": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(a.root) if f.lower().endswith(".png"))
    zero = [f for f in files if os.path.getsize(os.path.join(a.root, f)) == 0]

    pts, per_img = [], []
    up_all, dn_all = [], []
    nn_all, ystd_all = [], []
    for fn in files:
        if fn in set(zero):
            continue
        a_img = np.array(Image.open(os.path.join(a.root, fn)).convert("RGB"))
        A = a_img.astype(np.int16)
        R, G, B = A[..., 0], A[..., 1], A[..., 2]
        red, yel = RED(R, G, B), YELLOW(R, G, B)
        lum = A.mean(2)
        H, W = lum.shape
        rc, ra = components(red)
        yc, ya = components(yel)
        meta = parse_name(fn)

        pan = panels(lum)
        for (cy, cx), area in zip(rc, ra):
            which = 0
            for pi, (x0, y0, x1, y1, _s) in enumerate(pan):
                if x0 <= cx < x1 and y0 <= cy < y1:
                    which = pi
                    break
            pts.append({"file": fn, "person": meta["person"],
                        "exam": meta["exam"], "is_control": meta["is_control"],
                        "color": "red", "x": round(float(cx), 2),
                        "y": round(float(cy), 2), "area_px": area,
                        "panel_rank": which, "img_w": W, "img_h": H})

        if len(rc) >= 5:
            u, dn = tip_evidence(lum, red | yel, rc)
            up_all += u
            dn_all += dn
        if len(rc) >= 8:
            arr = np.array(rc)
            ys, xs = arr[:, 0], arr[:, 1]
            d = []
            for i in range(len(arr)):
                dist = np.sqrt((ys - ys[i]) ** 2 + (xs - xs[i]) ** 2)
                dist[i] = np.inf
                d.append(dist.min())
            nn_all.append(float(np.median(d)))
            ystd_all.append(float(ys.std() / H))

        per_img.append({"file": fn, **meta, "w": W, "h": H,
                        "n_red_dots": len(rc), "n_yellow_strokes": len(yc),
                        "red_px": int(red.sum()), "yellow_px": int(yel.sum()),
                        "n_panels": len(pan),
                        "marker_px_fraction": round(
                            float((red | yel).sum()) / (H * W), 5)})
    write_outputs(a, files, zero, pts, per_img, up_all, dn_all, nn_all,
                  ystd_all)


def write_outputs(a, files, zero, pts, per_img, up_all, dn_all, nn_all,
                  ystd_all):
    dp = pd.DataFrame(pts)
    di = pd.DataFrame(per_img)
    dp.to_csv(os.path.join(a.out_dir, "red_dots.csv"), index=False)
    di.to_csv(os.path.join(a.out_dir, "per_image.csv"), index=False)

    pat = di[~di.is_control]
    by_exam = pat.groupby("exam").agg(
        images=("file", "size"), red=("n_red_dots", "sum"),
        yellow=("n_yellow_strokes", "sum")).reset_index()
    by_exam["person"] = by_exam.exam.str.replace(r"^\d{6}", "", regex=True)
    by_exam["red_per_image"] = (by_exam.red / by_exam.images).round(2)
    by_exam.to_csv(os.path.join(a.out_dir, "per_exam.csv"), index=False)

    out = {
        "source": {"doi": "10.17632/8wrjdknb5k.1", "licence": "CC BY 4.0",
                   "root": a.root},
        "inventory": {
            "png_files": len(files),
            "zero_byte_at_source": zero,
            "images_decoded": int(len(di)),
            "control_images_excluded_from_cohort": sorted(
                di[di.is_control].file.tolist()),
            "patient_images": int(len(pat)),
            "patient_exams": int(pat.exam.nunique()),
            "patient_persons": int(pat.person.nunique()),
            "persons_with_multiple_exams": int(
                (pat.groupby("person").exam.nunique() > 1).sum()),
            "images_with_two_panels": int((di.n_panels >= 2).sum()),
        },
        "correction_to_my_earlier_exact_colour_scan": {
            "earlier_red_components_exact_255_0_0": 13834,
            "now_red_components_tolerant": int(di.n_red_dots.sum()),
            "earlier_yellow": 2979,
            "now_yellow": int(di.n_yellow_strokes.sum()),
            "white_channel_dropped": "the 6499 'white' components I listed are "
                                     "solid unsaturated blobs -- specular "
                                     "highlights in the photograph, not "
                                     "markers. A hue histogram over saturated "
                                     "pixels has exactly two peaks, red and "
                                     "yellow, and no third.",
        },
        "red_dots_per_image": {
            "median": float(pat.n_red_dots.median()),
            "mean": round(float(pat.n_red_dots.mean()), 2),
            "min": int(pat.n_red_dots.min()), "max": int(pat.n_red_dots.max()),
            "images_with_zero": int((pat.n_red_dots == 0).sum()),
            "total": int(pat.n_red_dots.sum()),
        },
        "yellow_strokes_per_image": {
            "median": float(pat.n_yellow_strokes.median()),
            "min": int(pat.n_yellow_strokes.min()),
            "max": int(pat.n_yellow_strokes.max()),
            "total": int(pat.n_yellow_strokes.sum()),
            "correlation_with_red_count": round(
                float(pat.n_red_dots.corr(pat.n_yellow_strokes)), 3),
        },
        "what_the_red_dots_mark": {
            "median_nearest_neighbour_px": round(float(np.median(nn_all)), 1),
            "median_y_std_over_height": round(float(np.median(ystd_all)), 3),
            "mean_luminance_above_marker": round(float(np.mean(up_all)), 1),
            "mean_luminance_below_marker": round(float(np.mean(dn_all)), 1),
            "below_minus_above": round(
                float(np.mean(dn_all) - np.mean(up_all)), 1),
            "reading": "dots lie in a band (y_std/H ~0.14) at ~15 px spacing "
                       "across most of the width, and the dark structure is "
                       "BELOW each dot. That is one row of capillary loops "
                       "with the dot on each tip.",
        },
        "UNRESOLVED_nfc_vs_nfb": {
            "problem": "the paper annotates BOTH capillaries (NFC) and "
                       "bleedings (NFB), but only ONE dot colour exists.",
            "consequence": "red cannot be assigned to capillary_count without "
                           "external confirmation. Output is named "
                           "n_red_dots, not capillary_count.",
            "how_to_resolve": "read the source publication's figure legend, or "
                              "ask the depositors. Not answerable from pixels.",
        },
        "hard_constraints_for_any_use": [
            "no clean frame exists: markers cover pixels in every decoded "
            "image (median marker fraction %.4f), so appearance fields "
            "(clarity, blood_color, exudation) cannot be read off these "
            "images without inpainting, and inpainting itself changes "
            "clarity statistics."
            % float(di.marker_px_fraction.median()),
            "%d images are 2-panel composites (a large field plus a ~340x240 "
            "inset). A per-image dot count mixes two fields of view; "
            "panel_rank is emitted per dot so counts can be restricted to "
            "the largest panel." % int((di.n_panels >= 2).sum()),
            "cohort is %d persons over %d exams -- repeat visits, not new "
            "people. Any split must be BY PERSON."
            % (int(pat.person.nunique()), int(pat.exam.nunique())),
            "two files are 0 bytes at source, not a broken download.",
            "the two `control <name>.png` files are healthy controls with no "
            "date and no image index; excluded from the patient cohort here.",
        ],
        "images_opened_from_our_cohort": 0,
        "locked_cases_seen": 0,
        "models_run": 0,
    }
    with open(os.path.join(a.out_dir, "overlay.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print_summary(out, by_exam)


def print_summary(out, by_exam):
    inv = out["inventory"]
    print("=== Mendeley overlay decoded ===")
    print("patient images %d  exams %d  PERSONS %d (%d repeat)"
          % (inv["patient_images"], inv["patient_exams"],
             inv["patient_persons"], inv["persons_with_multiple_exams"]))
    r = out["red_dots_per_image"]
    print("red dots: total %d  median/img %.0f  max %d  zero in %d imgs"
          % (r["total"], r["median"], r["max"], r["images_with_zero"]))
    y = out["yellow_strokes_per_image"]
    print("yellow strokes: total %d  median/img %.0f  corr with red %+.3f"
          % (y["total"], y["median"], y["correlation_with_red_count"]))
    w = out["what_the_red_dots_mark"]
    print("tip test: L above %.1f, below %.1f (%+.1f)"
          % (w["mean_luminance_above_marker"],
             w["mean_luminance_below_marker"], w["below_minus_above"]))
    print("2-panel composites: %d images" % inv["images_with_two_panels"])
    print("\ntop exams by dot count:")
    print(by_exam.sort_values("red", ascending=False)
          .head(8).to_string(index=False))


if __name__ == "__main__":
    main()
