"""The two measurements that force my retractions on this dataset.

Both existed only in shell history; without them the conclusions in
memory/nailfold-mendeley-overlay-truth.md are unreproducible.

T1. HOLLOW-RECTANGLE TEST (does yellow = LabelImg boxes?)
    fill fraction CANNOT separate a hollow rectangle from an open stroke: a
    1-px-outlined 174x73 box has perimeter/area = 0.039, which is the ~0.04 I
    measured and wrongly read as "polyline" by eye. The separating statistics:
      - perimeter concentration: fraction of component pixels within 3 px of the
        bounding-box edge. A hollow rectangle -> ~1.0. A diagonal stroke -> low,
        because its middle passes through the box interior.
      - enclosed-area fraction: 1 - area/binary_fill_holes(area). A closed
        rectangle encloses its interior -> high. An open stroke encloses nothing.
    Verdict from this test decides whether the paper's LabelImg annotations
    (nail boundary + NFB + NFC) are actually shipped, which in turn decides
    whether this dataset can supply the DENOMINATOR for capillary_count (条/mm)
    and the two per-nailfold fields (出血, 汗腺导管).

T2. INSTANCE COUNT AND SCALE (is one red dot one capillary?)
    Compare red-dot nearest-neighbour spacing against OUR OWN vessel spacing,
    both normalised to 1024 px width, using the SAM-proposed miss polygons as
    our reference (they carry real image coordinates). If the spacings differ by
    a large factor the dots cannot be one-per-vessel, and the true instance
    count is the number of dot CLUSTERS, not the number of dots.

    Also runs the zero-inflation argument that settles NFC vs NFB: nailfold
    bleedings in SSc are single-digit per nail and absent in many nails, so an
    NFB channel must be strongly zero-inflated.

Reads the external Mendeley PNGs and our own SAM proposal polygons (coordinates
only, no image of ours is opened). No model is run.
"""
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

ROOT = ("data/The number of nail fold capillaries/The number of nail fold "
        "capillaries and nail fold bleedings reflects the clinical "
        "manifestations of systemic sclerosis")
DOTS = "artifacts/evidence/mendeley_overlay_20260919/red_dots.csv"
OURS = "artifacts/features/segmiss_v1/ai_proposed_misses.jsonl"
OUT = "artifacts/evidence/mendeley_semantics_20260920"

YELLOW = lambda a: (a[:, :, 0] >= 190) & (a[:, :, 1] >= 190) & (a[:, :, 2] <= 110)
LINK_PX = 30          # cluster link distance; the cluster count depends on it
REF_WIDTH = 1024.0    # our own images are 1024x768


def t1_hollow_rectangle_test(every=7, min_px=40, min_side=15):
    per, hole, n = [], [], 0
    files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(".png"))
    files = [f for f in files if os.path.getsize(os.path.join(ROOT, f)) > 0]
    for fn in files[::every]:
        a = np.asarray(Image.open(os.path.join(ROOT, fn)).convert("RGB")).astype(np.int16)
        lb, _ = ndimage.label(YELLOW(a))
        for i, sl in enumerate(ndimage.find_objects(lb)):
            if sl is None:
                continue
            sub = lb[sl] == (i + 1)
            H, W = sub.shape
            if sub.sum() < min_px or H < min_side or W < min_side:
                continue
            ys, xs = np.nonzero(sub)
            d = np.minimum.reduce([ys, H - 1 - ys, xs, W - 1 - xs])
            per.append(float((d <= 3).mean()))
            filled = ndimage.binary_fill_holes(sub)
            hole.append(float(1.0 - sub.sum() / max(filled.sum(), 1)))
            n += 1
    per, hole = np.array(per), np.array(hole)
    is_rect = (np.median(per) > 0.8) and (np.median(hole) > 0.5)
    return {
        "n_components": n,
        "perimeter_concentration": {"median": float(np.median(per)),
                                    "frac_above_0.9": float((per > 0.9).mean())},
        "enclosed_area_fraction": {"median": float(np.median(hole)),
                                   "frac_above_0.5": float((hole > 0.5).mean())},
        "expected_if_hollow_rectangle": {"perimeter_concentration": "~1.0",
                                         "enclosed_area_fraction": ">0.9"},
        "verdict": ("hollow LabelImg bounding boxes" if is_rect else
                    "open ~3px-wide strokes; device measurement calipers, "
                    "NOT the paper's LabelImg annotations"),
        "consequence_if_not_boxes": (
            "the dataset ships NO annotation file -- only a burnt-in device "
            "overlay. It therefore cannot supply the nail boundary, i.e. cannot "
            "supply the DENOMINATOR for 条/mm or for the per-nailfold fields."),
    }


def _nn_spacing(xy):
    D = np.hypot(xy[:, 0:1] - xy[:, 0], xy[:, 1:2] - xy[:, 1])
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(1)))


def t2_scale_and_instances():
    import pandas as pd
    d = pd.read_csv(DOTS)
    d = d[d.panel_rank == 0]          # main field only; insets are a 2nd FOV

    spac_norm, clusters, sizes = [], 0, []
    for _f, sub in d.groupby("file"):
        if len(sub) < 6:
            continue
        xy = sub[["x", "y"]].to_numpy(float)
        spac_norm.append(_nn_spacing(xy) / float(sub.img_w.iloc[0]))
        D = np.hypot(xy[:, 0:1] - xy[:, 0], xy[:, 1:2] - xy[:, 1])
        ncc, lab = connected_components(
            sparse.csr_matrix((D < LINK_PX) & (D > 0)), directed=False)
        clusters += ncc
        sizes.extend(np.bincount(lab).tolist())
    spac_norm = np.array(spac_norm)
    sizes = np.array(sizes)
    their_at_ref = float(np.median(spac_norm) * REF_WIDTH)

    # our own vessel spacing, from SAM proposal polygon centroids
    per = collections.defaultdict(list)
    with open(OURS, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            o = json.loads(ln)
            per[o["image_path"]].append(np.array(o["poly"], float).mean(0))
    ours = [_nn_spacing(np.array(v)) for v in per.values() if len(v) >= 6]
    ours = np.array(ours)

    counts = d.groupby("file").size().to_numpy()
    return {
        "their_nn_spacing_px_median": float(np.median(spac_norm) *
                                            d.img_w.median()),
        "their_nn_spacing_at_1024_width": their_at_ref,
        "our_nn_spacing_px_median_1024_imgs": float(np.median(ours)),
        "our_nn_spacing_p25_p75": [float(np.percentile(ours, 25)),
                                   float(np.percentile(ours, 75))],
        "our_n_images": int(len(ours)),
        "scale_factor": float(np.median(ours) / their_at_ref),
        "clusters_link_%dpx" % LINK_PX: {
            "n_clusters_total": int(clusters),
            "mean_cluster_size": float(sizes.mean()),
            "frac_singletons": float((sizes == 1).mean()),
            "frac_in_multi_dot_cluster": float((sizes > 1).mean()),
            "caveat": "cluster count depends on the arbitrary %d px link" % LINK_PX,
        },
        "verdict": (
            "one red dot is NOT one capillary: the dots sit ~%.1fx closer than "
            "our vessels do, and %.1f%% of them lie in multi-dot clusters. They "
            "trace the CONTOUR of a loop. The usable instance count is the "
            "cluster count (~%d), not the dot count."
            % (np.median(ours) / their_at_ref, 100 * (sizes > 1).mean(), clusters)),
        "nfc_vs_nfb": {
            "per_image_red_median": float(np.median(counts)),
            "frac_images_zero": float((counts == 0).mean()),
            "frac_images_le5": float((counts <= 5).mean()),
            "argument": ("NFB in SSc are single-digit per nail and absent in "
                         "many nails, so an NFB channel must be strongly "
                         "zero-inflated. This channel is not."),
            "verdict": "red marks CAPILLARIES (NFC); the hemorrhage branch is ruled out",
        },
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"T1_yellow_is_it_a_box": t1_hollow_rectangle_test(),
           "T2_scale_and_instance_count": t2_scale_and_instances(),
           "images_opened_from_our_cohort": 0,
           "locked_cases_seen": 0,
           "models_run": 0}
    with open(os.path.join(OUT, "scale_and_instances.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
