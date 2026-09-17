"""B1: native-resolution (1024x768) instance geometry feature extraction.

Masks in the npz are BIT-PACKED along the last axis: (N, 768, 128) uint8
-> np.unpackbits(axis=-1) -> (N, 768, 1024), matching `boxes` (1024 coords).
No resize anywhere. This is the fix for the 512x384 resize in
scripts/extract_geometry_features.py line 27.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

SEED = 20260913
NPZ_DIR = Path("/root/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev2")
OUT = Path("/root/nailfold/artifacts/experiments/v10_B_measurement")


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def skeleton_path_length(skel: np.ndarray) -> float:
    """Longest SHORTEST-path (geodesic diameter) on the skeleton, diagonal-aware.

    Double-sweep Dijkstra on the sparse 8-neighbour graph. Using shortest paths
    keeps this polynomial even when the skeleton contains cycles (the longest
    simple path is NP-hard and blew up on real masks).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    pts = np.transpose(np.nonzero(skel))
    n = len(pts)
    if n < 2:
        return float(n)
    lut = -np.ones(skel.shape, dtype=np.int32)
    lut[pts[:, 0], pts[:, 1]] = np.arange(n)
    r, c, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys, xs = pts[:, 0] + dy, pts[:, 1] + dx
        ok = (ys >= 0) & (ys < skel.shape[0]) & (xs >= 0) & (xs < skel.shape[1])
        j = np.where(ok, lut[np.clip(ys, 0, skel.shape[0] - 1), np.clip(xs, 0, skel.shape[1] - 1)], -1)
        m = j >= 0
        r.append(np.arange(n)[m]); c.append(j[m])
        w.append(np.full(m.sum(), 1.41421356 if (dy and dx) else 1.0))
    if not len(np.concatenate(r)):
        return float(n)
    g = coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n))
    g = g + g.T
    d0 = dijkstra(g, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    d1 = dijkstra(g, indices=a)
    d1[~np.isfinite(d1)] = -1
    return float(d1.max())


def qstats(v: np.ndarray, prefix: str) -> dict:
    v = v[np.isfinite(v)]
    if not len(v):
        return {f"{prefix}_{s}": np.nan for s in ("p10", "p50", "p90", "max", "mean")}
    return {
        f"{prefix}_p10": float(np.percentile(v, 10)),
        f"{prefix}_p50": float(np.percentile(v, 50)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_max": float(v.max()),
        f"{prefix}_mean": float(v.mean()),
    }


def instance_features(mask: np.ndarray) -> dict | None:
    """mask: bool (768,1024) native resolution."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = mask[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    area = float(sub.sum())

    # distance transform -> local width = 2 * DT
    dt = ndi.distance_transform_edt(np.pad(sub, 1))[1:-1, 1:-1]
    skel = skeletonize(sub)
    if skel.sum() < 3:
        return None

    sk_len = skeleton_path_length(skel)
    sy, sx = np.nonzero(skel)
    widths = 2.0 * dt[sy, sx]

    f: dict = {
        "sk_len": sk_len,
        "area": area,
        "bbox_h": float(h),
        "bbox_w": float(w),
        "aspect": float(h) / max(w, 1.0),
        "extent": area / max(h * w, 1.0),
        "sk_tortuosity": sk_len / max(np.hypot(h, w), 1.0),
    }
    try:
        from skimage.measure import label as sklabel, regionprops
        rp = regionprops(sklabel(sub.astype(np.uint8)))
        rp = max(rp, key=lambda r: r.area)
        f["solidity"] = float(rp.solidity)
        f["eccentricity"] = float(rp.eccentricity)
        f["perimeter"] = float(rp.perimeter)
        f["circularity"] = 4 * np.pi * area / max(rp.perimeter ** 2, 1e-6)
    except Exception:
        f.update(solidity=np.nan, eccentricity=np.nan, perimeter=np.nan, circularity=np.nan)

    # ANATOMICAL SEGMENTATION: capillary loop is a hairpin, apex at top.
    # segment along the skeleton's vertical extent within the instance.
    rel = (sy - sy.min()) / max(sy.max() - sy.min(), 1)
    apex_m = rel <= 0.20              # apex region: top 20%
    mid_x = (sx.min() + sx.max()) / 2.0
    limb_m = rel > 0.20
    # ascending (afferent) vs descending (efferent) limb by side of centreline
    asc_m = limb_m & (sx <= mid_x)
    des_m = limb_m & (sx > mid_x)

    f.update(qstats(widths, "w_all"))
    f.update(qstats(widths[apex_m], "w_apex"))
    f.update(qstats(widths[asc_m], "w_asc"))
    f.update(qstats(widths[des_m], "w_des"))
    # limb asymmetry: proxy for output/input ratio structure
    a50 = f.get("w_asc_p50", np.nan)
    d50 = f.get("w_des_p50", np.nan)
    f["w_limb_ratio"] = (d50 / a50) if (a50 and np.isfinite(a50) and np.isfinite(d50) and a50 > 0) else np.nan
    f["w_apex_over_limb"] = (f.get("w_apex_p50", np.nan) / a50) if (a50 and a50 > 0) else np.nan
    return f


_S = None
_N = None


def _worker(f: Path):
    d = np.load(f)
    sc, bx, mk = d["score"], d["boxes"], d["masks"]
    keep = np.nonzero(sc >= _S)[0]
    if len(keep):
        keep = keep[nms(bx[keep], sc[keep], _N)]
    stem = f.stem.split("__")
    case = f"{stem[0]}/{stem[1]}"
    out = []
    for i in keep:
        m = np.unpackbits(mk[i], axis=-1).astype(bool)  # (768,1024) NATIVE, no resize
        assert m.shape == (768, 1024), m.shape
        g = instance_features(m)
        if g is None:
            continue
        g.update(exam_case_id=case, frame=f.stem, score=float(sc[i]))
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=str(OUT / "instance_features_native.parquet"))
    args = ap.parse_args()

    np.random.seed(SEED)
    files = sorted(NPZ_DIR.glob("*.npz"))
    assert len(files) == 1687, len(files)

    if args.calibrate:
        # search score threshold x nms iou to bring per-frame median into 8..15
        grid_s = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        grid_n = [0.30, 0.40, 0.50]
        rows = []
        cache = [(np.load(f)["boxes"], np.load(f)["score"]) for f in files]
        for s in grid_s:
            for n in grid_n:
                cnts = []
                for b, sc in cache:
                    m = sc >= s
                    if m.sum() == 0:
                        cnts.append(0)
                        continue
                    k = nms(b[m], sc[m], n)
                    cnts.append(len(k))
                cnts = np.array(cnts)
                rows.append({"score_thr": s, "nms_iou": n, "median": float(np.median(cnts)),
                             "mean": float(cnts.mean()), "p10": float(np.percentile(cnts, 10)),
                             "p90": float(np.percentile(cnts, 90)), "frac_zero": float((cnts == 0).mean())})
        cal = pd.DataFrame(rows)
        cal.to_csv(OUT / "instance_filter_calibration_grid.csv", index=False)
        # target: median closest to 10 (COCO ref median 9), inside 8..15
        ok = cal[(cal["median"] >= 8) & (cal["median"] <= 15)].copy()
        if len(ok) == 0:
            ok = cal.copy()
        ok["dist"] = (ok["median"] - 10.0).abs()
        best = ok.sort_values(["dist", "frac_zero"]).iloc[0]
        cfg = {"score_thr": float(best.score_thr), "nms_iou": float(best.nms_iou),
               "median_instances_per_frame": float(best["median"]),
               "coco_reference_median": 9.0, "derived_by": "B1_self_calibration", "seed": SEED}
        (OUT / "instance_filter_config_B1.json").write_text(json.dumps(cfg, indent=2))
        print("CALIBRATED", json.dumps(cfg))
        print(cal.to_string())
        return

    score_thr, nms_iou = args.score_thr, args.nms_iou
    assert score_thr is not None and nms_iou is not None
    from multiprocessing import Pool
    global _S, _N
    _S, _N = score_thr, nms_iou
    with Pool(16) as pool:
        chunks = pool.imap(_worker, files, chunksize=4)
        recs = []
        for k, rr in enumerate(chunks):
            recs.extend(rr)
            if k % 200 == 0:
                print(f"{k}/{len(files)} inst={len(recs)}", flush=True)

    df = pd.DataFrame(recs)
    df.to_parquet(args.out)
    print("instances", len(df), "cases", df.exam_case_id.nunique())
    print("per-frame median", df.groupby("frame").size().median())


if __name__ == "__main__":
    main()
