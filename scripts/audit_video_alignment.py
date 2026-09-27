#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 of video_kymo_20260925: are the videos the same exam as the report?

For every development case with usable video: find, for each CAPorg still, the
best-matching frame across all of the case's videos (normalised correlation at
256x192), and measure the longest stable segment of each video (global
phase-correlation shift < 1 px between consecutive frames, at 256 width).

Stop rules from preregistration.md §3:
  - share of cases whose best CAPorg match < 0.5 exceeds 50%  -> stop
  - share of cases whose longest stable segment < 3 s exceeds 50% -> stop

Development only; locked-47 containers are never opened.
Run:
  PYTHONIOENCODING=utf-8 python scripts/audit_video_alignment.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
AVAIL = ROOT / "artifacts/experiments/video_availability_20260922/video_availability_by_case.csv"
OUT = ROOT / "artifacts/experiments/video_kymo_20260925"
SIZE = (256, 192)
STABLE_PX = 1.0


def read_gray(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), SIZE).astype(np.float32))
    cap.release()
    return np.asarray(frames), fps


def znorm(a: np.ndarray) -> np.ndarray:
    axes = tuple(range(a.ndim - 2, a.ndim))
    m = a.mean(axis=axes, keepdims=True)
    s = a.std(axis=axes, keepdims=True) + 1e-6
    return (a - m) / s


def longest_stable(frames: np.ndarray, fps: float) -> dict:
    shifts = []
    win = cv2.createHanningWindow(SIZE, cv2.CV_32F)
    for a, b in zip(frames[:-1], frames[1:]):
        (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
        shifts.append(float(np.hypot(dx, dy)))
    shifts = np.asarray(shifts)

    def run(series):
        best = cur = start = best_start = 0
        for i, s in enumerate(series):
            if s < STABLE_PX:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best:
                    best, best_start = cur, start
            else:
                cur = 0
        return best, best_start

    # Two definitions are reported side by side. `raw` (every consecutive pair
    # < 1 px) was the first implementation. `smooth` (centred 5-frame rolling
    # median < 1 px) was added AFTER the first run returned a 0.7 s median; it is
    # the reading of "帧间位移中位" in preregistration §3, but because it was
    # chosen after seeing a number, the stop rule is judged on BOTH and the
    # stricter one is quoted.
    smooth = pd.Series(shifts).rolling(5, center=True, min_periods=1).median().to_numpy()
    rb, _ = run(shifts)
    sb, ss = run(smooth)
    sec = lambda n: round((n + 1) / fps, 2) if fps and n else 0.0
    return dict(stable_frames=int(sb + 1 if sb else 0), stable_start=int(ss),
                stable_seconds=sec(sb), stable_seconds_raw=sec(rb),
                median_shift_px=round(float(np.median(shifts)), 3) if len(shifts) else None)


def main() -> None:
    lab = pd.read_csv(LABELS)
    locked = set(lab[lab.development_fold.isna()].exam_case_id.astype(str))
    av = pd.read_csv(AVAIL)
    av = av[av.in_development & av.usable]
    assert not set(av.case) & locked, "locked case in availability table"

    case_rows, video_rows = [], []
    for case, g in av.groupby("case"):
        vids = []
        for rel in g.file:
            fr, fps = read_gray(DATA / rel)
            st = longest_stable(fr, fps)
            vids.append((rel, fr, fps))
            video_rows.append(dict(case=case, file=rel, frames=len(fr), fps=fps, **st))
        stills = sorted((DATA / case).glob("CAPorg*.jpg"))
        best_per_still = []
        for p in stills:
            # cv2.imread cannot open non-ASCII paths on Windows
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            q = znorm(cv2.resize(im, SIZE).astype(np.float32))
            best = (-1.0, None, None)
            for rel, fr, _ in vids:
                if not len(fr):
                    continue
                c = (znorm(fr) * q).mean(axis=(1, 2))
                i = int(c.argmax())
                if c[i] > best[0]:
                    best = (float(c[i]), rel, i)
            best_per_still.append(best)
        corrs = [b[0] for b in best_per_still]
        vstable = [r["stable_seconds"] for r in video_rows if r["case"] == case]
        vraw = [r["stable_seconds_raw"] for r in video_rows if r["case"] == case]
        case_rows.append(dict(
            case=case, n_videos=len(vids), n_stills=len(stills),
            best_match_max=round(max(corrs), 4) if corrs else None,
            best_match_median=round(float(np.median(corrs)), 4) if corrs else None,
            stills_matched_ge_0_5=int(sum(c >= 0.5 for c in corrs)),
            longest_stable_seconds=max(vstable) if vstable else 0.0,
            longest_stable_seconds_raw=max(vraw) if vraw else 0.0))
        print(case, case_rows[-1]["best_match_max"], case_rows[-1]["longest_stable_seconds"], flush=True)

    cdf = pd.DataFrame(case_rows)
    vdf = pd.DataFrame(video_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(OUT / "alignment_by_case.csv", index=False)
    vdf.to_csv(OUT / "alignment_by_video.csv", index=False)

    n = len(cdf)
    assert cdf.best_match_max.notna().mean() > 0.9, "stills failed to load; fix I/O before reading results"
    low_match = float((cdf.best_match_max.astype(float).fillna(0) < 0.5).mean())
    short = float((cdf.longest_stable_seconds < 3).mean())
    short_raw = float((cdf.longest_stable_seconds_raw < 3).mean())
    doc = dict(
        cases=n,
        share_best_match_below_0_5=round(low_match, 4),
        share_longest_stable_below_3s_smooth=round(short, 4),
        share_longest_stable_below_3s_raw=round(short_raw, 4),
        best_match_max_quantiles=cdf.best_match_max.astype(float).quantile([.1, .25, .5, .75, .9]).round(3).to_dict(),
        stable_seconds_quantiles_smooth=cdf.longest_stable_seconds.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
        stable_seconds_quantiles_raw=cdf.longest_stable_seconds_raw.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
        stop_rule_alignment_triggered=low_match > 0.5,
        stop_rule_stability_triggered_smooth=short > 0.5,
        stop_rule_stability_triggered_raw=short_raw > 0.5,
        note_post_hoc=("smooth definition was added after the first run; "
                       "quote the raw result when the two disagree"),
        locked_cases_seen=0,
    )
    (OUT / "alignment_summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
