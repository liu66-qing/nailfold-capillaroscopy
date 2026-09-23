#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream C: corpus-wide video availability for the time-base fields.

Prior video work exists and is NOT repeated here. Already on disk:
  artifacts/video_baseline_metrics_v2.json, video_hybrid_cv_metrics.json,
  video_v3_cv_metrics.json, video_static_ft_cv_metrics.json   -- four modelling
      variants, already run on exactly the five time-base fields
      (flow_state, vasomotion, rbc_aggregation, wbc_count, microthrombus)
  artifacts/invalid_video_repro/20260804_123546/report.json   -- single-case
      pipeline behaviour when video fails to decode
  scripts/extract_video_features.py, train_video_*.py         -- the pipeline

What none of them answer, and what this script answers: across the whole
corpus, HOW MANY development cases actually have decodable video, how many
frames, and how that intersects the fields that need a time base. Without that
number the earlier modelling results cannot be read, because a weak result on
a third of the cohort means something different from a weak result on all of it.

This is an INVENTORY. No model is trained, no field is predicted, nothing is
resized or re-encoded. It only opens each file, reads frame count / fps /
resolution, and joins against the label table.

Note on scope: the earlier pipeline globbed `recovered_archive*/*/*.mp4`, i.e.
converted clips only. The raw corpus holds .avi/.mpg/.wmv, so a count based on
.mp4 alone would understate availability. This script counts the raw containers.

DEVELOPMENT ONLY. Cases are joined to development_fold; locked-47 rows are
counted for completeness of the denominator but never opened, and no locked
case appears in any per-case output.

Run:
  PYTHONIOENCODING=utf-8 python scripts/check_video_availability.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
OUT_DIR = ROOT / "artifacts" / "experiments" / "video_availability_20260922"
OUT_JSON = OUT_DIR / "video_availability.json"
OUT_CSV = OUT_DIR / "video_availability_by_case.csv"

VIDEO_EXT = {".avi", ".mpg", ".mpeg", ".wmv", ".mp4", ".mov", ".mkv", ".asf"}

# Fields that need a time base to be observed at all (nailfold-field-units-root-cause)
TIME_BASE_FIELDS = ["flow_state", "vasomotion", "rbc_aggregation",
                    "wbc_count", "microthrombus"]

PRIOR_WORK = {
    "artifacts/video_baseline_metrics_v2.json": "video baseline, 5 time-base fields",
    "artifacts/video_hybrid_cv_metrics.json": "hybrid static+video CV",
    "artifacts/video_v3_cv_metrics.json": "v3 video CV, per-class recall recorded",
    "artifacts/video_static_ft_cv_metrics.json": "static fine-tuned comparison",
    "artifacts/invalid_video_repro/20260804_123546/report.json":
        "single-case behaviour when video cannot be decoded",
}


def case_of(p: Path) -> str | None:
    parts = p.relative_to(DATA).parts
    return "%s/%s" % (parts[0], parts[1]) if len(parts) >= 2 else None


def probe(path: Path) -> dict:
    """Open the container and read what it reports. Decodes one frame only."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return dict(opens=False, frames=0, fps=None, width=None, height=None,
                    first_frame_decodes=False)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, _ = cap.read()
    cap.release()
    return dict(opens=True, frames=frames, fps=round(fps, 3) if fps else None,
                width=w, height=h, first_frame_decodes=bool(ok))


def main() -> None:
    lab = pd.read_csv(LABELS)
    key = "exam_case_id"
    dev = lab[lab.development_fold.notna()][key].astype(str).tolist()
    locked = set(lab[lab.development_fold.isna()][key].astype(str))
    dev_set = set(dev)
    print("development cases %d, locked %d" % (len(dev_set), len(locked)))

    files = [p for p in DATA.rglob("*")
             if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    print("video-like files found: %d" % len(files))

    rows = []
    for p in sorted(files):
        c = case_of(p)
        if c is None or c in locked:      # locked containers are never opened
            continue
        info = probe(p)
        info.update(case=c, file=str(p.relative_to(DATA)),
                    ext=p.suffix.lower(), bytes=p.stat().st_size,
                    in_development=c in dev_set)
        rows.append(info)
        if len(rows) % 50 == 0:
            print("probed %d" % len(rows), flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no probeable video outside locked-47")
        return

    df["usable"] = df.opens & df.first_frame_decodes & (df.frames > 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["case", "file"]).to_csv(OUT_CSV, index=False)

    d = df[df.in_development]
    by_case = d.groupby("case").agg(
        files=("file", "size"), usable_files=("usable", "sum"),
        total_frames=("frames", "sum"), max_frames=("frames", "max"))
    with_any = set(d.case.unique())
    with_usable = set(d[d.usable].case.unique())

    # frame-count sufficiency: a flow reading needs more than a couple of frames
    thresholds = {}
    for t in (1, 5, 10, 30, 60):
        thresholds["cases_with_max_frames_ge_%d" % t] = int(
            (by_case.max_frames >= t).sum())

    field_cov = {}
    for f in TIME_BASE_FIELDS:
        if f not in lab.columns:
            field_cov[f] = dict(present_in_label_table=False)
            continue
        sub = lab[lab.development_fold.notna()].copy()
        sub["_lab"] = sub[f].map(lambda s: "" if pd.isna(s) else str(s).strip())
        labelled = set(sub[sub._lab != ""][key].astype(str))
        field_cov[f] = dict(
            present_in_label_table=True,
            development_cases_with_a_label=len(labelled),
            of_those_with_usable_video=len(labelled & with_usable),
            of_those_with_no_video_at_all=len(labelled - with_any),
            fraction_with_usable_video=(round(
                len(labelled & with_usable) / len(labelled), 4)
                if labelled else None))

    res_counts = Counter("%dx%d" % (r.width, r.height)
                         for r in d[d.usable].itertuples())

    doc = dict(
        purpose=("Stream C: establish how many DEVELOPMENT cases have decodable "
                 "video and how that intersects the five fields that need a "
                 "time base. Inventory only; no model, no prediction."),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        prior_work_cited_not_repeated=PRIOR_WORK,
        denominators=dict(development_cases=len(dev_set),
                          locked_cases_excluded_unopened=len(locked)),
        files=dict(video_like_files_in_data=len(files),
                   probed_outside_locked=int(len(df)),
                   by_extension=df.ext.value_counts().to_dict(),
                   usable=int(df.usable.sum()),
                   open_but_zero_frames=int((df.opens & (df.frames <= 0)).sum()),
                   fail_to_open=int((~df.opens).sum())),
        development_coverage=dict(
            cases_with_any_video_file=len(with_any),
            cases_with_usable_video=len(with_usable),
            fraction_of_development_with_usable_video=round(
                len(with_usable) / len(dev_set), 4),
            cases_with_no_video_at_all=len(dev_set - with_any),
            frames_per_case=dict(
                median=float(by_case.max_frames.median()),
                mean=round(float(by_case.max_frames.mean()), 2),
                min=int(by_case.max_frames.min()),
                max=int(by_case.max_frames.max())),
            **thresholds),
        resolutions_of_usable_video=dict(res_counts.most_common()),
        time_base_field_coverage=field_cov,
        readings=[
            ("the fraction of development cases with usable video is the ceiling "
             "on any video result: a model cannot be evaluated on cases that "
             "have no time base to read"),
            ("this does not make the five time-base fields deliverable; it only "
             "says how large the eligible subset is. The four prior video "
             "modelling runs already reported weak BA on these fields and are "
             "not rerun here"),
        ],
        forbidden=[
            ("this inventory is not a supervision source: counting frames is not "
             "region morphology supervision, and detection counts are not either"),
            ("no micron or per-mm quantity is derived from any video here"),
            "locked-47 containers were never opened; no locked case is listed",
        ],
        locked_cases_seen=0,
    )
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(dict(
        development_cases=len(dev_set),
        cases_with_usable_video=len(with_usable),
        fraction=doc["development_coverage"]
        ["fraction_of_development_with_usable_video"],
        median_frames=doc["development_coverage"]["frames_per_case"]["median"],
        field_coverage={k: v.get("fraction_with_usable_video")
                        for k, v in field_cov.items()},
        out=str(OUT_JSON.name)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
