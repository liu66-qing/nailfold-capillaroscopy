from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def transcode(job: tuple[Path, Path, int, int]) -> tuple[str, str]:
    source, destination, width, fps = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"scale={width}:-2",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return str(source), "ok"
    except subprocess.CalledProcessError as error:
        return str(source), error.stderr.strip() or f"ffmpeg exit {error.returncode}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    files = pd.read_csv(args.files)
    labels = pd.read_csv(args.labels)
    usable_cases = set(
        labels.loc[labels["flow_state"].notna(), "exam_case_id"].astype(str)
    )
    videos = files[
        (files["role"] == "video") & files["exam_case_id"].isin(usable_cases)
    ].copy()
    jobs: list[tuple[Path, Path, int, int]] = []
    for row in videos.to_dict("records"):
        relative = Path(str(row["path"]))
        destination = (args.output_root / relative).with_suffix(".mp4")
        jobs.append((args.data_root / relative, destination, args.width, args.fps))

    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(transcode, jobs), start=1):
            if result[1] != "ok":
                errors.append(result)
            if index % 10 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} errors={len(errors)}", flush=True)
    if errors:
        for source, error in errors:
            print(source, error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
