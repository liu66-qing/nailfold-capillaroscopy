from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--output-index", type=Path, required=True); parser.add_argument("--frames-per-video", type=int, default=4); args = parser.parse_args()
    audit = pd.read_csv(args.audit); rows, errors = [], []
    for number, row in enumerate(audit.itertuples(index=False), 1):
        path = args.data / row.video_path; positions = np.linspace(int(row.start), int(row.start) + int(row.used) - 1, args.frames_per_video, dtype=int); capture = cv2.VideoCapture(str(path)); wanted = set(positions.tolist()); decoded = 0
        try:
            while wanted:
                ok, frame = capture.read()
                if not ok: break
                if decoded in wanted:
                    archive, case = str(row.exam_case_id).split("/"); relative = Path(archive) / case / f"video_{number:04d}_frame_{decoded:04d}.jpg"; destination = args.output_dir / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]): raise IOError(f"failed to write {destination}")
                    rows.append({"exam_case_id": row.exam_case_id, "image_path": relative.as_posix()}); wanted.remove(decoded)
                decoded += 1
        finally: capture.release()
        if wanted: errors.append({"video_path": row.video_path, "missing_positions": sorted(wanted)})
        if number % 20 == 0 or number == len(audit): print(f"{number}/{len(audit)} frames={len(rows)} errors={len(errors)}", flush=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(args.output_index, index=False)
    if errors: raise RuntimeError(f"{len(errors)} videos had missing frames: {errors[:3]}")


if __name__ == "__main__": main()
