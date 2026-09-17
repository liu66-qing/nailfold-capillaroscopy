from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--cap-index", type=Path, required=True); parser.add_argument("--video-index", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    cap = pd.read_csv(args.cap_index); video = pd.read_csv(args.video_index); cap.image_path = "data/domain_adapted/" + cap.image_path.astype(str); video.image_path = "artifacts/video/stable_training_frames/" + video.image_path.astype(str); combined = pd.concat((cap, video), ignore_index=True)
    if combined.duplicated(["exam_case_id", "image_path"]).any(): raise ValueError("duplicate images")
    args.output.parent.mkdir(parents=True, exist_ok=True); combined.to_csv(args.output, index=False); print(f"images={len(combined)} cases={combined.exam_case_id.nunique()} cap={len(cap)} video={len(video)}")


if __name__ == "__main__": main()
