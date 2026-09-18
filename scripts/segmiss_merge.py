#!/usr/bin/env python
"""Build augmented image-level geometry = baseline conf=0.05 geometry + miss features.

This exists so that scripts/eval_geom_imagelevel.py can be reused UNMODIFIED:
that script takes --geom pointing at any image-level geometry CSV and applies
the fixed ruler (accuracy minus one fixed majority answer, baseline recomputed
inside 2000 case-level bootstraps, CI lower bound > 0, development_fold only).

PRESPECIFIED CONFIGURATION (declared before seeing any field result):
  arm "c05_plus_miss" = all baseline c05 geometry columns UNION all miss
  columns, evaluated through the ruler's own prespecified 'concat' arm.
Anything else reported is labelled exploratory / non-deliverable.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",
                    default="/root/autodl-tmp/nailfold/artifacts/features/seg_instance_c05/image_geometry.csv")
    ap.add_argument("--miss", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="union", choices=["union", "miss_only"])
    args = ap.parse_args()

    B = pd.read_csv(args.baseline)
    B["exam_case_id"] = B["exam_case_id"].astype(str)
    B["image_path"] = B["image_path"].astype(str)

    M = pd.read_csv(args.miss)
    M["exam_case_id"] = M["exam_case_id"].astype(str)
    M["image_path"] = M["image_path"].astype(str)
    drop = [c for c in ("img_w", "img_h", "fold", "n_yolo") if c in M.columns]
    M = M.drop(columns=drop)

    keys = ["exam_case_id", "image_path"]
    if args.mode == "miss_only":
        out = B[keys + ["fold"]].merge(M, on=keys, how="inner")
    else:
        out = B.merge(M, on=keys, how="inner")
    out.to_csv(args.out, index=False)
    print(f"mode={args.mode} rows={len(out)} cols={out.shape[1]} "
          f"cases={out.exam_case_id.nunique()} -> {args.out}")


if __name__ == "__main__":
    main()
