#!/usr/bin/env python
"""Locked-47 spatial-pooled DINOv2 features, method-identical to features_spatial/native.

Why this file exists
--------------------
`extract_dinov2_spatial.py` refuses to touch locked-47 by design (it filters to
evaluation_role == development and asserts the intersection is empty), because it
feeds the training features. This is its locked-side twin: same encoder weights,
same preset, same preprocessing, same poolings, same top-k.

It imports `load_batch`, `build` and `POOLINGS` from that file rather than
restating them, so the preprocessing cannot silently diverge between the features
the classifier was fitted on and the features it is tested on. A divergence there
would look exactly like a generalisation failure and be misattributed.

Provenance that must match features_spatial/native/metadata.json:
  weights_sha256  0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
  preset          native   -> input 518x686, patch grid 37x49
  poolings        cls, mean, max, topk_mean, std
  topk            16
The script asserts the weights hash and aborts if it differs.

No labels are read. No metric is computed. Writes only into its own directory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_dinov2_spatial import POOLINGS, PRESETS, build, load_batch  # noqa: E402

EXPECT_WEIGHTS_SHA = ("0b8b82f85de91b424aded121c7e1dcc2b7bc6d0a"
                      "deea651bf73a13307fad8c73")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=Path("artifacts/features/dinov2_locked/index.csv"))
    ap.add_argument("--roles", type=Path,
                    default=Path("artifacts/manifest/locked_evaluation_v1_reviewed.csv"))
    ap.add_argument("--image-root", type=Path,
                    default=Path("/root/autodl-tmp/nailfold/data"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/root/autodl-tmp/nailfold/artifacts/features_spatial/native_locked"))
    ap.add_argument("--weights", type=Path,
                    default=Path("/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"))
    ap.add_argument("--preset", choices=list(PRESETS), default="native")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    got = sha256(a.weights)
    assert got == EXPECT_WEIGHTS_SHA, (
        f"encoder weights differ from the delivered run\n  expected {EXPECT_WEIGHTS_SHA}"
        f"\n  got      {got}\nrefusing to produce non-comparable features")

    roles = pd.read_csv(a.roles)
    roles["exam_case_id"] = roles["exam_case_id"].astype(str)
    locked = set(roles.loc[roles.evaluation_role == "locked_test", "exam_case_id"])
    dev = set(roles.loc[roles.evaluation_role == "development", "exam_case_id"])
    assert len(locked) == 47, f"expected 47 locked cases, got {len(locked)}"

    index = pd.read_csv(a.index)
    index["exam_case_id"] = index["exam_case_id"].astype(str)
    index = index[index.exam_case_id.isin(locked)].reset_index(drop=True)
    assert not (set(index.exam_case_id) & dev), "a development case reached the locked index"
    assert index.exam_case_id.nunique() == 47, \
        f"index covers {index.exam_case_id.nunique()} locked cases, expected 47"

    res = PRESETS[a.preset]
    square = a.preset == "square_baseline"
    device = torch.device(a.device)
    model = build(a.weights, res, device)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    n = len(index)
    print(f"locked images {n} / cases {index.exam_case_id.nunique()} "
          f"/ preset {a.preset} {res}", flush=True)
    sinks = {k: open_memmap(a.output_dir / ("features_%s.npy" % k), mode="w+",
                            dtype=np.float16, shape=(n, 768)) for k in POOLINGS}

    with torch.inference_mode():
        for s in range(0, n, a.batch_size):
            chunk = index.image_path.iloc[s:s + a.batch_size].tolist()
            x = load_batch(chunk, a.image_root, res, device, square)
            tok = model.forward_features(x)
            npre = model.num_prefix_tokens
            cls, patch = tok[:, 0].float(), tok[:, npre:].float()
            kk = min(a.topk, patch.shape[1])
            vals = {"cls": cls, "mean": patch.mean(1),
                    "max": patch.max(1).values,
                    "topk_mean": patch.topk(kk, dim=1).values.mean(1),
                    "std": patch.std(1)}
            e = s + len(chunk)
            for k, v in vals.items():
                sinks[k][s:e] = v.cpu().numpy().astype(np.float16)
            if (s // a.batch_size) % 10 == 0:
                print("%d/%d" % (e, n), flush=True)
    for v in sinks.values():
        v.flush()

    index[["exam_case_id", "image_path"]].to_csv(a.output_dir / "index.csv", index=False)
    meta = {
        "schema_version": "dinov2-spatial-features/1.0",
        "preset": a.preset, "input_size": list(res),
        "square_center_crop": square, "poolings": POOLINGS, "topk": a.topk,
        "evaluation_role": "locked_test", "locked_test_examined": True,
        "development_cases_seen": 0,
        "cases": int(index.exam_case_id.nunique()), "images": n,
        "weights_sha256": got,
        "matches_development_features": "features_spatial/native (same weights "
                                        "sha256, preset, poolings, topk, and the "
                                        "same load_batch/build functions imported "
                                        "from extract_dinov2_spatial.py)",
        "limitations": [
            "locked-47 test features; producing them consumes the held-out cohort",
            "pos_embed bicubically resampled from the 37x37 pretrain grid",
            "max/topk/std are frozen-encoder statistics, not learned detectors",
            "features only; this file makes no accuracy claim about any field",
        ],
    }
    (a.output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
