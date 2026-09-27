#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract deployed-anchor features under simulated device shifts.

The product will receive images from an unknown new device. We have no labelled
image from that device, so the first question is not "which adaptation helps"
but "how far does the current model move when only the camera changes".

Each condition applies ONE fixed appearance change to every development image,
then runs the unchanged deployed anchor (DINOv2-B/14, 518x686 direct resize,
five poolings). The parameters were fixed before any result was read and span
the range measured on the external devices on disk (HF 640x480, saturation
~24 vs ours ~147; Mendeley 1295x537).

Some conditions change the very thing a field reports (blur vs clarity, colour
vs blood_color). That is intended: the evaluation reports those pairs as
"target-changing" rather than as robustness failures.

DEVELOPMENT ONLY (load_index aborts on any locked-47 id).

  PYTHONIOENCODING=utf-8 python scripts/device_shift_features.py --cond all
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import extract_medical_encoders as E  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

OUT = ROOT / "artifacts" / "experiments" / "device_shift_20260926" / "features"


def wb(im, r, b):
    a = np.asarray(im).astype(np.float32)
    a[..., 0] *= r
    a[..., 2] *= b
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def gamma(im, g):
    a = np.asarray(im).astype(np.float32) / 255.0
    return Image.fromarray(np.clip(255 * a ** g, 0, 255).astype(np.uint8))


def jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def lowres(im, s):
    w, h = im.size
    return im.resize((int(w * s), int(h * s)), Image.BICUBIC).resize((w, h), Image.BICUBIC)


# name -> (function, which field's own target it may legitimately change)
CONDITIONS = {
    "clean": (lambda im: im, None),
    "lowres_0.5": (lambda im: lowres(im, 0.5), "clarity"),
    "jpeg_q30": (lambda im: jpeg(im, 30), None),
    "warm": (lambda im: wb(im, 1.15, 0.85), "blood_color"),
    "cool": (lambda im: wb(im, 0.85, 1.20), "blood_color"),
    "desat_0.5": (lambda im: ImageEnhance.Color(im).enhance(0.5), "blood_color"),
    "dark_g1.4": (lambda im: gamma(im, 1.4), None),
    "bright_g0.7": (lambda im: gamma(im, 0.7), None),
    "blur_s1.5": (lambda im: im.filter(ImageFilter.GaussianBlur(1.5)), "clarity"),
    "hf_like": (lambda im: gamma(wb(ImageEnhance.Color(lowres(im, 0.625)).enhance(0.3),
                                    0.95, 1.15), 0.8), "blood_color"),
}


def prep(path, fn, dep):
    with Image.open(path) as im:
        im = fn(im.convert("RGB"))
        t = TF.resize(im, list(dep), interpolation=TF.InterpolationMode.BICUBIC)
        return TF.normalize(TF.to_tensor(t), *E.IMAGENET), (dep[0], dep[1])


def run(cond, model, grid, ix, device, bs=8, topk=16):
    fn, _ = CONDITIONS[cond]
    out = OUT / cond
    if (out / "done.json").exists():
        print("skip", cond)
        return
    out.mkdir(parents=True, exist_ok=True)
    dep = E.ARMS["anchor_dinov2b_deployed"]["deployed_res"]
    n, dim = len(ix), model.num_features
    sinks = {k: open_memmap(out / ("features_%s.npy" % k), mode="w+",
                            dtype=np.float16, shape=(n, dim)) for k in E.POOLINGS}
    with torch.inference_mode():
        for s in range(0, n, bs):
            rows = ix.image_path.iloc[s:s + bs].tolist()
            pp = [prep(E.IMAGE_ROOT / r, fn, dep) for r in rows]
            x = torch.stack([p[0] for p in pp]).to(device)
            tok = model.forward_features(x)
            vals = E.pool(tok, model.num_prefix_tokens, grid, [p[1] for p in pp], 14, topk)
            for k, v in vals.items():
                sinks[k][s:s + len(rows)] = v.float().cpu().numpy().astype(np.float16)
            if (s // bs) % 50 == 0:
                print(cond, s + len(rows), n, flush=True)
    for v in sinks.values():
        v.flush()
    ix[["exam_case_id", "image_path"]].to_csv(out / "index.csv", index=False)
    (out / "done.json").write_text(json.dumps(dict(cond=cond, images=n, locked_cases_seen=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="all")
    a = ap.parse_args()
    ix = E.load_index()
    device = torch.device("cuda:0")
    cfg = E.ARMS["anchor_dinov2b_deployed"]
    model, _ = E.build(cfg, device, img_size=list(cfg["deployed_res"]))
    grid = (cfg["deployed_res"][0] // 14, cfg["deployed_res"][1] // 14)
    conds = list(CONDITIONS) if a.cond == "all" else a.cond.split(",")
    for c in conds:
        run(c, model, grid, ix, device)


if __name__ == "__main__":
    main()
