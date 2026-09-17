"""Extract DINOv2 features with SPATIAL pooling statistics, not just a global mean.

Why this and not another mean-pool run:

The 9 dead fields are almost all sparse localized findings -- hemorrhage,
microthrombus, rbc_aggregation, crossings. A bleed can occupy well under 1% of a
1024x768 frame. Both features we have collapse that:
  features_cls.npy        one global token per image
  features_meanpatch.npy  average over 37x49=1813 patches
so a lesion covering ~18 patches is diluted ~100x. A max/top-k over patches keeps
it. That is a different mechanism from the closed frozen-feature MIL path, which
pooled over FRAMES (each frame already mean-pooled) -- never over SPACE.

Outputs per image (all float16, 768-dim each unless noted):
  cls          CLS token                        (reference, same as before)
  mean         mean over patches                (reference, same as before)
  max          per-dimension max over patches   NEW  -- strongest local response
  topk_mean    mean of top-k patches per dim    NEW  -- max, but less noisy
  std          per-dimension std over patches   NEW  -- spatial heterogeneity

std is included because "is the frame uniform or patchy" is exactly what
clarity/exudation graders describe verbally, and no existing feature encodes it.

Hard constraints (AUDIT section 14):
  - locked-47 is never read. Case ids come from development_fold non-NaN only,
    and the run aborts if any locked case id appears in the index.
  - writes only into its own --out directory; touches no existing artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.functional as TF
from numpy.lib.format import open_memmap
from PIL import Image
from timm.layers import resample_abs_pos_embed

# ImageNet stats, matching extract_dinov2_fullfov.py
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

PRESETS = {
    "native": (518, 686),        # aspect-preserving, full field of view
    "native_hi": (770, 1022),    # closer to the true 1024x768
    "square_baseline": (518, 518),  # control: same 75% centre crop as the audit
}


def load_batch(paths, root, res, device, square):
    """Byte-identical preprocessing to extract_dinov2_fullfov.load_batch."""
    out = []
    for p in paths:
        with Image.open(Path(root) / str(p)) as im:
            im = im.convert("RGB")
            if square:
                t = TF.resize(im, 518, interpolation=TF.InterpolationMode.BICUBIC)
                t = TF.center_crop(t, [518, 518])
            else:
                t = TF.resize(im, list(res), interpolation=TF.InterpolationMode.BICUBIC)
            t = TF.normalize(TF.to_tensor(t), MEAN, STD)
        out.append(t)
    return torch.stack(out).to(device)


def build(weights, res, device):
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0,
        img_size=res, dynamic_img_size=True)
    sd = dict(torch.load(weights, map_location="cpu", weights_only=True))
    want = (model.patch_embed.num_patches + 1, 768)
    if tuple(sd["pos_embed"].shape[1:]) != want:
        sd["pos_embed"] = resample_abs_pos_embed(
            sd["pos_embed"], new_size=list(model.patch_embed.grid_size),
            num_prefix_tokens=1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [u for u in unexpected if u != "mask_token"]
    if missing or unexpected:
        raise RuntimeError("state mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    return model.to(device).eval()


POOLINGS = ["cls", "mean", "max", "topk_mean", "std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="native")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    index = pd.read_csv(a.index)
    roles = pd.read_csv(a.roles, usecols=["exam_case_id", "evaluation_role"])
    index = index.merge(roles, on="exam_case_id", validate="many_to_one")
    locked = set(index[index.evaluation_role.ne("development")].exam_case_id)
    index = index[index.evaluation_role.eq("development")].drop(columns="evaluation_role")
    index = index.reset_index(drop=True)
    if index.exam_case_id.nunique() != 186:
        raise RuntimeError("expected 186 development cases, got %d"
                           % index.exam_case_id.nunique())
    assert not (set(index.exam_case_id) & locked), "locked case leaked into extraction index"

    res = PRESETS[a.preset]
    square = a.preset == "square_baseline"

    dev = torch.device(a.device)
    model = build(a.weights, res, dev)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    n = len(index)
    sinks = {k: open_memmap(a.output_dir / ("features_%s.npy" % k), mode="w+",
                            dtype=np.float16, shape=(n, 768)) for k in POOLINGS}

    with torch.inference_mode():
        for s in range(0, n, a.batch_size):
            chunk = index.image_path.iloc[s:s + a.batch_size].tolist()
            x = load_batch(chunk, a.image_root, res, dev, square)
            tok = model.forward_features(x)
            npre = model.num_prefix_tokens
            cls, patch = tok[:, 0].float(), tok[:, npre:].float()
            kk = min(a.topk, patch.shape[1])
            vals = {
                "cls": cls,
                "mean": patch.mean(1),
                "max": patch.max(1).values,
                "topk_mean": patch.topk(kk, dim=1).values.mean(1),
                "std": patch.std(1),
            }
            e = s + len(chunk)
            for k, v in vals.items():
                sinks[k][s:e] = v.cpu().numpy().astype(np.float16)
            if (s // a.batch_size) % 20 == 0:
                print("%d/%d" % (e, n), flush=True)
    for v in sinks.values():
        v.flush()

    index[["exam_case_id", "image_path"]].to_csv(a.output_dir / "index.csv", index=False)
    ref = pd.read_csv(a.index)
    meta = {
        "schema_version": "dinov2-spatial-features/1.0",
        "preset": a.preset, "input_size": list(res),
        "patch_grid": list(model.patch_embed.grid_size),
        "square_center_crop": square,
        "poolings": POOLINGS, "topk": a.topk,
        "evaluation_role": "development", "locked_cases_seen": 0,
        "cases": int(index.exam_case_id.nunique()), "images": n,
        "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        "index_order_matches_source": bool(
            len(ref) == n and (ref.image_path.tolist() == index.image_path.tolist())),
        "limitations": [
            "development-set features only; no locked-47 involvement",
            "pos_embed bicubically resampled from the 37x37 pretrain grid",
            "max/topk/std are frozen-encoder statistics, not learned detectors: "
            "they can only surface what DINOv2 already separates without finetuning",
            "no lesion localisation is claimed; patch indices are not retained",
        ],
    }
    (a.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
