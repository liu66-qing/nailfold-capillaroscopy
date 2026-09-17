"""Extract frozen DINOv2-base per-frame tokens for all dev cases and cache to disk.

Caches CLS (768) + raw patch tokens are too big, so we cache CLS + the patch-token
tensor mean/second-moment? No -- GatedPool is a *learned* head, so we must keep the
patch tokens. We keep them at reduced precision (float16) to fit on disk.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image

MANIFEST = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
IMAGE_INDEX = "/root/nailfold/artifacts/features/image_index.csv"
DATA_ROOT = "/root/nailfold/data"
WEIGHTS = "/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth"
OUTDIR = "/root/autodl-tmp/nailfold/exp_i_feats"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda:%d" % args.gpu
    os.makedirs(OUTDIR, exist_ok=True)

    man = pd.read_csv(MANIFEST, dtype={"exam_case_id": str})
    man = man[man.evaluation_role == "development"]
    cases = sorted(set(man.exam_case_id))
    idx = pd.read_csv(IMAGE_INDEX, dtype={"exam_case_id": str})
    idx = idx[idx.exam_case_id.isin(cases)]
    print("cases=%d frames=%d" % (idx.exam_case_id.nunique(), len(idx)), flush=True)

    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0
    )
    sd = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("blocks") for k in sd):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
    print("  sample missing:", list(missing)[:6], flush=True)
    print("  sample unexpected:", list(unexpected)[:6], flush=True)
    model.eval().to(dev)

    cfg = timm.data.resolve_model_data_config(model)
    print("data cfg:", cfg, flush=True)
    tf = timm.data.create_transform(**cfg, is_training=False)

    done = 0
    for case in cases:
        safe = case.replace("/", "__")
        out = os.path.join(OUTDIR, safe + ".npz")
        if os.path.exists(out):
            done += 1
            continue
        rows = idx[idx.exam_case_id == case]
        cls_list, patch_list = [], []
        for _, r in rows.iterrows():
            p = os.path.join(DATA_ROOT, r["image_path"])
            try:
                img = Image.open(p).convert("RGB")
            except OSError as exc:
                print("  [skip]", p, exc, flush=True)
                continue
            x = tf(img).unsqueeze(0).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.forward_features(x)  # (1, 1+N, 768) incl. prefix
            toks = toks.float().squeeze(0).cpu()
            npre = getattr(model, "num_prefix_tokens", 1)
            cls_list.append(toks[0].numpy().astype(np.float16))
            patch_list.append(toks[npre:].numpy().astype(np.float16))
        if not cls_list:
            print("  [noframes]", case, flush=True)
            continue
        np.savez_compressed(
            out,
            cls=np.stack(cls_list),          # (F, 768)
            patch=np.stack(patch_list),      # (F, N, 768)
        )
        done += 1
        if done % 20 == 0:
            print("  %d/%d cases" % (done, len(cases)), flush=True)
    print("DONE cached=%d -> %s" % (done, OUTDIR), flush=True)


if __name__ == "__main__":
    sys.exit(main())
