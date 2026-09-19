#!/usr/bin/env python
"""Control 2: extract features from a CONVENTIONAL ImageNet backbone.

The point of this control is to answer one question: does DINOv2 actually beat an
ordinary ImageNet-pretrained backbone on this data? If it does not, the backbone
is not the bottleneck and no amount of encoder work will help.

Kept deliberately comparable to extract_dinov2_spatial.py: same image list, same
bicubic resize, same ImageNet normalisation, same frozen-and-eval regime, same
output layout (index.csv + features_mean.npy) so evidence_group4_controls.py can
read it without special-casing.

DEVELOPMENT ONLY by default. --roles filters on evaluation_role and the script
refuses to write if a locked case would be included unless explicitly asked.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.functional as TF
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_batch(paths, root, res, device):
    out = []
    for p in paths:
        with Image.open(os.path.join(root, str(p))) as im:
            im = im.convert("RGB")
            t = TF.resize(im, list(res), interpolation=TF.InterpolationMode.BICUBIC)
            t = TF.normalize(TF.to_tensor(t), MEAN, STD)
        out.append(t)
    return torch.stack(out).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18.a1_in1k")
    ap.add_argument("--index", required=True,
                    help="csv with exam_case_id,image_path")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role", default="development")
    ap.add_argument("--res", type=int, nargs=2, default=[518, 686])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    ix = pd.read_csv(a.index)
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    role = dict(zip(m.exam_case_id, m.evaluation_role))
    ix["role"] = ix.exam_case_id.map(role)
    ix = ix[ix.role == a.role].reset_index(drop=True)
    if a.role == "development":
        leaked = set(ix.exam_case_id) & set(
            m.exam_case_id[m.evaluation_role == "locked_test"])
        assert not leaked, f"locked cases in a development extraction: {leaked}"
    print("images:", len(ix), "cases:", ix.exam_case_id.nunique())

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = timm.create_model(a.model, pretrained=True, num_classes=0,
                              global_pool="avg").to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(a.model, "params", n_par)

    feats = []
    with torch.no_grad():
        for i in range(0, len(ix), a.batch):
            batch = ix.image_path.iloc[i:i + a.batch].tolist()
            x = load_batch(batch, a.image_root, a.res, dev)
            f = model(x)
            feats.append(f.float().cpu().numpy())
            if i % (a.batch * 20) == 0:
                print(" ", i, "/", len(ix), flush=True)
    F = np.concatenate(feats, 0)
    print("features", F.shape)

    os.makedirs(a.out_dir, exist_ok=True)
    ix[["exam_case_id", "image_path"]].to_csv(
        os.path.join(a.out_dir, "index.csv"), index=False)
    np.save(os.path.join(a.out_dir, "features_mean.npy"), F)
    with open(os.path.join(a.out_dir, "metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "what": "CONTROL 2: conventional ImageNet-pretrained backbone, frozen, "
                    "global average pooled. Comparison point for DINOv2.",
            "model": a.model, "parameters": int(n_par),
            "pretrained": "ImageNet-1k (timm default weights for this tag)",
            "frozen": True, "augmentation": "none",
            "input_size": a.res, "normalisation": "ImageNet mean/std",
            "evaluation_role": a.role,
            "locked_cases_seen": 0 if a.role == "development" else "N/A",
            "cases": int(ix.exam_case_id.nunique()), "images": int(len(ix)),
            "dim": int(F.shape[1]),
            "limitations": [
                "this is a FROZEN feature probe of an ImageNet backbone, not a "
                "fully finetuned supervised CNN. It bounds what an ordinary "
                "backbone's representation offers under the delivered estimator; "
                "a finetuned CNN could do better and that remains untested.",
                "global average pooling only; no multi-scale, no spatial "
                "statistics, so it is a slightly weaker probe than the 5-pooling "
                "DINOv2 ensemble it is compared against",
            ],
        }, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
