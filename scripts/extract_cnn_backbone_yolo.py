#!/usr/bin/env python
"""Control 2, offline version: features from a CONVENTIONAL CNN backbone.

The server has no internet, so timm/HF ImageNet weights cannot be downloaded.
The only conventional CNN weights on disk are the COCO-pretrained YOLO11 files, so
this control uses the YOLO11m-seg CNN trunk (layers 0..10, ending at SPPF/C2PSA),
frozen, global-average-pooled.

WHAT THIS CONTROL DOES AND DOES NOT SHOW
  - It DOES answer: does a conventional convolutional backbone's frozen
    representation, under the identical estimator and identical folds, reach
    DINOv2's numbers? If yes, the backbone is not the bottleneck.
  - It does NOT show what a fully finetuned supervised CNN would do.
  - The pretraining corpus is COCO detection, not ImageNet classification. That is
    a real difference from the control the user asked for and is stated in the
    metadata rather than glossed over.

Same resize, same ImageNet normalisation, same frozen+eval regime, same output
layout as the DINOv2 store. DEVELOPMENT ONLY; asserts no locked case.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
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


class Trunk(torch.nn.Module):
    """YOLO11 backbone layers 0..last_idx, run sequentially.

    Layers 0..10 of a YOLO11-seg model form a plain feed-forward CNN trunk
    (Conv/C3k2/SPPF/C2PSA) with no skip connections reaching outside it, so a
    sequential forward is exact for that prefix.
    """

    def __init__(self, layers, last_idx=10):
        super().__init__()
        self.layers = torch.nn.ModuleList(list(layers)[:last_idx + 1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11m-seg.pt")
    ap.add_argument("--last-idx", type=int, default=10)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role", default="development")
    ap.add_argument("--res", type=int, nargs=2, default=[518, 686])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    ix = pd.read_csv(a.index)
    ix["exam_case_id"] = ix.exam_case_id.astype(str)
    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    ix["role"] = ix.exam_case_id.map(dict(zip(m.exam_case_id, m.evaluation_role)))
    ix = ix[ix.role == a.role].reset_index(drop=True)
    if a.role == "development":
        leaked = set(ix.exam_case_id) & set(
            m.exam_case_id[m.evaluation_role == "locked_test"])
        assert not leaked, f"locked cases in a development extraction: {leaked}"
    print("images:", len(ix), "cases:", ix.exam_case_id.nunique(), flush=True)

    from ultralytics import YOLO
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    yolo = YOLO(a.weights)
    trunk = Trunk(yolo.model.model, a.last_idx).to(dev).eval().float()
    n_par = sum(p.numel() for p in trunk.parameters())
    print("trunk params", n_par, flush=True)

    feats = []
    with torch.no_grad():
        for i in range(0, len(ix), a.batch):
            x = load_batch(ix.image_path.iloc[i:i + a.batch].tolist(),
                           a.image_root, a.res, dev)
            f = trunk(x)                      # B,C,H,W
            f = f.mean(dim=(2, 3))            # global average pool
            feats.append(f.float().cpu().numpy())
            if i % (a.batch * 25) == 0:
                print(" ", i, "/", len(ix), flush=True)
    F = np.concatenate(feats, 0)
    print("features", F.shape, flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    ix[["exam_case_id", "image_path"]].to_csv(
        os.path.join(a.out_dir, "index.csv"), index=False)
    np.save(os.path.join(a.out_dir, "features_mean.npy"), F)
    with open(os.path.join(a.out_dir, "metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "what": "CONTROL 2: conventional CNN backbone, frozen, global average "
                    "pooled. Comparison point for DINOv2 under an identical "
                    "estimator and identical folds.",
            "weights_file": a.weights,
            "pretraining": "COCO detection/segmentation (NOT ImageNet "
                           "classification -- the server is offline and no "
                           "ImageNet weights are on disk)",
            "trunk_layers": f"model.model[0..{a.last_idx}] "
                            "(Conv/C3k2/SPPF/C2PSA)",
            "parameters": int(n_par), "frozen": True, "augmentation": "none",
            "input_size": a.res, "normalisation": "ImageNet mean/std",
            "pooling": "global average over the final feature map",
            "dim": int(F.shape[1]),
            "evaluation_role": a.role,
            "locked_cases_seen": 0 if a.role == "development" else "N/A",
            "cases": int(ix.exam_case_id.nunique()), "images": int(len(ix)),
            "limitations": [
                "COCO-pretrained, not ImageNet-pretrained: a real deviation from "
                "the requested control, stated rather than hidden",
                "frozen feature probe, NOT a finetuned supervised CNN; a "
                "finetuned CNN remains untested",
                "global average pooling only, so this is a slightly weaker probe "
                "than the 5-pooling DINOv2 ensemble it is compared against; if it "
                "MATCHES DINOv2 despite that handicap the conclusion is strong, "
                "if it loses slightly the conclusion is weak",
            ],
        }, fh, indent=2, ensure_ascii=False)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
