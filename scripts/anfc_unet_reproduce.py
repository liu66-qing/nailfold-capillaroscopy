#!/usr/bin/env python
"""Reproduce the ANFC U-Net stage (arXiv 2312.05930, repo Image_Segmentation).

Faithful to the repo's config.yaml: their U_Net class (imported unchanged from
third_party/ANFC), green channel only, Normalize(0.5, 0.5), BCE on sigmoid,
Adam lr 2e-4 betas (0.5, 0.999), 100 epochs with linear decay after epoch 70,
threshold 0.5. Their inference splits a 1024x768 image into 2x2 patches and
resizes each to 256 wide, i.e. an effective 512x384 image; training here uses
the same scale (images halved) with random 256x256 crops, and inference runs the
whole 512x384 image through the fully convolutional net -- the same pixels per
vessel as their patch scheme.

Data: the shared base (seg_base: ANFC-coco train restored to 1024x768 + our
vascular-dataset crops packed at native scale). Metrics are theirs
(evaluation.py): per-image SE, PC, F1 averaged, on ANFC valid and on the
vascular-dataset validation canvases separately.

  python scripts/anfc_unet_reproduce.py train --out E:/nailfold_tmp/unet
  python scripts/anfc_unet_reproduce.py infer --out E:/nailfold_tmp/unet --cases dev
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "ANFC"))
from Image_Segmentation.image_segmentation.model import U_Net  # noqa: E402

BASE = "E:/nailfold_tmp/seg_base"
SEED = 20260927


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), 1)


def load_pair(img_path):
    im = imread(img_path)
    H, W = im.shape[:2]
    m = np.zeros((H, W), np.uint8)
    lab = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    for line in open(lab).read().split("\n"):
        v = line.split()
        if len(v) >= 7:
            p = (np.asarray(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.fillPoly(m, [p], 1)
    im = cv2.resize(im, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    return im[..., 1], m          # BGR -> green channel, as in their loader


def to_tensor(g):
    return (torch.from_numpy(g).float()[None] / 255.0 - 0.5) / 0.5


def metrics(pred, gt):
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    se = tp / (tp + fn + 1e-6); pc = tp / (tp + fp + 1e-6)
    return se, pc, 2 * se * pc / (se + pc + 1e-6)


@torch.no_grad()
def evaluate(net, pairs, dev):
    net.eval()
    out = []
    for g, m in pairs:
        h, w = g.shape
        H, W = h - h % 16, w - w % 16
        x = to_tensor(g[:H, :W])[None].to(dev)
        p = torch.sigmoid(net(x))[0, 0].cpu().numpy() > 0.5
        out.append(metrics(p, m[:H, :W].astype(bool)))
    net.train()
    return np.mean(out, 0).round(4).tolist()


def train(a):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev = torch.device("cuda")
    tr = [load_pair(p) for p in sorted(glob.glob(BASE + "/train/images/*.jpg"))]
    va_anfc = [load_pair(p) for p in sorted(glob.glob(BASE + "/val/images/anfc_*.jpg"))]
    va_vd = [load_pair(p) for p in sorted(glob.glob(BASE + "/val/images/vd_*.jpg"))]
    net = U_Net(img_ch=1, output_ch=1).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-4, betas=(0.5, 0.999))
    crit = torch.nn.BCELoss()
    E, DECAY, B, CROP = 100, 70, 8, 256
    steps = max(1, len(tr) // B)
    log = []
    for ep in range(E):
        if ep >= DECAY:
            for g in opt.param_groups:
                g["lr"] = 2e-4 * (E - ep) / (E - DECAY)
        tot = 0.0
        for _ in range(steps):
            xs, ys = [], []
            for k in rng.integers(0, len(tr), B):
                g, m = tr[k]
                y0 = rng.integers(0, g.shape[0] - CROP + 1); x0 = rng.integers(0, g.shape[1] - CROP + 1)
                gc, mc = g[y0:y0 + CROP, x0:x0 + CROP], m[y0:y0 + CROP, x0:x0 + CROP]
                if rng.random() < 0.5:
                    gc, mc = gc[:, ::-1], mc[:, ::-1]
                if rng.random() < 0.7:     # their brightness / contrast jitter
                    gc = np.clip(gc * rng.uniform(0.95, 1.05) + rng.uniform(-25, 25), 0, 255)
                xs.append(to_tensor(np.ascontiguousarray(gc).astype(np.uint8)))
                ys.append(torch.from_numpy(np.ascontiguousarray(mc)).float()[None])
            x, y = torch.stack(xs).to(dev), torch.stack(ys).to(dev)
            loss = crit(torch.sigmoid(net(x)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        row = dict(epoch=ep, loss=round(tot / steps, 4))
        if ep % 10 == 9 or ep == E - 1:
            row["anfc_val_SE_PC_F1"] = evaluate(net, va_anfc, dev)
            row["vd_val_SE_PC_F1"] = evaluate(net, va_vd, dev)
        log.append(row); print(row, flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(a.out, "unet.pt"))
    json.dump(dict(log=log, final=log[-1],
                   paper=dict(SE=0.653, F1=0.487),
                   note="ANFC valid shares all 42 subjects with train"),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=2)


@torch.no_grad()
def infer(a):
    """U-Net masks on the 233 cases -> connected components -> the same
    per-image geometry as the YOLO arms (extract_instance_geometry functions)."""
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from extract_instance_geometry import image_row, pairwise_topology, polygon_features
    from mendeley_seg_geometry import DEV_IX, LOCK_IX, OUT, sha256
    dev = torch.device("cuda")
    w = os.path.join(a.out, "unet.pt")
    net = U_Net(img_ch=1, output_ch=1).to(dev)
    net.load_state_dict(torch.load(w, map_location=dev)); net.eval()
    ix = pd.concat([pd.read_csv(DEV_IX, dtype=str), pd.read_csv(LOCK_IX, dtype=str)],
                   ignore_index=True)
    rows = []
    for n, r in enumerate(ix.itertuples(), 1):
        im = imread(os.path.join(ROOT, "data", r.image_path))
        H0, W0 = im.shape[:2]
        g = cv2.resize(im, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA)[..., 1]
        h, wd = g.shape
        pad = np.pad(g, ((0, (-h) % 16), (0, (-wd) % 16)), mode="reflect")
        pr = torch.sigmoid(net(to_tensor(pad)[None].to(dev)))[0, 0].cpu().numpy()[:h, :wd]
        m = (pr > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inst = []
        for c in cs:
            if cv2.contourArea(c) * 4 < a.min_area:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) * 2   # back to full scale
            f = polygon_features(c)
            if f is not None:
                inst.append(f)
        row = image_row(inst, pairwise_topology(inst, W0, H0), W0, H0)
        row.update(exam_case_id=r.exam_case_id, image_path=r.image_path,
                   fg_frac=float(m.mean()))
        rows.append(row)
        if n % 300 == 0:
            print(n, len(ix), flush=True)
    out = os.path.join(OUT, "UNET")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "image_geometry.csv"), index=False)
    json.dump(dict(arm="UNET", weights=w, sha256=sha256(w), threshold=0.5,
                   min_area_px_fullscale=a.min_area, n_images=len(rows)),
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_area", type=float, default=141.0)  # p2 of human polygon area
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)


if __name__ == "__main__":
    main()
