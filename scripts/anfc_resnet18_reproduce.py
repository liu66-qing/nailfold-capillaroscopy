#!/usr/bin/env python
"""Reproduce the ANFC abnormal/normal ResNet18 stage (Object_Detection/nailfold_classifier).

Faithful to their classifier.yaml / train_classifier_model.py: ImageNet-pretrained
resnet18, 224x224, ImageNet normalisation, train_size normal 80 / abnormal 50,
SGD lr 0.01 on the new head and 0.001 on pretrained layers, weight decay 5e-4,
StepLR(5, 0.9), 30 epochs, batch 4, colour jitter 0.1, left-right flip 0.3,
invert 0.3. Crops are the instance bboxes from ANFC-coco train (the Roboflow
640x640 stretch undone to 1024x768 first).

Caveats stated up front: ANFC-coco has only 57 "abnormal" train instances and 9
in valid, and valid shares all 42 subjects with train, so valid numbers are
optimistic.

infer: S0 instances (our segmenter, conf 0.25) on the 233 cases -> bbox crops ->
P(abnormal). Per image: abn_share (argmax abnormal), abn_prob_mean, abn_prob_p90,
n_inst. Written as E:/nailfold_tmp/seg_geom/RESNET18/image_geometry.csv so
mendeley_seg_eval.py treats it as one more sixth-route arm.

  python scripts/anfc_resnet18_reproduce.py train --out E:/nailfold_tmp/resnet18
  python scripts/anfc_resnet18_reproduce.py infer --out E:/nailfold_tmp/resnet18 \
      --seg E:/nailfold_tmp/seg_runs/S0/weights/best.pt
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANFC = os.path.join(ROOT, "data/ANFC-THU.v1i.coco")
SEED = 20260927
CAT = {1: "abnormal", 5: "normal"}      # ANFC-coco category ids
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), 1)


def crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(np.ceil(x + w))), min(H, int(np.ceil(y + h)))
    return img[y0:y1, x0:x1] if x1 - x0 >= 4 and y1 - y0 >= 4 else None


def anfc_crops(split):
    d = json.load(open(os.path.join(ANFC, split, "_annotations.coco.json")))
    ims = {i["id"]: i for i in d["images"]}
    cache, out = {}, []
    for a in d["annotations"]:
        if a["category_id"] not in CAT:
            continue
        im = ims[a["image_id"]]
        if im["id"] not in cache:
            cache[im["id"]] = cv2.resize(imread(os.path.join(ANFC, split, im["file_name"])),
                                         (1024, 768), interpolation=cv2.INTER_CUBIC)
        sx, sy = 1024 / im["width"], 768 / im["height"]
        x, y, w, h = a["bbox"]
        c = crop(cache[im["id"]], x * sx, y * sy, w * sx, h * sy)
        if c is not None:
            out.append((c, 0 if CAT[a["category_id"]] == "abnormal" else 1))   # their order
    return out


def to_t(bgr, train, rng):
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if train:
        if rng.random() < 0.3:
            rgb = 1 - rgb
        if rng.random() < 0.3:
            rgb = rgb[:, ::-1]
        rgb = np.clip(rgb * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1), 0, 1)  # brightness/contrast 0.1
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    return torchvision.transforms.functional.normalize(t, MEAN, STD)


def build():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def predict(m, crops, dev, bs=64):
    m.eval(); out = []
    for i in range(0, len(crops), bs):
        x = torch.stack([to_t(c, False, None) for c in crops[i:i + bs]]).to(dev)
        out.append(torch.softmax(m(x), 1)[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(a):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev = torch.device(a.device)
    tr_all, va = anfc_crops("train"), anfc_crops("valid")
    ab = [c for c in tr_all if c[1] == 0]; no = [c for c in tr_all if c[1] == 1]
    pick = lambda L, n: [L[i] for i in rng.choice(len(L), min(n, len(L)), replace=False)]
    tr = pick(ab, 50) + pick(no, 80)
    m = build().to(dev)
    head = list(m.fc.parameters()); hid = {id(p) for p in head}
    opt = torch.optim.SGD([{"params": head, "lr": 0.01},
                           {"params": [p for p in m.parameters() if id(p) not in hid], "lr": 0.001}],
                          lr=0.01, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5, 0.9)
    ce = nn.CrossEntropyLoss(); log = []
    for ep in range(30):
        m.train(); order = rng.permutation(len(tr)); tot = 0.0
        for i in range(0, len(order), 4):
            b = [tr[k] for k in order[i:i + 4]]
            x = torch.stack([to_t(c, True, rng) for c, _ in b]).to(dev)
            y = torch.tensor([l for _, l in b]).to(dev)
            loss = ce(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        log.append(dict(epoch=ep, loss=round(tot / max(1, len(order) // 4), 4)))
    p = predict(m, [c for c, _ in va], dev); y = np.array([l for _, l in va])
    hat = (p > 0.5)
    ev = dict(n_valid=len(y), n_abnormal=int((y == 0).sum()),
              abnormal_recall=round(float(hat[y == 0].mean()), 4),
              normal_specificity=round(float((~hat[y == 1]).mean()), 4),
              predicted_abnormal_share=round(float(hat.mean()), 4))
    from sklearn.metrics import roc_auc_score
    ev["auc_abnormal"] = round(float(roc_auc_score(y == 0, p)), 4)
    os.makedirs(a.out, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(a.out, "resnet18.pt"))
    json.dump(dict(train_n=dict(abnormal=len(pick(ab, 50)), normal=80), valid=ev, log=log,
                   note="valid shares all 42 subjects with train; 9 abnormal instances only"),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=2)
    print(json.dumps(ev, indent=2))


def infer(a):
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from mendeley_seg_geometry import DEV_IX, LOCK_IX, OUT, sha256
    from ultralytics import YOLO
    dev = torch.device("cuda")
    m = build().to(dev)
    w = os.path.join(a.out, "resnet18.pt")
    m.load_state_dict(torch.load(w, map_location=dev))
    seg = YOLO(a.seg)
    ix = pd.concat([pd.read_csv(DEV_IX, dtype=str), pd.read_csv(LOCK_IX, dtype=str)],
                   ignore_index=True)
    rows = []
    for n, r in enumerate(ix.itertuples(), 1):
        im = imread(os.path.join(ROOT, "data", r.image_path))
        res = seg.predict(im, conf=a.conf, imgsz=1024, verbose=False, device=0)[0]
        cs = []
        for x0, y0, x1, y1 in res.boxes.xyxy.cpu().numpy():
            c = crop(im, x0, y0, x1 - x0, y1 - y0)
            if c is not None:
                cs.append(c)
        p = predict(m, cs, dev)
        rows.append(dict(exam_case_id=r.exam_case_id, image_path=r.image_path, n_inst=len(p),
                         abn_share=float((p > 0.5).mean()) if len(p) else np.nan,
                         abn_prob_mean=float(p.mean()) if len(p) else np.nan,
                         abn_prob_p90=float(np.percentile(p, 90)) if len(p) else np.nan))
        if n % 300 == 0:
            print(n, len(ix), flush=True)
    out = os.path.join(OUT, "RESNET18")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "image_geometry.csv"), index=False)
    json.dump(dict(arm="RESNET18", weights=w, sha256=sha256(w), seg=a.seg, seg_sha256=sha256(a.seg),
                   seg_conf=a.conf, n_images=len(rows)),
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default="E:/nailfold_tmp/seg_runs/S0/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)


if __name__ == "__main__":
    main()
