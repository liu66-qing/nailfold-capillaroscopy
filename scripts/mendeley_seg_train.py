#!/usr/bin/env python
"""Train one segmenter arm. All arms share seed, hyper-parameters and val set.

  python scripts/mendeley_seg_train.py --name S0 --train E:/nailfold_tmp/seg_base/train
  python scripts/mendeley_seg_train.py --name S1 --train E:/nailfold_tmp/seg_base/train \
      --extra E:/nailfold_tmp/mendeley_pseudo/inpaint
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_W = os.path.join(ROOT, "weights/sam2/yolo11m-seg.pt")
VAL = "E:/nailfold_tmp/seg_base/val/images"
PROJECT = "E:/nailfold_tmp/seg_runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train = [os.path.join(a.train, "images")]
    if a.extra:
        # PREREG rev 4: one tile per Mendeley image (seeded), not all 4 corner tiles,
        # so an arm costs ~3x S0 instead of ~9x. Same tile choice for S1 and S1n.
        import glob
        import random
        by = {}
        for p in sorted(glob.glob(os.path.join(a.extra, "images", "*.jpg"))):
            by.setdefault(os.path.basename(p).rsplit("_", 2)[0], []).append(p)
        rnd = random.Random(20260927)
        pick = [rnd.choice(v) for _, v in sorted(by.items())]
        lst = os.path.join(PROJECT, a.name + "_extra.txt")
        os.makedirs(PROJECT, exist_ok=True)
        open(lst, "w").write("\n".join(pick) + "\n")
        train.append(lst)
    yml = os.path.join(PROJECT, a.name + ".yaml")
    os.makedirs(PROJECT, exist_ok=True)
    with open(yml, "w") as f:
        f.write("train:\n" + "".join("  - %s\n" % t for t in train))
        f.write("val: %s\nnames:\n  0: vessel\n" % VAL)
    from ultralytics import YOLO
    last = os.path.join(PROJECT, a.name, "weights", "last.pt")
    if a.resume and os.path.exists(last):
        # exact continuation (weights + optimizer + epoch); workers=0 because the
        # close_mosaic dataloader rebuild deadlocked with workers=2 on Windows
        YOLO(last).train(resume=True, workers=0)
        return
    YOLO(BASE_W).train(data=yml, imgsz=1024, epochs=a.epochs, batch=a.batch,
                       workers=2, seed=20260927, deterministic=True,
                       project=PROJECT, name=a.name, exist_ok=True,
                       degrees=10, fliplr=0.5, mosaic=1.0, close_mosaic=10,
                       plots=False, verbose=False)


if __name__ == "__main__":
    main()
