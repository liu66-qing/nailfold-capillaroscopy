#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the AI-annotator exam from Capillary-Dataset human boxes (public, apache-2.0).

preregistration.md §5: 200 human-boxed capillaries (50 per class: hairpin,
crossing, tortuous, bushy) + 20 background crops that contain no human box.
Each crop is saved under a random id so the file name carries no class; the
answer key is written to a separate file that is never shown to any model.

Scored questions (binary, matching our two morphology fields):
  Q_cross    : crossing vs not          -> crossing_ratio
  Q_abnormal : tortuous/bushy vs hairpin -> malformation_ratio
  Q_vessel   : capillary vs background   -> sanity check

Run:
  PYTHONIOENCODING=utf-8 python scripts/build_ai_exam.py
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/Capillary-Dataset/Morphology_detection"
OUT = ROOT / "artifacts/experiments/video_kymo_20260925/ai_exam"
NAMES = {0: "bushy", 1: "crossing", 2: "hairpin", 3: "tortuous"}
PER_CLASS = 50
N_BACKGROUND = 20
SEED = 20260925
MARGIN = 0.25      # box expanded by 25% on each side
MIN_SIDE = 64      # skip boxes smaller than this in pixels (too small to judge)
OUT_SIDE = 320


def boxes_of(txt: Path, w: int, h: int):
    for line in txt.read_text().split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        yield c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


def expand(b, w, h):
    x1, y1, x2, y2 = b
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)


def save(im, box, path):
    crop = im.crop(tuple(round(v) for v in box))
    s = OUT_SIDE / max(crop.size)
    crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS).save(path, quality=95)


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> None:
    rng = random.Random(SEED)
    pool = {c: [] for c in NAMES}
    images = sorted((SRC / "images/train").glob("*.jpg"))
    for img in images:
        txt = SRC / "labels/train" / (img.stem + ".txt")
        if not txt.exists():
            continue
        w, h = Image.open(img).size
        for c, *b in boxes_of(txt, w, h):
            if min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE * 0.5 and max(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE:
                pool[c].append((img, b))

    items = []
    for c, cands in pool.items():
        rng.shuffle(cands)
        # at most one box per source image per class, to spread over subjects
        seen = set()
        for img, b in cands:
            if img.name in seen:
                continue
            seen.add(img.name)
            items.append(dict(kind="vessel", cls=NAMES[c], img=img, box=b))
            if sum(i["cls"] == NAMES[c] for i in items) == PER_CLASS:
                break

    # background: a box-sized window that overlaps no human box at all
    bg = 0
    rng.shuffle(images)
    for img in images:
        if bg == N_BACKGROUND:
            break
        txt = SRC / "labels/train" / (img.stem + ".txt")
        if not txt.exists():
            continue
        w, h = Image.open(img).size
        humans = [expand(b, w, h) for _, *b in boxes_of(txt, w, h)]
        for _ in range(50):
            bw, bh = rng.uniform(70, 140), rng.uniform(120, 220)
            x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
            cand = (x1, y1, x1 + bw, y1 + bh)
            if not any(overlaps(cand, hb) for hb in humans):
                items.append(dict(kind="background", cls="background", img=img, box=cand))
                bg += 1
                break

    rng.shuffle(items)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    key = []
    for i, it in enumerate(items):
        qid = "q%03d" % i
        im = Image.open(it["img"]).convert("RGB")
        box = expand(it["box"], *im.size) if it["kind"] == "vessel" else it["box"]
        save(im, box, OUT / "crops" / (qid + ".jpg"))
        key.append(dict(qid=qid, cls=it["cls"], source_image=it["img"].name,
                        is_vessel=it["kind"] == "vessel",
                        is_crossing=it["cls"] == "crossing",
                        is_abnormal=it["cls"] in ("tortuous", "bushy") if it["cls"] != "crossing" and it["kind"] == "vessel" else None))
    (OUT.parent / "ai_exam_key" / "answer_key_DO_NOT_SHOW.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(key), Counter(k["cls"] for k in key), "source images", len({k["source_image"] for k in key}))


if __name__ == "__main__":
    sys.exit(main())
