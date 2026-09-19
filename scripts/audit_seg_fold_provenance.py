#!/usr/bin/env python
"""Read the training provenance recorded inside each seg_vessel_fold{k}/best.pt.

Why this matters right now
--------------------------
The one-shot locked-47 evaluation needs locked geometry, which means running
these five segmenter checkpoints on locked images. That is only legitimate if
locked-47 was held out of every one of the five training sets. The fold dataset
directories are gone from both servers, so the only surviving record of what
each checkpoint trained on is the `train_args` dict Ultralytics embeds in the
checkpoint. This script prints it and nothing else -- it makes no claim about
contamination, it just surfaces what is recoverable.
"""
import json

import torch

rows = []
for k in range(5):
    p = f"artifacts/models/seg_vessel_fold{k}/weights/best.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    ta = d.get("train_args") or {}
    m = d.get("model")
    try:
        params = sum(x.numel() for x in m.parameters()) / 1e6
    except Exception:
        params = None
    rows.append({
        "fold": k,
        "params_M": round(params, 2) if params else None,
        "nc": getattr(m, "nc", None),
        "names": getattr(m, "names", None),
        "date": d.get("date"),
        "version": d.get("version"),
        "epoch": d.get("epoch"),
        "best_fitness": (float(d["best_fitness"])
                         if d.get("best_fitness") is not None else None),
        "train_args": {kk: ta.get(kk) for kk in
                       ("model", "data", "epochs", "imgsz", "batch", "device",
                        "project", "name", "seed", "fraction", "val", "split")},
    })

print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
