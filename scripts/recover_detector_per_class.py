"""Per-class held-out metrics for detectors that were trained before this was logged.

results.csv records only the "all" row, and each fold's YOLO tree is deleted right
after training to keep 3224 scratch files per fold out of the repository. So the
per-class numbers cannot be read off the finished runs -- but they are the numbers
that matter: an "all" mAP50 of 0.44 looks identical whether the malformed and cross
classes are being found or the vessel class is carrying the average by itself, and
the A2 hypothesis rests entirely on those two classes.

This rebuilds each fold's validation side from the frozen pools and re-validates the
saved best.pt against it. write_local_fold is deterministic given the same frozen
inputs, so the rebuilt held-out set is the same one the fold was validated on. Only
the val side is materialised -- nothing is trained here.

  python scripts/recover_detector_per_class.py --detectors detectors_local.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_morph_detector_folds import (  # noqa: E402
    DET, LOCAL_NAMES, _shim_numpy_trapz, local_assignment, write_local_fold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="detectors_local.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    _shim_numpy_trapz()
    from ultralytics import YOLO

    man = json.loads((DET / a.detectors).read_text(encoding="utf-8"))
    keep, locked_ids = local_assignment()
    out = {}
    for fold_s, info in sorted(man["folds"].items()):
        run = ROOT / info["weights"]
        if not run.exists():
            out[fold_s] = {"status": "weights_missing"}
            continue
        tgt = run.parent.parent / "per_class.json"
        if tgt.exists() and "error" not in json.loads(tgt.read_text(encoding="utf-8")):
            out[fold_s] = {"status": "already_recorded"}
            continue
        work = DET / ("_recover_tree_fold%s" % fold_s)
        shutil.rmtree(work, ignore_errors=True)
        built = write_local_fold(float(fold_s), keep, locked_ids, work)
        if built["val_images"] != info["val_images"]:
            raise RuntimeError("rebuilt val set differs for fold %s: %d vs %d"
                               % (fold_s, built["val_images"], info["val_images"]))
        r = YOLO(str(run)).val(data=str(work / "dataset.yaml"), imgsz=a.imgsz,
                               batch=8, workers=0, device=a.device, verbose=False,
                               plots=False)
        per = {str(LOCAL_NAMES.get(i, i)): round(float(ap_), 4)
               for i, ap_ in enumerate(list(r.box.ap50))}
        rec = dict(map50=round(float(r.box.map50), 4),
                   map50_95=round(float(r.box.map), 4), ap50_per_class=per,
                   val_images=built["val_images"], val_cases=built["val_cases"],
                   note=("recovered after training by rebuilding this fold's val side "
                         "from the frozen pools; image count verified against the "
                         "manifest written at training time"))
        tgt.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        out[fold_s] = rec
        shutil.rmtree(work, ignore_errors=True)
        print("fold %s: %s" % (fold_s, per), flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
