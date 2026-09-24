"""Per-class held-out metrics for the EXTERNAL morphology-pretraining detector.

train_one() wrapped its per-class capture in try/except so that a metrics-shape
change could never lose a trained fold. That guard fired here: the external run's
per_class.json holds only

  ValueError('The truth value of an array with more than one element is ambiguous')

because `getattr(r, "names", None) or {}` evaluates a numpy-backed names object
in a boolean context. The local folds survived it, the external run did not, so
the four external class APs were never written.

Why they are needed rather than nice to have: the external boxes span a whole
hairpin and are 7.12x our box height fraction, and their class definitions
(bushy / crossing / hairpin / tortuous) are NOT our report fields. If this
detector cannot even find its own classes, then a null on the A2x/A3 rungs
cannot be attributed to "vessel-box morphology pretraining does not help our
fields" -- the more likely reading would be that the pretraining target was
never learned. Those are different findings and must not be reported as one.

The external tree is still on disk (it is reused across the external and
external_then_local sources), so this only re-validates the saved best.pt
against the already-frozen person-grouped val side. Nothing is trained, no local
image is read, and no local fold is touched.

  python scripts/recover_external_pretrain_per_class.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_morph_detector_folds import (  # noqa: E402
    DET, EXT_NAMES, _shim_numpy_trapz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="external_pretrain")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    run = DET / a.run
    best = run / "weights" / "best.pt"
    data = DET / "_tree_external" / "dataset.yaml"
    for p in (best, data):
        if not p.exists():
            raise RuntimeError("missing %s" % p)

    # The split this validates against must be the person-grouped one, not the
    # superseded subject_id split that put 5 people on both sides. Assert it from
    # the tree's own manifest rather than trusting the directory name.
    tree = json.loads((DET / "external_tree.json").read_text(encoding="utf-8"))
    if tree.get("people_on_both_sides") != 0:
        raise RuntimeError("external tree is not person-grouped: %s" % tree)

    _shim_numpy_trapz()
    from ultralytics import YOLO

    r = YOLO(str(best)).val(data=str(data), imgsz=a.imgsz, batch=8, workers=2,
                            device=a.device, verbose=False, plots=False)
    per = {str(EXT_NAMES.get(i, i)): round(float(ap_), 4)
           for i, ap_ in enumerate(list(r.box.ap50))}
    rec = dict(
        map50=round(float(r.box.map50), 4),
        map50_95=round(float(r.box.map), 4),
        ap50_per_class=per,
        val_images=tree["val_images"], val_people=tree["val_people"],
        grouping=tree["grouping"],
        classes_are_not_our_fields=(
            "bushy/crossing/hairpin/tortuous are the external authors' "
            "definitions; these APs say whether the pretraining target was "
            "learned, NOT whether any of our report fields can be predicted"),
        not_a_generalisation_claim=(
            "this split exists to drive the detector's early stopping; its mAP "
            "is not quoted as external generalisation"),
        recovered_because=(
            "train_one()'s per-class capture raised on the numpy-backed names "
            "object and its try/except wrote only the error"),
    )
    (run / "per_class.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
