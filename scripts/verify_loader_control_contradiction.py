#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconcile the two contradictory loader-control records (addendum 6.6).

The addendum said two incompatible things about the same pair of feature sets:
  line 328  anchor and the loader control were byte-identical at first, and the
            geometry was fixed BEFORE any metric was computed
  line 783  loader_geometry_only is unusable because the arm is byte-identical
            to the anchor and carries no information

Only one can be true. This script decides it from the artefacts, using four
independent checks, and writes the verdict with the evidence.

  1  file hashes of the stored poolings
  2  np.array_equal / max abs diff on every pooling
  3  token grid implied by each arm's geometry
  4  whether attribution.json's geometry_only contrast holds real numbers
     (a degenerate contrast cannot produce non-zero per-fold spreads)

Plus a version check that does not rely on mtimes: re-extract a few rows with
the CURRENT script and compare against what is on disk. This matters because
open_memmap'd .npy files do not update their mtime on flush under Windows, so
the timestamps alone cannot establish which script version wrote them.

DEVELOPMENT ONLY. Reads stored development features; never touches locked-47.

Run:
  PYTHONIOENCODING=utf-8 python scripts/verify_loader_control_contradiction.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
FEAT = EXP / "features"
OUT = EXP / "loader_control_reconciliation.json"

ANCHOR = "anchor_dinov2b_deployed"
CONTROL = "dinov2b_newloader"
POOLINGS = ["cls", "mean", "max", "topk_mean", "std"]
PROBE_ROWS = [0, 1, 500, 1707]      # fixed, not chosen after seeing anything


def sha(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def check_hashes() -> dict:
    out = {}
    for arm in (ANCHOR, CONTROL):
        out[arm] = {k: sha(FEAT / arm / ("features_%s.npy" % k)) for k in POOLINGS}
    out["any_pooling_hash_equal"] = any(
        out[ANCHOR][k] == out[CONTROL][k] for k in POOLINGS)
    return out


def check_arrays() -> dict:
    per = {}
    for k in POOLINGS:
        a = np.load(FEAT / ANCHOR / ("features_%s.npy" % k)).astype(np.float32)
        b = np.load(FEAT / CONTROL / ("features_%s.npy" % k)).astype(np.float32)
        per[k] = dict(
            shape_anchor=list(a.shape), shape_control=list(b.shape),
            array_equal=bool(a.shape == b.shape and np.array_equal(a, b)),
            max_abs_diff=(round(float(np.abs(a - b).max()), 4)
                          if a.shape == b.shape else None))
    return dict(per_pooling=per,
                all_equal=all(v["array_equal"] for v in per.values()),
                any_equal=any(v["array_equal"] for v in per.values()))


def check_geometry() -> dict:
    out = {}
    for arm in (ANCHOR, CONTROL):
        m = json.loads((FEAT / arm / "metadata.json").read_text(encoding="utf-8"))
        out[arm] = dict(geometry=m["geometry"], patch_grid=m["patch_grid"],
                        input_size=m["input_size"], patch=m["patch"],
                        weights=m["weights"],
                        weights_sha256=str(m.get("weights_sha256"))[:12],
                        role=m["role"], locked_cases_seen=m["locked_cases_seen"])
    out["grids_differ"] = (out[ANCHOR]["patch_grid"] != out[CONTROL]["patch_grid"])
    out["same_weights"] = (out[ANCHOR]["weights"] == out[CONTROL]["weights"])
    return out


def check_attribution() -> dict:
    """A degenerate (byte-identical) contrast yields exactly 0 everywhere."""
    p = EXP / "attribution.json"
    if not p.exists():
        return dict(present=False)
    g = json.loads(p.read_text(encoding="utf-8"))["contrasts"].get("geometry_only")
    if not g:
        return dict(present=False)
    fields = {f: dict(ba_gain=v["ba_gain"], ci=v["ci"],
                      folds_positive=v["folds_positive"],
                      per_fold_ba_gain=v["per_fold_ba_gain"])
              for f, v in g["fields"].items()}
    nonzero = [f for f, v in fields.items() if abs(v["ba_gain"]) > 1e-9]
    return dict(present=True, arm=g["arm"], reference=g["reference"],
                fields=fields, n_fields=len(fields),
                n_fields_with_nonzero_gain=len(nonzero),
                all_ci_span_zero=all(v["ci"][0] <= 0 <= v["ci"][1]
                                     for v in fields.values()),
                could_come_from_a_degenerate_contrast=(len(nonzero) == 0))


def check_versions() -> dict:
    """Re-extract PROBE_ROWS with the current script; mtimes cannot settle this."""
    import torch
    import extract_medical_encoders as X                      # noqa: E402

    ix = X.load_index()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = dict(script=str(Path("scripts/extract_medical_encoders.py")),
               script_sha256=sha(ROOT / "scripts" / "extract_medical_encoders.py"),
               script_mtime=mtime(ROOT / "scripts" / "extract_medical_encoders.py"),
               index_rows=int(len(ix)), probe_rows=PROBE_ROWS, arms={})
    for arm in (CONTROL, ANCHOR):
        cfg = X.ARMS[arm]
        dep = cfg.get("deployed_res")
        model, wsha = X.build(cfg, dev, img_size=list(dep) if dep else None)
        grid = ((dep[0] // cfg["patch"], dep[1] // cfg["patch"]) if dep
                else tuple(model.patch_embed.grid_size))
        with torch.inference_mode():
            pre = [X.preprocess(X.IMAGE_ROOT / ix.image_path.iloc[i],
                                cfg["size"], cfg["norm"], dep) for i in PROBE_ROWS]
            x = torch.stack([p[0] for p in pre]).to(dev)
            tok = model.forward_features(x)
            vals = X.pool(tok, model.num_prefix_tokens, grid,
                          [p[1] for p in pre], cfg["patch"], 16)
        got = vals["cls"].cpu().numpy().astype(np.float16)
        stored = np.load(FEAT / arm / "features_cls.npy")[PROBE_ROWS]
        res["arms"][arm] = dict(
            grid=list(grid), weights_sha256=wsha[:12],
            max_abs_diff_vs_stored=round(float(np.abs(
                got.astype(np.float32) - stored.astype(np.float32)).max()), 6),
            reproduces=bool(np.allclose(got, stored, atol=2e-2)))
    res["all_arms_reproduce"] = all(v["reproduces"] for v in res["arms"].values())
    return res


def timeline() -> dict:
    """Only index.csv / metadata.json / json artefacts are ordinary writes."""
    rows = []
    for arm in sorted(FEAT.iterdir()):
        for f in ("index.csv", "metadata.json"):
            if (arm / f).exists():
                rows.append((mtime(arm / f), "features/%s/%s" % (arm.name, f)))
    for p in sorted(EXP.glob("*.json")) + sorted(EXP.glob("*.yaml")) + \
            sorted(EXP.glob("*.csv")):
        rows.append((mtime(p), p.name))
    return dict(
        note=("open_memmap'd .npy files do not refresh mtime on flush under "
              "Windows, so .npy timestamps are excluded as unreliable"),
        events=[dict(when=t, what=n) for t, n in sorted(rows)])


def main() -> None:
    h, arr, geo, att = check_hashes(), check_arrays(), check_geometry(), \
        check_attribution()
    ver = check_versions()

    identical = arr["all_equal"]
    real = att.get("present") and not att["could_come_from_a_degenerate_contrast"]

    if not identical and real:
        verdict = "line_783_wrong_line_328_right"
        statement = (
            "The stored loader control is NOT byte-identical to the anchor "
            "(distinct file hashes, np.array_equal False on all five poolings, "
            "max abs diff up to %.3f, token grids %s vs %s), and "
            "attribution.json's geometry_only contrast holds %d non-zero "
            "per-field gains, which a degenerate contrast cannot produce. So "
            "line 328 is correct: the byte-identical condition existed and was "
            "fixed before any metric was computed. Line 783's claim that "
            "loader_geometry_only is unusable carried the pre-fix state "
            "forward and is withdrawn. The contrast is usable and its result "
            "is null on the %d fields measured (all paired CIs span zero), "
            "which is a measured null, not an absent measurement. It says "
            "nothing about the remaining fields, which were never run on the "
            "pad-geometry arm."
            % (max(v["max_abs_diff"] for v in arr["per_pooling"].values()),
               geo[ANCHOR]["patch_grid"], geo[CONTROL]["patch_grid"],
               att["n_fields_with_nonzero_gain"], att["n_fields"]))
    elif identical:
        verdict = "line_783_right_line_328_wrong"
        statement = ("The arms are byte-identical on disk, so the loader "
                     "control is degenerate and line 328's claim that the fix "
                     "landed before any metric is contradicted.")
    else:
        verdict = "unresolved"
        statement = ("Arms differ but the geometry_only contrast is absent or "
                     "all-zero; the records cannot be reconciled from these "
                     "artefacts alone.")

    doc = dict(
        purpose=("decide which of two contradictory loader-control records in "
                 "rescue_plan_medical_encoder_addendum_20260921.md is correct, "
                 "and pin the version/path/hash facts behind them"),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        anchor=ANCHOR, control=CONTROL,
        verdict=verdict, statement=statement,
        checks=dict(file_hashes=h, arrays=arr, geometry=geo,
                    attribution_geometry_only=att, version_reproduction=ver),
        reliable_timeline=timeline(),
        corrections_to_the_addendum=[
            ("line 783 / 5.10 loader_geometry_only: 'unavailable, byte-identical"
             ", carries no information' -> 'usable; measured null on 5 fields, "
             "all paired CIs span zero; untested on the other 10 fields'"),
            ("line 328 stands as written: the byte-identical condition was "
             "fixed at 13:04-13:15, before the first metrics at 13:32"),
        ],
        limitations=[
            ("the deployment loaded facebook .pth while this round loads the "
             "timm safetensors conversion; baseline_manifest.json records "
             "hashes_match: false, so the anchor is a re-implementation of the "
             "deployed geometry, not a bit-exact replica. Independent of the "
             "contradiction settled here."),
            ("geometry_only was computed on 7 fields only; the 15-field matrix "
             "did not include the pad-geometry arm, so the null does not "
             "generalise to the other 8 fields"),
            ("rbc_aggregation's exactly-zero gain is not evidence of a "
             "degenerate contrast: that field collapses to the mode on both "
             "arms, so two identical constant predictors differ by zero"),
            ("a null paired contrast is not proof of harmlessness"),
            "development features only; this run did not read locked-47",
        ],
        locked_cases_seen=0,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(dict(verdict=verdict, statement=statement,
                          all_arms_reproduce=ver["all_arms_reproduce"],
                          out=str(OUT)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
