"""Separate the capacity effect from the geometry change in round one.

The anchor runs the deployed geometry (direct resize to 518x686) while every new
arm runs the new loader (aspect-preserving resize, neutral pad to square). So
"dinov2l_capacity minus anchor" contains both a capacity change and a geometry
change and cannot attribute either.

This re-pairs the same stored out-of-fold predictions three ways:

  geometry_only   dinov2b_newloader vs anchor            same weights, new loader
  capacity_only   dinov2l_capacity  vs dinov2b_newloader same loader, more capacity
  medical_only    biomedclip_medical vs dinov2b_newloader same loader, medical weights
                  (still confounded with 224x224 input and patch 16)

No model is refit. Paired bootstrap is case-level with the protocol's seed.

  PYTHONIOENCODING=utf-8 python scripts/isolate_capacity_vs_geometry.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
SEED, N_BOOT = 20260921, 2000
MULTICLASS_FIELDS = {"papilla"}
CONTRASTS = [
    ("geometry_only", "dinov2b_newloader", "anchor_dinov2b_deployed",
     "same weights, deployed geometry replaced by the new loader"),
    ("capacity_only", "dinov2l_capacity", "dinov2b_newloader",
     "same loader and pretraining corpus, ViT-B/14 -> ViT-L/14"),
    ("medical_only", "biomedclip_medical", "dinov2b_newloader",
     "same loader; medical PMC-15M weights, but also 224x224 and patch 16"),
]


def harden(d: pd.DataFrame, field: str) -> pd.DataFrame:
    d = d.copy()
    if field not in MULTICLASS_FIELDS:
        d["pred"] = (d["pred"] >= 0.5).astype(float).where(d["pred"].notna())
    return d


def pair(oof: pd.DataFrame, field: str, arm: str, ref: str, rng) -> dict:
    d = harden(oof[oof.field == field], field)
    a = d[d.arm == ref].set_index("exam_case_id")
    b = d[d.arm == arm].set_index("exam_case_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    keep = a.pred.notna() & b.pred.notna() & a.y_true.notna()
    a, b = a[keep], b[keep]
    y, pa, pb = a.y_true.to_numpy(), a.pred.to_numpy(), b.pred.to_numpy()
    if len(np.unique(y)) < 2:
        return dict(n=int(len(y)), unevaluable="single class")
    obs = (balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa))
    boot = np.empty(N_BOOT)
    n = len(y)
    for i in range(N_BOOT):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            boot[i] = np.nan
            continue
        boot[i] = (balanced_accuracy_score(y[s], pb[s])
                   - balanced_accuracy_score(y[s], pa[s]))
    boot = boot[~np.isnan(boot)]
    folds = {}
    for f in sorted(a.fold.dropna().unique()):
        m = (a.fold == f).to_numpy()
        if len(np.unique(y[m])) < 2:
            folds[str(int(f))] = None
            continue
        folds[str(int(f))] = round(
            balanced_accuracy_score(y[m], pb[m])
            - balanced_accuracy_score(y[m], pa[m]), 4)
    same = sum(1 for v in folds.values() if v is not None and v > 0)
    return dict(n=int(n), ba_ref=round(balanced_accuracy_score(y, pa), 4),
                ba_arm=round(balanced_accuracy_score(y, pb), 4),
                ba_gain=round(float(obs), 4),
                ci=[round(float(np.percentile(boot, 2.5)), 4),
                    round(float(np.percentile(boot, 97.5)), 4)],
                ci_excludes_zero=bool(np.percentile(boot, 2.5) > 0
                                      or np.percentile(boot, 97.5) < 0),
                per_fold_ba_gain=folds,
                folds_positive="%d/%d" % (same, len(folds)),
                cases_arm_right_ref_wrong=int(((pb == y) & (pa != y)).sum()),
                cases_ref_right_arm_wrong=int(((pa == y) & (pb != y)).sum()))


def main() -> None:
    oof = pd.read_csv(EXP / "predictions_oof.csv", dtype={"exam_case_id": str})
    rng = np.random.default_rng(SEED)
    fields = list(dict.fromkeys(oof.field))
    out = {}
    for name, arm, ref, why in CONTRASTS:
        out[name] = dict(arm=arm, reference=ref, isolates=why,
                         fields={f: pair(oof, f, arm, ref, rng) for f in fields})
    res = dict(
        purpose="attribute the round-one differences to geometry, capacity or "
                "medical pretraining instead of reading every arm against the "
                "deployed anchor",
        seed=SEED, n_boot=N_BOOT, contrasts=out,
        caveats=[
            "development set only; locked-47 was not read",
            "medical_only remains confounded: BiomedCLIP is 224x224 patch 16 "
            "against 518x518 patch 14, so a null there does not prove medical "
            "pretraining is useless, only that this system is not better",
            "no model was refit; C was already chosen inside training folds",
        ])
    (EXP / "attribution.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in out:
        print("== %s (%s vs %s)" % (name, out[name]["arm"], out[name]["reference"]))
        for f, v in out[name]["fields"].items():
            if "unevaluable" in v:
                print("  %-28s unevaluable" % f)
                continue
            print("  %-28s gain %+.4f ci [%+.4f,%+.4f] folds %s" % (
                f, v["ba_gain"], v["ci"][0], v["ci"][1], v["folds_positive"]))
    print("\nwrote %s" % (EXP / "attribution.json"))


if __name__ == "__main__":
    main()
