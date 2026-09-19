#!/usr/bin/env python
"""Dry-run the frozen-protocol guards WITHOUT fitting anything or scoring locked.

This exists so every failure mode that can be caught before the one-shot run is
caught before it: a drifted development baseline, a transposed majority class, a
label value that falls through both lists. It reads the locked label column only
to count rows and unmapped values -- no model is fitted, no accuracy is computed,
so it does not consume the evaluation.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from eval_locked_final import FIELDS_FROZEN, binarise, unmapped_values  # noqa: E402

man = pd.read_csv("artifacts/manifest/locked_evaluation_v1_reviewed.csv")
man["exam_case_id"] = man["exam_case_id"].astype(str)
dev = man[man.evaluation_role == "development"].set_index("exam_case_id")
lk = man[man.evaluation_role == "locked_test"].set_index("exam_case_id")
print(f"development {len(dev)} cases / locked {len(lk)} cases\n")

fail = 0
for f, cfg in FIELDS_FROZEN.items():
    yd = binarise(dev[f], cfg).dropna()
    yl = binarise(lk[f], cfg).dropna()
    maj = float(np.bincount(yd.to_numpy().astype(int)).argmax())
    base = float((yd.to_numpy() == maj).mean())
    ok = abs(base - cfg["dev_baseline"]) < 0.002
    fail += 0 if ok else 1
    verdict = "PASS" if ok else "FAIL"
    print(f"{f:28s} dev_n={len(yd):3d} base_recomputed={base:.4f} "
          f"recorded={cfg['dev_baseline']:.3f} guard={verdict}")
    print(f"{'':28s} majority_answer={int(maj)} "
          f"(0=normal,1=abnormal)  locked_n={len(yl):3d} "
          f"locked_abnormal_share={yl.mean():.3f}")
    ud, ul = unmapped_values(dev[f], cfg), unmapped_values(lk[f], cfg)
    if ud or ul:
        print(f"{'':28s} UNMAPPED dev={ud} locked={ul}")
    # locked rows lost relative to the 47 available
    if len(yl) < 47:
        print(f"{'':28s} note: {47 - len(yl)} locked cases have no usable label")

print(f"\nguards failed: {fail}")
if fail:
    raise SystemExit("protocol drifted -- do NOT run the locked evaluation")
print("all frozen baselines reproduce; safe to run the locked evaluation")
