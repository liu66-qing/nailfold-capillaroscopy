#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much does each field move when only the camera changes, and does
training on other simulated devices protect against an unseen one?

Two arms, same head as the shipped anchor (5 poolings, StandardScaler -> PCA64
-> LogReg C=0.03, probability mean over poolings, case mean over images),
same development_fold OOF:

  S0  train on clean features, test fold under shift X
  S1  train on clean + every shift EXCEPT X (and except its target-changing
      twin family), test fold under shift X  -- leave-one-shift-out, so the
      test device is never seen in training

Reported per field x shift: BA, both class recalls, share of cases whose
predicted class flips versus the clean prediction, and prediction diversity
(largest predicted-class share). A field "survives" a shift when BA drop <=0.05
and neither recall falls below 0.5.

Shifts that change a field's own target (blur/lowres vs clarity; colour vs
blood_color) are flagged, because there a changed answer may be correct.

DEVELOPMENT ONLY.

  PYTHONIOENCODING=utf-8 python scripts/eval_device_shift.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_field_matrix_three_arms as R  # noqa: E402
from device_shift_features import CONDITIONS  # noqa: E402

FEAT = ROOT / "artifacts" / "experiments" / "device_shift_20260926" / "features"
OUT = FEAT.parent
FIELDS = ["clarity", "subpapillary_venous_plexus", "exudation", "blood_color",
          "microthrombus", "malformation_ratio"]
COLOUR = {"warm", "cool", "desat_0.5", "hf_like"}
SHARP = {"lowres_0.5", "blur_s1.5"}


def load(cond):
    d = FEAT / cond
    ix = pd.read_csv(d / "index.csv", dtype={"exam_case_id": str})
    return ix, {p: np.load(d / ("features_%s.npy" % p)).astype(np.float32)
                for p in R.POOLINGS}


def family(c):
    return "colour" if c in COLOUR else "sharp" if c in SHARP else c


def oof(train_sets, test_set, ix, y_case, folds, order):
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    fs = pd.Series([folds[c] for c in order], index=order)
    for fo in sorted(fs.unique()):
        te = set(fs.index[fs == fo])
        trm = dm & ~ix.exam_case_id.isin(te).to_numpy()
        tem = dm & ix.exam_case_id.isin(te).to_numpy()
        ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
        ids = ix.exam_case_id[tem].to_numpy()
        probs = []
        for p in R.POOLINGS:
            Xtr = np.concatenate([s[p][trm] for s in train_sets])
            yy = np.concatenate([ytr] * len(train_sets))
            probs.append(R.fit_predict(Xtr, yy, test_set[p][tem], False, [0.0, 1.0])[0])
        s = pd.Series(np.mean(probs, 0), index=ids).groupby(level=0).mean()
        for c, v in s.items():
            out[pos[c]] = v
    return out


def metrics(y, p, p_clean):
    yp = (p >= 0.5).astype(int)
    rec = recall_score(y, yp, labels=[0, 1], average=None, zero_division=0)
    return dict(ba=round(float(balanced_accuracy_score(y, yp)), 4),
                recall0=round(float(rec[0]), 3), recall1=round(float(rec[1]), 3),
                flip_vs_clean=round(float((yp != (p_clean >= 0.5)).mean()), 3),
                top_pred_share=round(float(max(yp.mean(), 1 - yp.mean())), 3))


def main():
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    assert len(dev) == 186
    folds = dev.development_fold.to_dict()
    conds = [c for c in CONDITIONS if (FEAT / c / "done.json").exists()]
    assert "clean" in conds
    data = {c: load(c) for c in conds}
    ix = data["clean"][0]
    for c in conds:
        assert (data[c][0].image_path.values == ix.image_path.values).all(), c
    shifts = [c for c in conds if c != "clean"]
    rows = []
    for f in FIELDS:
        y_case, _, _ = R.target(dev, f)
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy().astype(int)
        clean_s0 = oof([data["clean"][1]], data["clean"][1], ix, y_case, folds, order)
        base = metrics(y, clean_s0, clean_s0)
        allaug = oof([data[c][1] for c in conds], data["clean"][1], ix, y_case, folds, order)
        rows.append(dict(field=f, shift="clean", arm="S0", **base))
        rows.append(dict(field=f, shift="clean", arm="S1_all", **metrics(y, allaug, clean_s0)))
        for sh in shifts:
            ps0 = oof([data["clean"][1]], data[sh][1], ix, y_case, folds, order)
            tr = [data["clean"][1]] + [data[c][1] for c in shifts
                                       if c != sh and family(c) != family(sh)]
            ps1 = oof(tr, data[sh][1], ix, y_case, folds, order)
            tc = (f == "clarity" and sh in SHARP) or (f == "blood_color" and sh in COLOUR)
            for arm, p in (("S0", ps0), ("S1_loso", ps1)):
                m = metrics(y, p, clean_s0)
                m["ba_drop_vs_clean"] = round(base["ba"] - m["ba"], 4)
                m["survives"] = bool(m["ba_drop_vs_clean"] <= 0.05 and
                                     min(m["recall0"], m["recall1"]) >= 0.5)
                rows.append(dict(field=f, shift=sh, arm=arm, target_changing=tc, **m))
            print(f, sh, rows[-2]["ba"], rows[-1]["ba"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "device_shift_results.csv", index=False)
    summ = {}
    for arm in ("S0", "S1_loso"):
        d = df[(df.arm == arm) & (df["shift"] != "clean") & (~df.target_changing.fillna(False))]
        summ[arm] = d.groupby("field").agg(survive=("survives", "mean"),
                                           mean_drop=("ba_drop_vs_clean", "mean"),
                                           worst_drop=("ba_drop_vs_clean", "max")).round(3).to_dict("index")
    summ["locked_cases_seen"] = 0
    summ["shifts"] = shifts
    (OUT / "device_shift_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
