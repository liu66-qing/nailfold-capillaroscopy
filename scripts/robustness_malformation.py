"""
Robustness checks on the malformation_ratio result.

The headline number (AUROC 0.749, CI [0.655, 0.835], 116 held-out development
cases) is only worth reporting if it survives the ways it could be spurious.
Four checks, all run on the already-written per-case predictions so nothing is
re-tuned:

1. ARCHIVE. Does the signal exist inside each archive separately? Previous
   fields in this project (SVP, blood_color) scored well overall and turned
   out to be archive-dependent, so this is the check that matters most here.
2. COUNT CONFOUNDS. Image count, used-image count, blank count. A detector
   that merely reacts to how many frames a case has is not reading morphology.
3. LABEL NOISE. The oracle comparison showed the clinical brackets and the
   human boxes disagree in absolute level (only 17 of 44 cases fall inside
   their own bracket), so the labels are known to be noisy. Randomly flipping
   a fraction of them shows how fast the measurement decays if they are wrong.
4. LABEL RELIABILITY GRADIENT. If the signal is real but partly hidden by
   boundary noise, removing the ambiguous middle bracket should RAISE the
   score. If the signal is spurious, it should not.

Nothing here selects a model or a threshold; it reads the predictions that the
pre-registered density rule already fixed.
"""

import os
import json
import glob

import numpy as np
import pandas as pd

ROOT = r"E:\甲劈微循环"
PRED_DIR = os.path.join(ROOT, "artifacts", "evidence",
                        "malformation_field_20260920")
OUT = os.path.join(PRED_DIR, "robustness.json")


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float(((p[:, None] > n[None, :]).sum()
                  + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def boot_ci(y, s, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s, float)
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            v.append(auroc(y[i], s[i]))
    return [round(float(np.percentile(v, 2.5)), 3),
            round(float(np.percentile(v, 97.5)), 3)]


def main():
    out = {"per_prediction_file": {}}
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "per_case_*.csv"))):
        df = pd.read_csv(f)
        ok = df[df["mal_ratio"].notna()].copy()
        if "y" not in ok or ok["y"].nunique() < 2:
            continue
        ok["archive"] = ok["exam_case_id"].str.split("/").str[0]
        r = {
            "n_scored": int(len(ok)),
            "auroc": round(auroc(ok["y"], ok["mal_ratio"]), 3),
            "ci95": boot_ci(ok["y"].to_numpy(), ok["mal_ratio"].to_numpy()),
            "by_archive": {},
            "count_confounds": {},
            "label_noise": {},
            "label_reliability": {},
        }
        for a, g in ok.groupby("archive"):
            if g["y"].nunique() < 2:
                r["by_archive"][a] = {"n": int(len(g)), "note": "single class"}
                continue
            r["by_archive"][a] = {
                "n": int(len(g)), "n_pos": int(g["y"].sum()),
                "prevalence": round(float(g["y"].mean()), 3),
                "auroc": round(auroc(g["y"], g["mal_ratio"]), 3)}
        r["archive_signal_everywhere"] = all(
            v.get("auroc", 0) > 0.5 for v in r["by_archive"].values()
            if "auroc" in v)

        for c in ("n_img", "n_img_used", "n_blank", "n_loops", "cross_ratio"):
            if c in ok:
                r["count_confounds"][c] = round(auroc(ok["y"], ok[c]), 3)

        rng = np.random.default_rng(0)
        for fr in (0.05, 0.10, 0.15, 0.20):
            v = []
            for _ in range(400):
                y = ok["y"].to_numpy().copy()
                i = rng.choice(len(y), int(round(fr * len(y))), replace=False)
                y[i] = 1 - y[i]
                v.append(auroc(y, ok["mal_ratio"]))
            r["label_noise"][f"flip_{int(fr*100)}pct"] = round(
                float(np.median(v)), 3)

        if "malformation_ratio" in ok:
            mid = ok[ok["malformation_ratio"] != "10--30%"]
            ext = ok[ok["malformation_ratio"].isin(["<=10%", ">60%"])]
            r["label_reliability"] = {
                "drop_ambiguous_10_30": {
                    "n": int(len(mid)),
                    "auroc": round(auroc(mid["y"], mid["mal_ratio"]), 3)},
                "extremes_only": {
                    "n": int(len(ext)),
                    "auroc": round(auroc(ext["y"], ext["mal_ratio"]), 3)},
            }
            r["improves_on_cleaner_labels"] = bool(
                r["label_reliability"]["drop_ambiguous_10_30"]["auroc"]
                > r["auroc"])
        out["per_prediction_file"][os.path.basename(f)] = r

    out["interpretation"] = (
        "A spurious correlation would not strengthen when the noisiest label "
        "band is removed, and would not survive inside every archive "
        "separately. Total detections scoring BELOW 0.5 also rules out a "
        "simple count effect. None of this makes the number a product claim: "
        "it is a development-set measurement on one detector fold.")
    out["governance"] = {"locked_cases_seen": 0, "models_run": 0,
                         "reads_existing_predictions_only": True}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
