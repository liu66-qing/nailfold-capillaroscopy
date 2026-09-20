"""
ORACLE TEST for crossing_ratio -- run BEFORE spending any GPU.

Question
--------
Our own 血管数据集 carries HUMAN-drawn boxes with a `cross_vessel` class
(10880 boxes). 96 of its 582 original_ids were provenance-matched to
development cases. So for those cases we can compute the crossing ratio
from HUMAN boxes -- no detector, no training, no model error at all.

Those human boxes are the ORACLE: the ceiling a perfect detector would
reach. If the oracle ratio does not track our clinical `crossing_ratio`
label, then NO amount of detector training on either 血管数据集 or the HF
Capillary-Dataset can deliver this field, because the failure is in the
label-to-box relationship, not in the model.

This mirrors the lesson already recorded for the segmenter
(nailfold-seg-misses-abnormal): mAP looked fine while the downstream sign
was reversed. Test the ceiling first this time.

Governance
----------
- locked-47 is never read. Only EXACT_RECOVERED_DEVELOPMENT rows are used
  and the result is asserted against the locked id set.
- No model is trained or run. No image is transmitted anywhere.
- Ratios are unitless counts; no micron claim is made anywhere.
- Writes only to its own output directory.
"""

import os
import json
import collections

import numpy as np
import pandas as pd

ROOT = r"E:\甲劈微循环"
MAPPING = os.path.join(
    ROOT, "artifacts", "audits", "vascular_dataset_governance_20260830",
    "source_case_mapping.csv")
LABELS = os.path.join(ROOT, "data", "yolo_det_3class", "all_labels")
MANIFEST = os.path.join(ROOT, "artifacts", "manifest", "locked_evaluation_v1.csv")
OUT_DIR = os.path.join(ROOT, "artifacts", "evidence", "oracle_crossing_20260920")

# from data/yolo_det_3class/fold0/dataset.yaml
CLS_VESSEL, CLS_MALFORMED, CLS_CROSS = 0, 1, 2


def read_boxes(original_id):
    """All boxes for every augmentation of one original_id.

    The 20 augmentations of one original are geometric variants of the SAME
    nailfold, so their counts are averaged rather than pooled -- pooling
    would weight originals by how many augmentations happen to exist.
    """
    per_aug = []
    for k in range(64):  # 20 expected; scan wider in case the count varies
        p = os.path.join(LABELS, f"{original_id}_{k}.txt")
        if not os.path.exists(p):
            continue
        c = collections.Counter()
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                c[int(float(line.split()[0]))] += 1
        per_aug.append(c)
    return per_aug


def ratios(per_aug):
    """crossing and malformation ratio in OUR field definition.

    denominator = all loops = vessel + malformed + cross, matching how the HF
    audit defined it so the two sources stay comparable.
    """
    cr, mr, tot = [], [], []
    for c in per_aug:
        n = c[CLS_VESSEL] + c[CLS_MALFORMED] + c[CLS_CROSS]
        if n == 0:
            continue
        cr.append(c[CLS_CROSS] / n)
        mr.append(c[CLS_MALFORMED] / n)
        tot.append(n)
    if not cr:
        return None
    return dict(crossing=float(np.mean(cr)), malformation=float(np.mean(mr)),
                n_loops=float(np.mean(tot)), n_aug=len(cr))


# our clinical label is a bracket string; map to the bracket midpoint so a
# rank correlation is meaningful. '10--30%' appears once and is kept.
BIN_MID = {
    "<=30%": 0.15, "30--60%": 0.45, "60--80%": 0.70, ">80%": 0.90,
    "10--30%": 0.20,
    "<=10%": 0.05, ">60%": 0.80,
}


def spearman(x, y):
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def perm_p(x, y, rho, n_perm=20000, seed=0):
    """Permutation p-value, two-sided. n is ~50 so an asymptotic p is unsafe."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    hits = 0
    for _ in range(n_perm):
        if abs(spearman(x, rng.permutation(y))) >= abs(rho):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    dev_ids = set(man.loc[man["development_fold"].notna(), "exam_case_id"])
    locked_ids = set(man.loc[man["development_fold"].isna(), "exam_case_id"])

    mp = pd.read_csv(MAPPING)
    dev_rows = mp[mp["source_mapping_status"] == "EXACT_RECOVERED_DEVELOPMENT"]

    rec = []
    for _, r in dev_rows.iterrows():
        case = str(r["development_cases"])
        if case in locked_ids:
            raise AssertionError(f"locked case {case} present in DEV mapping")
        if case not in dev_ids:
            continue
        rr = ratios(read_boxes(r["original_id"]))
        if rr is None:
            continue
        rr.update(original_id=int(r["original_id"]), exam_case_id=case)
        rec.append(rr)

    box = pd.DataFrame(rec)
    # several original_ids map to the SAME dev case (85 and 86 -> a1/29), so
    # average them: case-level uniqueness is the standing rule here.
    box_case = box.groupby("exam_case_id", as_index=False).agg(
        crossing=("crossing", "mean"), malformation=("malformation", "mean"),
        n_loops=("n_loops", "mean"), n_orig=("original_id", "nunique"))

    lab = man[["exam_case_id", "crossing_ratio", "malformation_ratio",
               "development_fold"]].copy()
    df = box_case.merge(lab, on="exam_case_id", how="left")
    assert df["exam_case_id"].is_unique
    assert df["development_fold"].notna().all(), "a non-development case slipped in"

    out = {
        "n_mapped_dev_original_ids": int(len(dev_rows)),
        "n_cases_with_human_boxes": int(len(df)),
        "mean_loops_per_image": round(float(df["n_loops"].mean()), 2),
        "governance": {
            "locked_cases_seen": 0,
            "models_run": 0,
            "images_transmitted": 0,
            "micron_claims": 0,
            "external_boxes_are_clinical_gold_standard": False,
        },
        "fields": {},
    }

    for field, bcol in (("crossing_ratio", "crossing"),
                        ("malformation_ratio", "malformation")):
        sub = df[df[field].notna() & df[field].isin(BIN_MID)].copy()
        sub["y"] = sub[field].map(BIN_MID)
        res = {
            "n_labelled_cases": int(len(sub)),
            "label_distribution": {str(k): int(v) for k, v in
                                   sub[field].value_counts().items()},
            "oracle_ratio_median": round(float(sub[bcol].median()), 4),
            "oracle_ratio_p25_p75": [round(float(sub[bcol].quantile(.25)), 4),
                                     round(float(sub[bcol].quantile(.75)), 4)],
        }
        if len(sub) >= 8 and sub["y"].nunique() >= 2:
            rho = spearman(sub[bcol].to_numpy(), sub["y"].to_numpy())
            res["spearman_rho"] = round(rho, 4)
            res["perm_p_two_sided"] = round(perm_p(sub[bcol].to_numpy(),
                                                   sub["y"].to_numpy(), rho), 5)
            res["per_label_oracle_mean"] = {
                str(k): round(float(v), 4) for k, v in
                sub.groupby(field)[bcol].mean().items()}
            res["per_label_n"] = {str(k): int(v) for k, v in
                                  sub.groupby(field)[bcol].size().items()}
            # a positive rho that is not monotonic across bins is not usable
            # as a product signal, so record the bin means in label order.
            order = sorted(res["per_label_oracle_mean"].items(),
                           key=lambda kv: BIN_MID[kv[0]])
            vals = [v for _, v in order]
            res["bin_order"] = [k for k, _ in order]
            res["bin_means_in_label_order"] = vals
            res["monotonic_increasing"] = all(
                b >= a for a, b in zip(vals, vals[1:]))
        else:
            res["spearman_rho"] = None
            res["note"] = "too few labelled cases or single label value"
        out["fields"][field] = res

    with open(os.path.join(OUT_DIR, "oracle.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    df.to_csv(os.path.join(OUT_DIR, "per_case_oracle.csv"), index=False,
              encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
