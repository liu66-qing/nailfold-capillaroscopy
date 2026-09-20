"""
Case-level evaluation of `malformation_ratio` from detector output (stage S2 verdict).

This is the ONLY number that decides whether the field is deliverable. The
detector's mAP is an internal metric on augmented copies of the same
nailfolds and says nothing about the field.

Design
------
- Evaluation set: the 119 development cases that have a `malformation_ratio`
  label AND no human boxes in the detector's training pool. The 44 cases used
  by the oracle test DO have human boxes, so they are contaminated and are
  excluded here. Case-level, one row per exam_case_id.
- Comparison the verdict rests on:
    mode baseline          0.538 accuracy on this subset (prevalence 0.462)
    oracle (human boxes)   AUROC 0.749 [0.585, 0.894]   <- the ceiling
    detector               what this script measures
  A detector AUROC that does not clear the CI floor of the oracle is not
  evidence of a working field.
- Two confounds are measured, not assumed:
    n_loops (total detections)  -- scored 0.575 on the oracle data
    cross_vessel ratio          -- scored 0.459
  If the malformed ratio does not beat both, the signal is a count artifact.
- Bracket labels map to a binary target (>10% malformed). The bracketed
  template value '[<10%]' is dropped to `unknown`, never auto-classified.

Governance
----------
- Reads only development cases. locked-47 is never opened; the case list is
  built from development_fold.notna() and asserted against the locked set.
- No image leaves the machine.
- Reports per-fold and pooled numbers with bootstrap CIs. No threshold is
  tuned on the evaluation set; the ratio is used as a continuous score and
  AUROC is threshold-free.
"""

import os
import json
import glob
import argparse

import numpy as np
import pandas as pd

ROOT = r"E:\甲劈微循环"
MANIFEST = os.path.join(ROOT, "artifacts", "manifest", "locked_evaluation_v1.csv")
MAPPING = os.path.join(
    ROOT, "artifacts", "audits", "vascular_dataset_governance_20260830",
    "source_case_mapping.csv")
EXP = os.path.join(ROOT, "artifacts", "experiments", "det_malformation_20260920")
OUT_DIR = os.path.join(ROOT, "artifacts", "evidence", "malformation_field_20260920")

# reference detection density from the human boxes on the 50 mapped cases
# (artifacts/evidence/oracle_crossing_20260920/oracle.json mean_loops_per_image)
HUMAN_LOOPS_PER_IMAGE = 6.84

POS = {"10--30%", "30--60%", ">60%"}
NEG = {"<=10%"}
CLS_VESSEL, CLS_MALFORMED, CLS_CROSS = 0, 1, 2


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
    y = np.asarray(y)
    s = np.asarray(s, float)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(auroc(y[i], s[i]))
    return (round(float(np.percentile(vals, 2.5)), 3),
            round(float(np.percentile(vals, 97.5)), 3))


def case_images(d, caporg_only=True):
    """Capillary frames for one case.

    A case directory mixes real capillaroscopy frames (CAPorg*.jpg, 1097 of
    1586 across the evaluation set) with scanned report pages (rep_*.jpg, 489).
    Running a vessel detector over a report scan yields noise boxes that dilute
    the case-level ratio, so report pages are excluded by default.

    This choice is fixed BEFORE any field result is computed, not selected
    afterwards: it drops 31% of images and loses 0 of the 118 cases, i.e. every
    case retains at least one real frame, so it cannot be a cherry-pick. The
    --no-caporg-filter flag exists to report the unfiltered number alongside.
    """
    imgs = sorted(x for x in os.listdir(d)
                  if x.lower().endswith((".jpg", ".jpeg", ".png")))
    if caporg_only:
        cap = [x for x in imgs if x.startswith("CAPorg")]
        if cap:
            return cap
    return imgs


def eval_cases(weights, cases, imgsz, conf, device=0, max_det=300,
               caporg_only=True, min_gray_std=8.0):
    """Run the detector over every image of every case; aggregate per case."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(weights)
    rows = []
    for case in cases:
        d = os.path.join(ROOT, "data", *case.split("/"))
        imgs = case_images(d, caporg_only)
        per_img = []
        n_blank = 0
        for fn in imgs:
            p = os.path.join(d, fn)
            # cv2.imread returns None on these non-ASCII paths on Windows;
            # imdecode(fromfile) is the working form. ~2.7% of frames are
            # near-blank (gray std down to 0.0) and are skipped.
            im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None or float(
                    cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std()) < min_gray_std:
                n_blank += 1
                continue
            r = model.predict(p, imgsz=imgsz, conf=conf,
                              device=device, max_det=max_det, verbose=False)[0]
            c = r.boxes.cls.cpu().numpy().astype(int)
            nm = int((c == CLS_MALFORMED).sum())
            nx = int((c == CLS_CROSS).sum())
            nv = int((c == CLS_VESSEL).sum())
            tot = nm + nx + nv
            if tot == 0:
                continue
            per_img.append((nm / tot, nx / tot, tot))
        if not per_img:
            rows.append(dict(exam_case_id=case, mal_ratio=np.nan,
                             cross_ratio=np.nan, n_loops=0.0, n_img=len(imgs),
                             n_img_used=0, n_blank=n_blank))
            continue
        a = np.array(per_img, float)
        rows.append(dict(exam_case_id=case, mal_ratio=float(a[:, 0].mean()),
                         cross_ratio=float(a[:, 1].mean()),
                         n_loops=float(a[:, 2].mean()), n_img=len(imgs),
                         n_img_used=len(per_img), n_blank=n_blank))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="default: best.pt of every fold found in EXP")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-sweep", default=None,
                    help="comma list, e.g. 0.05,0.10,0.15,0.25,0.40. The "
                         "reported operating point is chosen by DETECTION "
                         "DENSITY closest to the human-box reference of 6.84 "
                         "loops/image, NOT by best AUROC -- picking the "
                         "threshold that maximises the field metric on the "
                         "evaluation set would be selection on the test set.")
    ap.add_argument("--tag", default="s768")
    ap.add_argument("--no-caporg-filter", action="store_true",
                    help="also score scanned report pages (rep_*.jpg)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["exam_case_id"] = man["exam_case_id"].astype(str)
    locked = set(man.loc[man["development_fold"].isna(), "exam_case_id"])
    dev = man[man["development_fold"].notna()].copy()

    mp = pd.read_csv(MAPPING)
    contaminated = set(mp.loc[mp["source_mapping_status"] ==
                              "EXACT_RECOVERED_DEVELOPMENT",
                              "development_cases"].astype(str))

    lab = dev[dev["malformation_ratio"].isin(POS | NEG)].copy()
    clean = lab[~lab["exam_case_id"].isin(contaminated)].copy()
    assert not (set(clean["exam_case_id"]) & locked), "locked case in eval set"
    clean["y"] = clean["malformation_ratio"].isin(POS).astype(int)

    wlist = ([args.weights] if args.weights else
             sorted(glob.glob(os.path.join(EXP, f"fold*_{args.tag}",
                                           "weights", "best.pt"))))
    if not wlist:
        raise SystemExit(f"no weights found under {EXP} for tag {args.tag}")

    out = {
        "n_eval_cases": int(len(clean)),
        "n_positive": int(clean["y"].sum()),
        "prevalence": round(float(clean["y"].mean()), 3),
        "mode_baseline_accuracy": round(
            float(max(clean["y"].mean(), 1 - clean["y"].mean())), 3),
        "oracle_ceiling_auroc": 0.749,
        "oracle_ceiling_ci": [0.585, 0.894],
        "conf": args.conf, "imgsz": args.imgsz,
        "caporg_only": not args.no_caporg_filter,
        "image_selection_note": (
            "report scans (rep_*.jpg, 489 of 1586) excluded and near-blank "
            "frames (gray std < 8) skipped; fixed before any result was "
            "computed, loses 0 of 118 cases"),
        "weights_evaluated": [os.path.relpath(w, ROOT) for w in wlist],
        "per_fold": {},
        "governance": {"locked_cases_seen": 0, "images_transmitted": 0,
                       "threshold_tuned_on_eval_set": False},
    }

    confs = ([float(x) for x in args.conf_sweep.split(",")]
             if args.conf_sweep else [args.conf])
    if len(confs) > 1:
        out["conf_sweep"] = {}
        out["operating_point_rule"] = (
            "detection density closest to the human-box reference of "
            f"{HUMAN_LOOPS_PER_IMAGE} loops/image; NOT best AUROC")
        for cf in confs:
            df = eval_cases(wlist[0], list(clean["exam_case_id"]), args.imgsz,
                            cf, caporg_only=not args.no_caporg_filter)
            df = df.merge(clean[["exam_case_id", "y"]], on="exam_case_id")
            ok = df[df["mal_ratio"].notna()]
            out["conf_sweep"][f"{cf}"] = {
                "n_scored": int(len(ok)),
                "loops_per_image": round(float(df["n_loops"].mean()), 2),
                "density_gap_vs_human": round(
                    abs(float(df["n_loops"].mean()) - HUMAN_LOOPS_PER_IMAGE), 2),
                "frac_ratio_saturated_at_1": round(
                    float((ok["mal_ratio"] >= 0.999).mean()), 3),
                "auroc": round(auroc(ok["y"], ok["mal_ratio"]), 3),
            }
        pick = min(out["conf_sweep"],
                   key=lambda k: out["conf_sweep"][k]["density_gap_vs_human"])
        out["selected_conf"] = float(pick)
        out["selected_by"] = "detection density, not AUROC"
        confs = [float(pick)]

    preds = {}
    for w in wlist:
        fold = os.path.basename(os.path.dirname(os.path.dirname(w)))
        df = eval_cases(w, list(clean["exam_case_id"]), args.imgsz, confs[0],
                        caporg_only=not args.no_caporg_filter)
        df = df.merge(clean[["exam_case_id", "y", "development_fold",
                             "malformation_ratio"]], on="exam_case_id")
        df.to_csv(os.path.join(OUT_DIR, f"per_case_{fold}_conf{args.conf}.csv"),
                  index=False, encoding="utf-8")
        preds[fold] = df
        ok = df[df["mal_ratio"].notna()]
        r = {"n_scored": int(len(ok)),
             "n_cases_with_zero_detections": int((df["n_img_used"] == 0).sum())}
        for nm, col in (("malformation_ratio", "mal_ratio"),
                        ("confound_n_loops", "n_loops"),
                        ("confound_cross_ratio", "cross_ratio")):
            a = auroc(ok["y"], ok[col])
            r[nm] = {"auroc": round(a, 3),
                     "ci95": list(boot_ci(ok["y"].to_numpy(), ok[col].to_numpy()))}
        r["beats_both_confounds"] = bool(
            r["malformation_ratio"]["auroc"] > r["confound_n_loops"]["auroc"]
            and r["malformation_ratio"]["auroc"] > r["confound_cross_ratio"]["auroc"])
        r["ci_floor_above_chance"] = bool(r["malformation_ratio"]["ci95"][0] > 0.5)
        out["per_fold"][fold] = r

    # ensemble: mean ratio across folds, which is the configuration a product
    # would actually ship; reported alongside, not instead of, per-fold
    if len(preds) > 1:
        base = None
        for fold, df in preds.items():
            k = df[["exam_case_id", "y", "mal_ratio", "n_loops",
                    "cross_ratio"]].set_index("exam_case_id")
            base = k if base is None else base.add(
                k[["mal_ratio", "n_loops", "cross_ratio"]], fill_value=0)
        n = len(preds)
        ens = base.copy()
        for c in ("mal_ratio", "n_loops", "cross_ratio"):
            ens[c] = ens[c] / n
        ens = ens.dropna(subset=["mal_ratio"])
        out["ensemble"] = {
            "n_scored": int(len(ens)),
            "malformation_ratio_auroc": round(auroc(ens["y"], ens["mal_ratio"]), 3),
            "ci95": list(boot_ci(ens["y"].to_numpy(), ens["mal_ratio"].to_numpy())),
        }

    with open(os.path.join(OUT_DIR, f"verdict_{args.tag}_conf{args.conf}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
