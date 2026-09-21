"""Is the inverted density sign an ANNOTATION problem or a DETECTOR problem?

Three of our fields show a correlation whose sign is backwards:
  crossing_ratio / malformation_ratio   rho = -0.257  (known)
  capillary_count vs vessel spacing     rho = +0.161  (found 2026-09-20; denser
      capillaries must mean SMALLER spacing, so positive is inverted)

Those were all computed from OUR segmenter's output, so they cannot separate the
two hypotheses:
  H_detector : the segmenter mislocates vessels, so any density we derive is wrong
  H_label    : the segmenter is fine and the clinical LABEL does not mean what we
               assume (e.g. capillary_count is per-mm at the operator's own
               magnification, or counted on a different field of view)

The 血管数据集 gives the separation. Its 分类数据集/annotations/*.xml are
Pascal-VOC vessel boxes drawn by an INDEPENDENT annotator, and
governance/source_case_mapping.csv maps 96 of those source images onto 50 of our
development cases by EXACT MD5 -- same pixels, independent vessel positions.

So on those 50 cases we can compute density twice:
  spacing_ext  from the external boxes
  spacing_ours from our SAM/YOLO proposals
and correlate each against the clinical capillary_count label.

  if spacing_ext correlates NEGATIVELY (correct sign) and ours stays positive
      -> H_detector. Our detector is the problem; retraining on better vessel
         locations (incl. the Mendeley clusters) is the right fix.
  if BOTH are positive / inverted
      -> H_label. The label does not mean what we assume, and no amount of
         detector work fixes it. Feeding Mendeley in would only amplify the error.
  if they disagree in magnitude only
      -> inconclusive, report as such.

Governance constraints honoured:
  - the 血管数据集 is HOLD_NO_UNCONDITIONAL_TRAIN_ASSETS. This script does not
    TRAIN on it; it reads annotation coordinates for a diagnostic correlation.
  - locked cases are excluded at CASE level via development_fold; the mapping
    file's own locked column is cross-checked and must contribute 0 rows.
  - no absolute micron claim is made anywhere: spacing is in pixels, normalised
    to a common width, and only the SIGN of a rank correlation is interpreted.
  - external annotation is NOT a clinical gold standard; it is used only as a
    second, independent estimate of where vessels are.
"""
import collections
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

GOV = "artifacts/audits/vascular_dataset_governance_20260830"
MAPPING = os.path.join(GOV, "source_case_mapping.csv")
XML_DIR = "data/血管数据集/分类数据集/annotations"
OURS = "artifacts/features/segmiss_v1/ai_proposed_misses.jsonl"
MANIFEST = "artifacts/manifest/locked_evaluation_v1.csv"
OUT = "artifacts/evidence/inverted_sign_20260920"
REF_W = 1024.0
ORDER = {"<1": 0, "3--4": 1, "5--6": 2, ">=7": 3}


def nn_spacing(xy):
    if len(xy) < 6:
        return None
    D = np.hypot(xy[:, 0:1] - xy[:, 0], xy[:, 1:2] - xy[:, 1])
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(1)))


def external_boxes():
    """original_id -> (centroids, declared image size) from the VOC xml."""
    out = {}
    for p in sorted(glob.glob(os.path.join(XML_DIR, "*.xml"))):
        stem = os.path.basename(p)[:-4]
        oid = stem.split("_")[0]
        try:
            r = ET.parse(p).getroot()
        except Exception:
            continue
        sz = r.find("size")
        w = float(sz.find("width").text) if sz is not None else None
        cs = []
        for ob in r.findall("object"):
            bb = ob.find("bndbox")
            if bb is None:
                continue
            x0, y0 = float(bb.find("xmin").text), float(bb.find("ymin").text)
            x1, y1 = float(bb.find("xmax").text), float(bb.find("ymax").text)
            cs.append(((x0 + x1) / 2, (y0 + y1) / 2))
        if cs:
            out.setdefault(oid, {"pts": [], "w": w})
            out[oid]["pts"].extend(cs)
            if w:
                out[oid]["w"] = w
    return out


def our_spacing_by_case():
    per = collections.defaultdict(list)
    with open(OURS, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            o = json.loads(ln)
            per[(o["exam_case_id"], o["image_path"])].append(
                np.array(o["poly"], float).mean(0))
    by = collections.defaultdict(list)
    for (cid, path), pts in per.items():
        s = nn_spacing(np.array(pts))
        if s is None:
            continue
        try:
            w, _h = Image.open(os.path.join("data", path)).size
        except Exception:
            continue
        by[cid].append(s / w * REF_W)
    return {k: float(np.median(v)) for k, v in by.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    dev = set(m.loc[m["development_fold"].notna(), "exam_case_id"])
    locked = set(m.loc[m["development_fold"].isna(), "exam_case_id"])

    mp = pd.read_csv(MAPPING)
    mp = mp[mp["development_cases"].notna()].copy()
    # hard case-level guard
    leak = [c for c in mp["development_cases"].unique() if c in locked]
    assert not leak, "locked case present in mapping: %s" % leak

    ext = external_boxes()
    ours = our_spacing_by_case()

    # external spacing per dev case (a case may have several source images)
    acc = collections.defaultdict(list)
    n_xml_hit = 0
    for _i, row in mp.iterrows():
        oid = str(row["original_id"])
        cid = row["development_cases"]
        if oid not in ext:
            continue
        e = ext[oid]
        s = nn_spacing(np.array(e["pts"], float))
        if s is None or not e["w"]:
            continue
        n_xml_hit += 1
        acc[cid].append(s / float(e["w"]) * REF_W)
    ext_case = {k: float(np.median(v)) for k, v in acc.items()}

    lab = m.set_index("exam_case_id")["capillary_count"].astype(str)
    rows = []
    for cid, se in ext_case.items():
        if cid not in dev:
            continue
        lv = lab.get(cid)
        if lv not in ORDER:
            continue
        rows.append({"exam_case_id": cid, "label_ord": ORDER[lv],
                     "spacing_ext": se, "spacing_ours": ours.get(cid, np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "paired_spacing.csv"), index=False)

    res = {"n_source_images_mapped": int(len(mp)),
           "n_xml_with_boxes_hit": n_xml_hit,
           "n_dev_cases_with_external_spacing": int(len(df)),
           "locked_cases_seen": 0,
           "models_run": 0,
           "external_annotation_is_gold_standard": False,
           "micron_claim": "none; pixel spacing only, sign of rank correlation only"}

    def corr(col):
        """Rank correlation WITH its CI, and a sign call that requires the CI to
        exclude 0.

        The first version of this function called any rho > 0 "INVERTED". That
        reads a direction off noise: at n=50 a rho of 0.07 (p=0.62) is
        indistinguishable from zero, and "inverted" and "no relationship" are
        different findings with different consequences. A sign is only named when
        the bootstrap CI excludes 0.
        """
        d = df[[col, "label_ord"]].dropna()
        if len(d) < 12 or d[col].nunique() < 3:
            return {"n": int(len(d)), "rho": None, "p": None,
                    "sign": None, "verdict": "insufficient n"}
        rho, p = stats.spearmanr(d[col], d["label_ord"])
        x = d[col].to_numpy(float)
        y = d["label_ord"].to_numpy(float)
        rng = np.random.default_rng(20260920)
        bs = []
        for _ in range(4000):
            i = rng.integers(0, len(x), len(x))
            if len(np.unique(y[i])) < 2 or len(np.unique(x[i])) < 2:
                continue
            r = stats.spearmanr(x[i], y[i]).statistic
            if not np.isnan(r):
                bs.append(r)
        lo, hi = (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))) \
            if len(bs) >= 100 else (float("nan"), float("nan"))
        if not (lo > 0 or hi < 0):
            sign = "indistinguishable from zero"
        elif rho < 0:
            sign = "negative (CORRECT)"
        else:
            sign = "positive (INVERTED)"
        # smallest |rho| this n could have resolved at 80% power, two-sided 0.05
        mde = float(np.tanh(2.80 / np.sqrt(max(len(d) - 3, 1))))
        return {"n": int(len(d)), "rho": round(float(rho), 4),
                "p": round(float(p), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "sign": sign, "min_detectable_rho_80pct_power": round(mde, 3)}

    res["external_boxes"] = corr("spacing_ext")
    res["our_detector"] = corr("spacing_ours")
    res["expected_sign"] = ("NEGATIVE: more capillaries per mm must mean "
                            "smaller nearest-neighbour spacing")

    e, o = res["external_boxes"], res["our_detector"]
    ez = e["sign"] == "indistinguishable from zero"
    oz = o["sign"] == "indistinguishable from zero"
    if e["rho"] is None or o["rho"] is None:
        res["verdict"] = "INCONCLUSIVE: not enough paired cases"
    elif ez and oz:
        res["verdict"] = (
            "NEITHER HYPOTHESIS TESTED. Both correlations are indistinguishable "
            "from zero, so this design cannot separate H_detector from H_label. "
            "The finding is that spacing carries no measurable information about "
            "the capillary_count label from either source -- which is itself the "
            "useful result: it is not that our detector is worse than an "
            "independent annotator, it is that this label is not a density "
            "readout at all. Consistent with capillary_count being blocked by "
            "the calibration ban (a per-mm count needs a pixel scale, and all "
            "four required evidence items are missing), so no spacing proxy can "
            "stand in for it.")
    elif ez or oz:
        res["verdict"] = ("INCONCLUSIVE: one of the two correlations is "
                          "indistinguishable from zero, so the comparison has no "
                          "contrast to interpret.")
    elif e["rho"] < 0 and o["rho"] > 0:
        res["verdict"] = ("H_detector: independent boxes give the CORRECT sign "
                          "while ours is inverted. The detector is the problem; "
                          "better vessel locations are the right fix.")
    elif e["rho"] > 0 and o["rho"] > 0:
        res["verdict"] = ("H_label: BOTH are inverted. The clinical label does "
                          "not mean what we assume, and no detector work fixes "
                          "it. Do NOT feed external instances into the segmenter "
                          "expecting this to resolve.")
    else:
        res["verdict"] = "INCONCLUSIVE: signs do not separate the hypotheses"
    res["superseded_verdict"] = (
        "An earlier run of this script printed 'H_label: BOTH are inverted'. "
        "That was produced by a judge that called any rho > 0 inverted, with no "
        "regard for whether the estimate differed from zero. It is withdrawn.")

    with open(os.path.join(OUT, "inverted_sign.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
