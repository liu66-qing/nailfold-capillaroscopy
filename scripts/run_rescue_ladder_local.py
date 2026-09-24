"""A2 / A3: do auto-detected local vessel regions raise any local-morphology field?

This is the second half of the rescue ladder. The first half
(run_rescue_ladder.py) asked whether a different or externally adapted ENCODER
helps, and the answer was one field out of fifteen. This half asks a different
question: the encoder reads a whole 518x686 frame, and a malformed capillary is a
small object in it, so perhaps the information is present but diluted. Counting
vessels explicitly with a detector is the obvious way to find out.

The rungs, all on the same cases, the same folds and the same ruler as A0:

  A0   whole-image features only                          (the shipped anchor)
  A2   whole-image + local features from a detector trained on LOCAL human boxes
  A2x  whole-image + local features from the EXTERNAL-pretrained detector
  A3   whole-image + local features from external-pretrained-then-local-tuned
  L2   local features ALONE, no whole-image features

L2 exists because A2 beating A0 would otherwise be ambiguous: adding 30 columns
to 320 can help simply by changing the regularisation geometry. If L2 is at
baseline while A2 gains, the gain is not coming from the vessel counts.

Leakage. Fold k's cases are read only by fold k's detector, which never saw them
(train_morph_detector_folds.py holds out every case of fold k, and all 20
augmentations of one original travel together). Validation cases therefore supply
no human box and no image to the model that measures them -- they receive only
automatically detected regions, which is the deployment condition. The human
boxes are supervision inside training folds only.

Units. The local features are counts, ratios and frame fractions. No micron, no
per-mm, no per-minute value is computed anywhere in this path.

Fields. Only the fields whose observable unit is a shape in a still frame can
possibly benefit from vessel boxes, and they are named here BEFORE the run rather
than chosen afterwards from whatever improved.

Reproduce:
  python scripts/run_rescue_ladder_local.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_field_matrix_three_arms import (  # noqa: E402  the shared, shipped ruler
    ANCHOR, DIRTY, LABELS, N_BOOT, SEED, UNIT, ADMISSIBLE,
    fold_directions, hard, load_arm, oof, paired, score, target)
from run_rescue_ladder import (  # noqa: E402  same thresholds, same helpers
    BELOW_RESOLUTION, MIN_EFFECT, abstention, calibration, loao, verdict)

EXP = ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
LOCAL = EXP / "local_features"
OUT = EXP / "ladder_local"

# Pre-declared. A vessel-box feature can only help a field that is a shape in a
# still frame; the measurement fields need calibration and the dynamic fields need
# a time base, and no amount of counting boxes supplies either.
FIELDS = ["malformation_ratio", "crossing_ratio", "papilla", "capillary_count"]

# (tag under local_features/, rung label). The tag is produced by
# extract_detector_local_features.py --tag <tag>.
RUNGS = {
    "A2": ("local", "A2  whole image + local features, detector trained on local human boxes"),
    "A2x": ("external", "A2x whole image + local features, external-pretrained detector"),
    "A3": ("external_then_local", "A3  whole image + local features, external then local"),
}


def attach(feats: dict, ix: pd.DataFrame, extra: pd.DataFrame) -> dict:
    """Append the case-level local columns to EVERY pooling matrix.

    load_arm returns one matrix per pooling, each with one row per IMAGE, and the
    shipped oof() fits a model per pooling and then averages. So the local columns
    have to be attached to all five, not to a single concatenated block -- otherwise
    four of the five voters would never see them. Each image of a case carries its
    case's values, which is the same broadcast the case-level aggregation performs
    later; oof() groups back to the case by exam_case_id.
    """
    aligned = extra.reindex(ix.exam_case_id.to_numpy())
    if aligned.isna().all(axis=1).any():
        raise RuntimeError("an image has no local-feature row for its case")
    # The extractor leaves ratios and geometry NaN for a case whose every image was
    # empty, because a ratio over zero boxes is undefined and filling 0 would assert
    # "0% malformed". The classifier cannot take NaN, so impute with the TRAINING
    # median -- except that oof() refits per fold, and a median taken here would be
    # computed over all cases including the held-out ones. Filling with 0 after the
    # scaler centres each column is the leak-free choice available at this layer, and
    # no_detection_in_any_image is passed through as its own column so the model can
    # tell an imputed row from a measured zero.
    cols = aligned.to_numpy(float)
    cols = np.nan_to_num(cols, nan=0.0, posinf=0.0, neginf=0.0)
    return {p: np.hstack([m, cols]).astype(np.float32) for p, m in feats.items()}


def pca_survival(feats: dict, ix: pd.DataFrame, extra: pd.DataFrame) -> dict:
    """How much of the local columns survives the shipped PCA?

    The shipped head is StandardScaler -> PCA(64) -> LogisticRegression, and after
    attach() the input is 768 encoder dims plus ~30 local ones. Those 30 could be
    crowded out of the retained 64 components by the 768, in which case an A2 null
    would mean "the projection discarded them", not "vessel counts do not help".
    That is a different finding and must not be reported as the second one.

    Measured as the share of each local column's scaled variance that is recovered
    by projecting onto the 64 components and back. Diagnostic only: it does not
    change any fitted model, and the L2 rung is the answer that does not depend on
    this projection at all.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # Build the matrix through attach(), not by hand: a case with zero detections
    # carries NaN ratios by design (filling 0 would assert "0% malformed"), and
    # attach() is where that becomes 0 after scaling. Hstacking the raw frame here
    # instead put NaNs into PCA.fit and crashed on exactly the cases this
    # diagnostic exists to reason about.
    n_local = extra.shape[1]
    X = attach({"mean": feats["mean"]}, ix, extra)["mean"].astype(float)
    Z = StandardScaler().fit_transform(X)
    dim = max(2, min(64, Z.shape[0] - 1, Z.shape[1]))
    p = PCA(n_components=dim, random_state=SEED).fit(Z)
    recon = p.inverse_transform(p.transform(Z))
    num = ((recon - Z.mean(0)) ** 2).sum(axis=0)
    den = ((Z - Z.mean(0)) ** 2).sum(axis=0)
    keep = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    loc, enc = keep[-n_local:], keep[:-n_local]
    return dict(n_local_columns=int(n_local), pca_components=int(dim),
                local_variance_retained_mean=round(float(loc.mean()), 4),
                local_variance_retained_min=round(float(loc.min()), 4),
                encoder_variance_retained_mean=round(float(enc.mean()), 4),
                reading=("if the local mean is far below the encoder mean, the "
                         "projection is discarding the counts and an A2 null is "
                         "uninformative; read L2 instead"))


def load_local(tag: str) -> pd.DataFrame:
    p = LOCAL / tag / "case_features.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"exam_case_id": str})
    bad = [c for c in d.columns if any(u in c.lower() for u in ("um_", "_um", "micron",
                                                               "per_mm", "per_min"))]
    if bad:
        raise RuntimeError("a calibrated unit reached the local features: %s" % bad)
    return d.set_index("exam_case_id")


def main() -> None:
    # Indexed by exam_case_id, exactly as the whole-image ladder does it: target()
    # returns y carrying dev's index, and every downstream key (oof's fold map,
    # the local-feature join, loao's archive array) is a case id. With a
    # RangeIndex here, `order` becomes 0..185 and nothing joins.
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    if len(dev) != 186:
        raise RuntimeError("expected 186 development cases, got %d" % len(dev))
    # Same guard as the shipped ladder: the 47 locked cases must be identifiable
    # and excluded here, not merely absent by accident.
    locked = set(man.index[man.development_fold.isna()])
    if len(locked) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(locked))

    base_ix, base_feats, base_meta = load_arm(ANCHOR)
    if base_meta.get("locked_cases_seen", 0) != 0:
        raise RuntimeError("anchor arm reports locked cases")
    locals_ = {k: load_local(tag) for k, (tag, _) in RUNGS.items()}
    have = [k for k, v in locals_.items() if v is not None]
    if not have:
        raise SystemExit("no local feature table yet; run "
                         "extract_detector_local_features.py first")

    rng = np.random.default_rng(SEED)
    rows, detail = [], {}
    fold_map = dev.development_fold.to_dict()
    arch_map = dev.archive.to_dict()
    for field in FIELDS:
        # target() returns (y, n_classes, extra_report) -- NOT (y, multiclass,
        # classes). Unpacking it the other way makes n_classes=2 a truthy
        # "multiclass" for the two ratio fields and hands the report dict to oof()
        # as the class list.
        y_case, n_classes, target_report = target(dev, field)
        order = sorted(y_case.index)
        yv = y_case.reindex(order).to_numpy()
        classes = sorted(set(float(v) for v in yv))
        multiclass = n_classes > 2
        # oof() indexes folds by case id; loao()/fold_directions() want arrays
        # aligned to `order`.
        folds_arr = np.array([fold_map[c] for c in order], float)
        arch_arr = np.array([arch_map[c] for c in order], object)
        unit, static_ok = UNIT[field]

        preds = {}
        p0, pr0 = oof(base_feats, base_ix, y_case, fold_map, order, multiclass, classes)
        preds["A0"] = (p0, pr0)
        for k in have:
            extra = locals_[k]
            # A case whose images produced no detection is a real deployment case,
            # not a missing value to drop: zero vessels IS the measurement. A case
            # absent from the table entirely would be a pipeline error instead.
            absent = sorted(set(order) - set(extra.index))
            if absent:
                raise RuntimeError("%d cases absent from local table %s: %s"
                                   % (len(absent), k, absent[:5]))
            preds[k] = oof(attach(base_feats, base_ix, extra), base_ix, y_case,
                           fold_map, order, multiclass, classes)
            # L2: the local columns ALONE. Without this, an A2 gain cannot be told
            # apart from 30 extra columns happening to change the regularisation
            # geometry of a 320-column problem. Same shape as a pooling matrix so
            # the identical oof() path runs, but every pooling sees only the counts.
            only = {p: np.zeros((len(base_ix), 0), np.float32) for p in base_feats}
            preds["L2_" + k] = oof(attach(only, base_ix, extra), base_ix, y_case,
                                   fold_map, order, multiclass, classes)
        detail[field] = {}
        for k in ["A0"] + have + ["L2_" + h for h in have]:
            p, prob = preds[k]
            sc = score(yv, p, prob, multiclass, classes, rng)
            pair = paired(yv, p, p0, multiclass, rng) if k != "A0" else None
            v = verdict(pair, sc)
            if k == "A0":
                label = "A0  shipped whole-image anchor"
            elif k.startswith("L2_"):
                label = ("L2  local features ALONE from the %s detector (control: is a "
                         "gain really coming from the vessel counts?)" % RUNGS[k[3:]][0])
            else:
                label = RUNGS[k][1]
            # Section 6 of the brief asks for LOAO, calibration and post-abstention
            # coverage/accuracy on EVERY arm of EVERY field, not only on the ones
            # that end up looking good. They are computed here, before any verdict
            # is read, so the set of arms they exist for cannot depend on results.
            lo = loao(yv, p, folds_arr, arch_arr, multiclass,
                      None if k == "A0" else p0)
            rows.append(dict(field=field, rung=k, label=label, n=int(len(yv)),
                             unit_of_observation=unit,
                             observable_from_static_image=static_ok,
                             baseline_constant=sc["baseline_constant"],
                             accuracy=sc["accuracy"], delta=sc["delta"],
                             balanced_accuracy=sc["balanced_accuracy"],
                             auroc_macro_ovr=sc.get("auroc_macro_ovr"),
                             collapsed_to_one_class=sc["collapsed_to_one_class"],
                             ba_gain=None if pair is None else pair["ba_gain"],
                             ba_ci_excludes_zero=None if pair is None
                             else pair["ba_ci_excludes_zero"],
                             loao_directions_positive=lo.get("directions_positive"),
                             loao_directions_clearly_negative=lo.get(
                                 "directions_clearly_negative"),
                             verdict=v))
            detail[field][k] = dict(
                score=sc, paired=pair, verdict=v, loao=lo,
                calibration=calibration(yv, prob, multiclass, classes),
                abstention=abstention(yv, p, prob, multiclass, classes),
                folds=None if pair is None
                else fold_directions(yv, p, p0, folds_arr, multiclass))
        detail[field]["target_report"] = target_report

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ladder_local.csv", index=False, encoding="utf-8-sig")
    (OUT / "ladder_local.json").write_text(json.dumps(dict(
        fields_pre_declared=FIELDS, rungs_available=have,
        min_effect=MIN_EFFECT, below_resolution=BELOW_RESOLUTION,
        pca_survival={k: pca_survival(base_feats, base_ix, locals_[k]) for k in have},
        detail=detail,
        limitations=[
            "development case-level OOF; this is not a product capability claim",
            "locked-47 not read by this run",
            "counts, ratios and frame fractions only; no micron/per-mm/per-minute value",
            "only 50 of 186 cases have human boxes, so each detector trains on 35-44 "
            "cases; a null here is a null at that annotation volume",
        ]), ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
