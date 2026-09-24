"""Stream A: the two controls the user authorised once the weights were granted.

Control 1  FIXED-HEAD DIRECT REPLACEMENT.  The shipped recipe is left exactly as
  deployed (five poolings, C=0.03, PCA 64, seed 20260917) and only the encoder is
  swapped.  This answers "can I drop this encoder into the delivered pipeline",
  and nothing else.

Control 2  SAME-BUDGET NESTED ADAPTATION.  C is selected inside the training
  folds only, with the SAME grid, the SAME inner protocol and the SAME number of
  fits for every arm including the anchor.  This answers "given an equal
  adaptation budget, does this encoder carry more signal".  A fixed-C null is not
  evidence against an encoder, which is why this control exists (explanation
  correction 2).  Nothing is ever selected on the outer test fold.

Five arms.  Two are pre-authorised Stream A candidates whose weights arrived this
round; the three others are the already-run arms, re-run here so both controls
share identical folds, identical cleaning and identical bootstrap draws.

The interpretable contrast differs per candidate and is stated per row:
  retfound_dinov2_meh  vs dinov2l_deployed_geometry -- capacity, patch size and
    geometry are all held fixed, so the continuation corpus is the only difference.
    RETFound is a CONTINUED DINOv2 run (patch_embed cosine 0.955 against the
    DINOv2-L this project uses, shuffle control -0.0017).  That establishes
    ancestry only: the same comparison shows relative Frobenius distance 0.495 on
    that tensor and attention cosine 0.165 by block 23, so the retinal SSL did
    retrain the network substantially and nothing here bounds the possible
    difference.  Read this row as "DINOv2 continued on retina vs DINOv2", not as
    "medical pretraining vs general pretraining".
  medsiglip_medical    NOT a pure pretraining contrast: its position embedding is
    a fixed 32x32 at 448, so it cannot run at the deployed 518x686.  Geometry and
    architecture differ too, and a difference cannot be attributed.

DEVELOPMENT ONLY.  locked-47 is never loaded; the run asserts locked_cases_seen=0.
RETFound is CC BY-NC 4.0: a research-period control that cannot ship.

  PYTHONIOENCODING=utf-8 python scripts/run_stream_a_medical_controls.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_field_matrix import ADMISSIBLE, IN_SCOPE, NUMERIC_FIELDS, UNIT  # noqa: E402
from run_field_matrix_three_arms import (C_FIXED, DIM_FIXED, EXP, LABELS,  # noqa: E402
                                         N_BOOT, POOLINGS, SEED, fit_predict,
                                         fold_directions, hard, load_arm, oof,
                                         paired, score, severe_cross_band, target)

OUT = EXP / "stream_a_controls"
ANCHOR = "anchor_dinov2b_deployed"
ARMS = {
    "anchor_dinov2b_deployed": "A  general vision ViT-B/14, deployed geometry",
    "dinov2l_deployed_geometry": "A  general vision ViT-L/14, deployed geometry",
    "biomedclip_medical": "B  medical pretraining, BiomedCLIP ViT-B/16 @224",
    "retfound_dinov2_meh": "B  medical pretraining, RETFound-DINOv2-MEH ViT-L/14 @deployed",
    "medsiglip_medical": "B  medical pretraining, MedSigLIP vision tower @448",
}
# matched reference per candidate: what the arm should be read against
REFERENCE = {"dinov2l_deployed_geometry": ANCHOR,
             "biomedclip_medical": ANCHOR,
             "retfound_dinov2_meh": "dinov2l_deployed_geometry",
             "medsiglip_medical": ANCHOR}
ATTRIBUTION = {
    "dinov2l_deployed_geometry":
        "capacity contrast: same pretraining family, larger encoder",
    "biomedclip_medical":
        "not a pure pretraining contrast: patch 16 at 224, geometry differs",
    "retfound_dinov2_meh":
        "PURE pretraining contrast against dinov2l (same capacity, patch and "
        "geometry), but bounded: continued DINOv2 run, patch_embed cosine 0.955",
    "medsiglip_medical":
        "not a pure pretraining contrast: fixed 32x32 position embedding forces "
        "448 square, so geometry and architecture differ as well",
}
# one grid, one inner protocol, identical for every arm -> equal budget
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def _pipe_parts(Xtr, seed=SEED):
    dim = max(2, min(DIM_FIXED, Xtr.shape[0] - 1, Xtr.shape[1]))
    sc = StandardScaler().fit(Xtr)
    pc = PCA(n_components=dim, random_state=seed).fit(sc.transform(Xtr))
    return sc, pc


def _fit_C(Ztr, ytr, Zts, C, multiclass, classes):
    """Same head as the shipped one, but C supplied by the caller."""
    if len(set(ytr)) < 2:
        lab = float(ytr[0])
        proba = np.zeros((len(Zts), len(classes)))
        proba[:, classes.index(lab)] = 1.0
        return np.full(len(Zts), lab), proba
    lr = LogisticRegression(C=C, max_iter=4000).fit(Ztr, ytr)
    pr = lr.predict_proba(Zts)
    full = np.zeros((len(Zts), len(classes)))
    for j, c in enumerate(lr.classes_):
        full[:, classes.index(float(c))] = pr[:, j]
    return (lr.predict(Zts).astype(float) if multiclass else pr[:, 1]), full


def oof_nested(feats, ix, y_case, folds, order, multiclass, classes):
    """Outer OOF with C chosen by inner leave-one-training-fold-out.

    The inner loop sees ONLY training folds.  The chosen C is recorded per outer
    fold so a gain that depends on one lucky selection is visible.  Aggregation
    over poolings is the shipped rule, unchanged.
    """
    dm = ix.exam_case_id.isin(y_case.index).to_numpy()
    pos = {c: i for i, c in enumerate(order)}
    out = np.full(len(order), np.nan)
    prob = np.full((len(order), len(classes)), np.nan)
    fs = pd.Series([folds.get(c, np.nan) for c in order], index=order)
    chosen = {}
    for fo in sorted(fs.dropna().unique()):
        te = set(fs.index[fs == fo])
        tr_ids = set(order) - te
        trm = dm & ix.exam_case_id.isin(tr_ids).to_numpy()
        tem = dm & ix.exam_case_id.isin(te).to_numpy()
        if trm.sum() == 0 or tem.sum() == 0:
            continue
        inner = sorted(v for v in fs.dropna().unique() if v != fo)
        per_pool = {}
        for p in POOLINGS:
            tot = {C: [] for C in C_GRID}
            for iv in inner:
                ite = set(fs.index[fs == iv]) & tr_ids
                itr = tr_ids - ite
                im_tr = dm & ix.exam_case_id.isin(itr).to_numpy()
                im_te = dm & ix.exam_case_id.isin(ite).to_numpy()
                if im_tr.sum() == 0 or im_te.sum() == 0:
                    continue
                ytr_i = y_case.reindex(ix.exam_case_id[im_tr]).to_numpy()
                ids_i = ix.exam_case_id[im_te].to_numpy()
                yte_i = y_case.reindex(pd.unique(ids_i)).to_numpy()
                sc, pc = _pipe_parts(feats[p][im_tr])
                Ztr = pc.transform(sc.transform(feats[p][im_tr]))
                Zts = pc.transform(sc.transform(feats[p][im_te]))
                for C in C_GRID:
                    pi, _ = _fit_C(Ztr, ytr_i, Zts, C, multiclass, classes)
                    s = pd.Series(pi, index=ids_i).groupby(level=0)
                    agg = (s.agg(lambda g: g.value_counts().index[0]) if multiclass
                           else s.mean()).reindex(pd.unique(ids_i)).to_numpy()
                    if len(np.unique(yte_i)) < 2:
                        continue
                    tot[C].append(balanced_accuracy_score(
                        yte_i, hard(agg, multiclass)))
            means = {C: (float(np.mean(v)) if v else -1.0) for C, v in tot.items()}
            # ties resolved towards the deployed C, then towards stronger
            # regularisation, so the tie-break cannot favour the candidate
            best = max(means.values())
            cands = [C for C in C_GRID if means[C] >= best - 1e-12]
            per_pool[p] = C_FIXED if C_FIXED in cands else min(cands)
        chosen[str(fo)] = per_pool
        ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
        ids = ix.exam_case_id[tem].to_numpy()
        got = []
        for p in POOLINGS:
            sc, pc = _pipe_parts(feats[p][trm])
            Ztr = pc.transform(sc.transform(feats[p][trm]))
            Zts = pc.transform(sc.transform(feats[p][tem]))
            got.append(_fit_C(Ztr, ytr, Zts, per_pool[p], multiclass, classes))
        pm = pd.DataFrame(np.mean([g[1] for g in got], axis=0),
                          index=ids).groupby(level=0).mean()
        for c, row in pm.iterrows():
            if c in pos:
                prob[pos[c]] = row.to_numpy()
        if multiclass:
            df = pd.DataFrame(np.stack([g[0] for g in got]).T, index=ids)
            for c, grp in df.groupby(level=0):
                vals, cnt = np.unique(grp.to_numpy().ravel(), return_counts=True)
                if c in pos:
                    out[pos[c]] = vals[cnt.argmax()]
        else:
            s = pd.Series(np.mean([g[0] for g in got], axis=0),
                          index=ids).groupby(level=0).mean()
            for c, v in s.items():
                if c in pos:
                    out[pos[c]] = v
    return out, prob, chosen


def build_rows(readout, predictor, dev, arms, folds, arch, rng):
    rows, chosen_rows = [], []
    for field in IN_SCOPE:
        y_case, k, extra = target(dev, field)
        order = sorted(y_case.index)
        y = y_case.reindex(order).to_numpy()
        classes = sorted(set(float(v) for v in y))
        mc = k > 2
        ok = np.ones(len(order), bool)
        preds, picks = {}, {}
        for a, (ix, f) in arms.items():
            got = predictor(f, ix, y_case, folds, order, mc, classes)
            preds[a] = (got[0], got[1])
            if len(got) > 2:
                picks[a] = got[2]
            ok &= ~np.isnan(got[0])
        unit, static_ok = UNIT[field]
        base = dict(field=field, readout=readout, n_classes=k, multiclass=mc,
                    target=ADMISSIBLE[field][0], unit_of_observation=unit,
                    observable_from_static_image=static_ok)
        if static_ok is False and field in ("flow_state", "microthrombus"):
            base["reading"] = ("STATIC APPEARANCE CORRELATION, NOT FLOW/EVENT "
                               "OBSERVATION")
        if field in NUMERIC_FIELDS:
            base["measurement_note"] = ("banded by the printed normal range; NOT "
                                        "a micrometre measurement capability")
            base.update(extra)
        for a in ARMS:
            p, pr = preds[a]
            r = dict(base, arm=a, arm_role=ARMS[a], **score(
                y[ok], p[ok], pr[ok], mc, classes, rng))
            if field in NUMERIC_FIELDS or field == "capillary_count":
                r["severe_cross_band_error_rate"] = severe_cross_band(y[ok], p[ok], mc)
            if a != ANCHOR:
                ref = REFERENCE[a]
                fa = np.array([folds[c] for c in order])[ok]
                r["reference_arm"] = ref
                r["attribution"] = ATTRIBUTION[a]
                r["paired_vs_reference"] = paired(
                    y[ok], p[ok], preds[ref][0][ok], mc, rng)
                r["fold_directions_vs_reference"] = fold_directions(
                    y[ok], p[ok], preds[ref][0][ok], fa, mc)
                if ref != ANCHOR:
                    r["paired_vs_anchor"] = paired(
                        y[ok], p[ok], preds[ANCHOR][0][ok], mc, rng)
            rows.append(r)
            if a in picks:
                for fo, pp in picks[a].items():
                    chosen_rows.append(dict(field=field, arm=a, outer_fold=fo,
                                            **{("C_" + k2): v
                                               for k2, v in pp.items()}))
        print("  %-26s %-7s n=%3d k=%d base %.3f | " % (
            field, readout, rows[-1]["n"], k, rows[-1]["baseline_constant"])
            + " ".join("%s d%+.3f" % (a.split("_")[0][:6],
                                      [r for r in rows if r["field"] == field
                                       and r["readout"] == readout
                                       and r["arm"] == a][0]["delta"])
                       for a in ARMS), flush=True)
    return rows, chosen_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    locked = set(man.index[man.development_fold.isna()])
    if len(dev) != 186 or len(locked) != 47:
        raise RuntimeError("expected 186 development / 47 locked, got %d / %d"
                           % (len(dev), len(locked)))
    folds = dev.development_fold.to_dict()
    arch = dev.archive

    arms, metas = {}, {}
    for a in ARMS:
        ix, feats, meta = load_arm(a)
        if set(ix.exam_case_id) & locked:
            raise RuntimeError("locked case present in arm %s" % a)
        if ix.image_path.astype(str).str.contains(r"rep[_0-9]").any():
            raise RuntimeError("report scan reached arm %s" % a)
        if ix.exam_case_id.nunique() != 186:
            raise RuntimeError("arm %s covers %d cases" % (a, ix.exam_case_id.nunique()))
        arms[a], metas[a] = (ix, feats), meta
    # every arm must share one index order, otherwise the folds are not identical
    ref_ix = arms[ANCHOR][0]
    for a in ARMS:
        if not arms[a][0].image_path.equals(ref_ix.image_path):
            raise RuntimeError("arm %s has a different image order" % a)

    t0 = time.time()
    print("control 1: fixed head, C=%.2f (direct replacement)" % C_FIXED, flush=True)
    fixed_rows, _ = build_rows("fixed_C0.03", oof, dev, arms, folds, arch,
                               np.random.default_rng(SEED))
    print("control 2: same-budget nested C selection inside training folds",
          flush=True)
    nested_rows, chosen = build_rows("nested_C_in_training_folds", oof_nested,
                                     dev, arms, folds, arch,
                                     np.random.default_rng(SEED))
    rows = fixed_rows + nested_rows

    out = dict(
        run="stream_a_medical_encoder_controls",
        question="with the two newly authorised medical encoders available, does "
                 "either beat its matched reference (a) as a direct drop-in to "
                 "the deployed fixed head, or (b) under an equal adaptation "
                 "budget with C selected inside training folds only",
        controls=dict(
            fixed_C0_03=dict(
                meaning="direct replacement into the deployed recipe; evaluates "
                        "swapping the encoder and nothing else",
                C=C_FIXED, selection="none"),
            nested_C_in_training_folds=dict(
                meaning="equal adaptation budget; C chosen by inner "
                        "leave-one-training-fold-out on training folds only",
                grid=C_GRID,
                identical_budget_for_every_arm=True,
                tie_break="towards the deployed C=0.03, then stronger "
                          "regularisation, so ties cannot favour a candidate",
                outer_test_never_used_for_selection=True)),
        shared=dict(poolings=POOLINGS, pca_dim=DIM_FIXED, seed=SEED,
                    n_bootstrap=N_BOOT,
                    aggregation="mean of pooling probabilities; majority vote "
                                "over poolings for multiclass",
                    split="development_fold, case-level OOF, identical across "
                          "arms and across both controls",
                    per_field_arm_selection="none"),
        arms={a: dict(role=ARMS[a], reference=REFERENCE.get(a),
                      attribution=ATTRIBUTION.get(a, "anchor"),
                      input_size=metas[a].get("input_size"),
                      patch=metas[a].get("patch"),
                      patch_grid=metas[a].get("patch_grid"),
                      embedding_dim=metas[a].get("feature_dim"),
                      geometry=metas[a].get("geometry"),
                      pretraining=metas[a].get("pretraining"),
                      weights_sha256=metas[a].get("weights_sha256"),
                      licence=metas[a].get("licence"),
                      limitations=metas[a].get("limitations")) for a in ARMS},
        locked_cases_seen=0,
        locked_note="locked-47 was not loaded in this run; development OOF only "
                    "and not launch capability",
        licence_note="RETFound is CC BY-NC 4.0 and cannot ship in a product; it "
                     "is a research-period control only",
        runtime_seconds=round(time.time() - t0, 1),
        fields=rows)
    (OUT / "stream_a_controls.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                  for r in rows]).to_csv(OUT / "stream_a_controls.csv",
                                         index=False, encoding="utf-8-sig")
    pd.DataFrame(chosen).to_csv(OUT / "nested_C_choices.csv", index=False,
                                encoding="utf-8-sig")
    print("\nwrote %s (%d rows, %.1f min)"
          % (OUT / "stream_a_controls.json", len(rows), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
