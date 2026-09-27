"""Measurement fields as cohort-relative binary outputs (偏大 / 偏小), scored the
way the product would ship them.

Why this target: the printed values sit on 3--5 ticks and the device ban forbids
micrometre claims, so the only honest product output is "above / below the
cohort median". The split point is the development-set median of the label,
declared here and never tuned.

Arms (all fixed, no per-field selection):
  A0     shipped DINOv2-B anchor, 5 poolings, PCA64, C=0.03
  G      detector geometry case features (external detector, 64 columns),
         StandardScaler + LogReg C=0.03
  A0+G   plain mean of the two probabilities (weight fixed at 0.5)
Readouts: development_fold OOF BA, both recalls, bootstrap CI, and
leave-one-archive-out (refit without the held-out archive).
locked-47 is never loaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_field_matrix_three_arms as R  # noqa: E402

GEOM = ROOT / "artifacts/experiments/rescue_external_20260922/local_features/external/case_features.csv"
OUT = ROOT / "artifacts/experiments/measurement_relative_20260927"
FIELDS = ["afferent_diameter", "efferent_diameter", "apex_diameter", "loop_length"]
C = 0.03
N_BOOT = 2000


def geom_fit(Xtr, ytr, Xts):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xts)[:, 1]


def a0_fit(feats, ix, y_case, train_ids, test_ids):
    trm = ix.exam_case_id.isin(train_ids).to_numpy()
    tem = ix.exam_case_id.isin(test_ids).to_numpy()
    ytr = y_case.reindex(ix.exam_case_id[trm]).to_numpy()
    ps = [R.fit_predict(feats[p][trm], ytr, feats[p][tem], False, [0.0, 1.0])[0]
          for p in R.POOLINGS]
    return pd.Series(np.mean(ps, axis=0),
                     index=ix.exam_case_id[tem].to_numpy()).groupby(level=0).mean()


def metrics(y, p, rng):
    h = (p >= 0.5).astype(float)
    ba = balanced_accuracy_score(y, h)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(balanced_accuracy_score(y[i], h[i]))
    return dict(ba=round(ba, 4), ba_ci=[round(float(np.percentile(bs, 2.5)), 4),
                                         round(float(np.percentile(bs, 97.5)), 4)],
                recall0=round(recall_score(y, h, pos_label=0), 4),
                recall1=round(recall_score(y, h, pos_label=1), 4),
                auroc=round(roc_auc_score(y, p), 4),
                pred_pos_share=round(float(h.mean()), 4))


def run_split(name, y_case, split, feats, ix, G):
    """split: Series case -> group id; each group is held out once."""
    ids = sorted(y_case.index)
    pa, pg = pd.Series(np.nan, index=ids), pd.Series(np.nan, index=ids)
    for g in sorted(split.unique()):
        te = [c for c in ids if split[c] == g]
        tr = [c for c in ids if split[c] != g]
        pa[te] = a0_fit(feats, ix, y_case, set(tr), set(te)).reindex(te).to_numpy()
        pg[te] = geom_fit(G.loc[tr].to_numpy(), y_case[tr].to_numpy(),
                          G.loc[te].to_numpy())
    return {"A0": pa, "G": pg, "A0+G": (pa + pg) / 2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(R.LABELS, dtype={"exam_case_id": str}).set_index("exam_case_id")
    dev = man[man.development_fold.notna()]
    assert len(dev) == 186
    locked = set(man.index[man.development_fold.isna()])
    ix, feats, _ = R.load_arm(R.ANCHOR)
    assert not (set(ix.exam_case_id) & locked)
    G = pd.read_csv(GEOM, dtype={"exam_case_id": str}).set_index("exam_case_id")
    assert not (set(G.index) & locked)
    G = G.fillna(G.median())
    rng = np.random.default_rng(R.SEED)
    rows = []
    for f in FIELDS:
        num = pd.to_numeric(R.clean_column(dev, f), errors="coerce").dropna()
        num = num[num.index.isin(G.index) & num.index.isin(set(ix.exam_case_id))]
        med = float(num.median())
        # ties at the median go to the lower class, declared
        y = (num > med).astype(float)
        for proto, split in [("dev_fold", dev.development_fold.reindex(y.index)),
                             ("loao", dev.archive.reindex(y.index))]:
            preds = run_split(proto, y, split, feats, ix, G)
            for arm, p in preds.items():
                m = metrics(y.to_numpy(), p.reindex(y.index).to_numpy(), rng)
                rows.append(dict(field=f, split_at_median=med, n=len(y),
                                 n_pos=int(y.sum()), protocol=proto, arm=arm, **m))
                print("%-18s %-8s %-5s ba %.3f %s r0 %.2f r1 %.2f auc %.3f"
                      % (f, proto, arm, m["ba"], m["ba_ci"], m["recall0"],
                         m["recall1"], m["auroc"]), flush=True)
            if proto == "loao":
                for arm, p in preds.items():
                    for a in sorted(dev.archive.unique()):
                        ii = [c for c in y.index if dev.archive[c] == a]
                        yy, pp = y[ii].to_numpy(), p[ii].to_numpy()
                        rows.append(dict(field=f, protocol="loao_per_archive",
                                         arm=arm, archive=a, n=len(ii),
                                         ba=round(balanced_accuracy_score(
                                             yy, (pp >= .5).astype(float)), 4),
                                         auroc=round(roc_auc_score(yy, pp), 4)))
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(
        run="measurement_relative", locked_cases_seen=0,
        target="label > development median (declared, not tuned)",
        arms=["A0", "G (detector geometry, external)", "A0+G mean 0.5"],
        note="relative to this cohort; not a micrometre output"), indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
