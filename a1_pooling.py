"""A1: per-field pooling operator search under multiclass / ordinal targets.

Development-only (186 cases). locked-47 is asserted absent.
Selection of (pooling, feature-set, head) happens on the validation fold only;
the test fold is scored once per field as OOF.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp as _lse
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 20260913
np.random.seed(SEED)

ROOT = "/root/nailfold"
OUT = f"{ROOT}/artifacts/experiments/v10_A1_pooling_multiclass"
LABELS = f"{ROOT}/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
FEATDIRS = {
    "dinov2": f"{ROOT}/artifacts/features/dinov2",
    "geometry": f"{ROOT}/artifacts/features/geometry_dev",
    "roi_quality": f"{ROOT}/artifacts/features/roi_quality_dev_v1",
}
BAD = {"[未见]", "个/min", "条/mm", "<1"}
FIELDS = {
    "exudation": ({"无": 0, "+": 1, "++": 1, "+++": 2}, True),
    "clarity": ({"清晰": 0, "不清": 1, "模糊": 1}, False),
    "blood_color": ({"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1}, False),
    "subpapillary_venous_plexus": (
        {"不见": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 2}, True),
    "microthrombus": ({"无": 0, "1--2": 1, ">2": 2}, True),
    "papilla": ({"平坦": 0, "浅波纹状": 1, "波纹状": 2}, True),
    "capillary_count": ({">=7": 0, "5--6": 1, "3--4": 2}, True),
}
POOLS = ["mean", "median", "max", "q90", "topk3_mean", "logsumexp"]
FSETS = ["dinov2", "geometry", "roi_quality", "concat"]


def pool(X, mode):
    if mode == "mean":
        return X.mean(0)
    if mode == "median":
        return np.median(X, 0)
    if mode == "max":
        return X.max(0)
    if mode == "q90":
        return np.quantile(X, 0.9, axis=0)
    if mode == "topk3_mean":
        k = min(3, X.shape[0])
        return np.sort(X, 0)[-k:].mean(0)
    if mode == "logsumexp":
        s = X.std() + 1e-6
        return _lse(X / s, axis=0) * s - np.log(X.shape[0]) * s
    raise ValueError(mode)


class Ordinal:
    """Cumulative-link head: K-1 binary '>=k' logits, differenced into probs."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.models_ = []
        for t in range(1, self.k):
            yt = (y >= t).astype(int)
            if len(np.unique(yt)) < 2:
                self.models_.append(float(yt.mean()))
            else:
                m = mk_pipe()
                m.fit(X, yt)
                self.models_.append(m)
        return self

    def predict(self, X):
        n = X.shape[0]
        ge = np.ones((n, self.k))
        for t in range(1, self.k):
            m = self.models_[t - 1]
            ge[:, t] = (np.full(n, m) if isinstance(m, float)
                        else m.predict_proba(X)[:, 1])
        ge[:, 1:] = np.minimum.accumulate(ge[:, 1:], axis=1)
        p = np.zeros((n, self.k))
        for c in range(self.k):
            hi = ge[:, c]
            lo = ge[:, c + 1] if c + 1 < self.k else 0.0
            p[:, c] = hi - lo
        p = np.clip(p, 0, None)
        p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
        return p.argmax(1)


def mk_pipe():
    return Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=60)),
        ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                  max_iter=2000, random_state=SEED)),
    ])


def fit_predict(Xtr, ytr, Xte, head, k):
    if head == "ordinal":
        return Ordinal(k).fit(Xtr, ytr).predict(Xte)
    m = mk_pipe()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def scores(y, p, k, folds=None):
    acc = float((y == p).mean())
    out = {"accuracy": acc, "adj1": float((np.abs(y - p) <= 1).mean()),
           "qwk": float(cohen_kappa_score(y, p, weights="quadratic",
                                          labels=list(range(k))))}
    rec, rare = {}, {}
    for c in range(k):
        m = y == c
        if m.sum() == 0:
            continue
        r = float((p[m] == c).mean())
        (rec if m.sum() >= 10 else rare)[str(c)] = {
            "recall": r, "n": int(m.sum())}
    out["recall"] = rec
    out["rare_classes"] = rare
    return out


def main():
    lab = pd.read_csv(LABELS)
    locked = set(lab.loc[lab["development_fold"].isna(), "exam_case_id"])
    dev = lab[lab["development_fold"].notna()].copy()
    dev_cases = list(dev["exam_case_id"])
    assert len(dev_cases) == 186, len(dev_cases)
    assert len(locked) == 47
    assert not (set(dev_cases) & locked), "locked leakage"

    # frame features -> case-level pooled vectors
    idx = pd.read_csv(f"{FEATDIRS['dinov2']}/index.csv")
    raw = {}
    for name, d in FEATDIRS.items():
        f = np.load(f"{d}/features.npy").astype(np.float64)
        ix = pd.read_csv(f"{d}/index.csv")
        assert len(ix) == len(f) == 1708
        assert list(ix["exam_case_id"]) == list(idx["exam_case_id"])
        raw[name] = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    frame_cases = np.array(idx["exam_case_id"])
    assert not (set(frame_cases) & locked), "locked frames present"

    pooled = {}  # (fset, pool) -> (n_cases, d)
    order = {c: i for i, c in enumerate(dev_cases)}
    rows = {name: {} for name in FEATDIRS}
    for name in FEATDIRS:
        for p in POOLS:
            M = np.zeros((len(dev_cases), raw[name].shape[1]))
            for c, i in order.items():
                sel = frame_cases == c
                M[i] = pool(raw[name][sel], p) if sel.sum() else 0.0
            rows[name][p] = M
    for p in POOLS:
        for name in FEATDIRS:
            pooled[(name, p)] = rows[name][p]
        pooled[("concat", p)] = np.hstack([rows[n][p] for n in FEATDIRS])

    fold = dev["development_fold"].astype(int).to_numpy()
    results, report = {}, {}
    for field, (mapping, ordinal) in FIELDS.items():
        vals = dev[field].astype("object")
        st = dev[field + "__status"].astype(str)
        ok = (~st.str.contains("missing")) & (~vals.isin(BAD)) & vals.notna()
        y = np.array([mapping.get(v, -1) for v in vals])
        ok = ok.to_numpy() & (y >= 0)
        k = max(mapping.values()) + 1
        heads = ["multinomial"] + (["ordinal"] if ordinal else [])
        cfgs = [(fs, p, h) for fs in FSETS for p in POOLS for h in heads]

        oof = {c: np.full(len(y), -1) for c in cfgs}
        valacc = {c: [] for c in cfgs}
        for tf in range(5):
            vf = (tf + 1) % 5
            tr = ok & ~np.isin(fold, [tf, vf])
            va, te = ok & (fold == vf), ok & (fold == tf)
            for c in cfgs:
                fs, p, h = c
                X = pooled[(fs, p)]
                pv = fit_predict(X[tr], y[tr], X[va], h, k)
                valacc[c].append(float((y[va] == pv).mean()))
                trf = ok & (fold != tf)
                oof[c][te] = fit_predict(X[trf], y[trf], X[te], h, k)

        # per-fold mode baseline computed on that fold's training folds
        base = np.full(len(y), -1)
        for tf in range(5):
            trf, te = ok & (fold != tf), ok & (fold == tf)
            base[te] = np.bincount(y[trf], minlength=k).argmax()
        bacc = float((y[ok] == base[ok]).mean())

        def summarise(c):
            p = oof[c][ok]
            s = scores(y[ok], p, k)
            s["mode_baseline_acc"] = bacc
            s["delta"] = s["accuracy"] - bacc
            pf = []
            for tf in range(5):
                m = ok & (fold == tf)
                pf.append(float((y[m] == oof[c][m]).mean()
                                - (y[m] == base[m]).mean()))
            s["per_fold_delta"] = pf
            s["folds_positive"] = int(sum(d > 0 for d in pf))
            s["val_acc_mean"] = float(np.mean(valacc[c]))
            return s

        allres = {f"{fs}|{p}|{h}": summarise((fs, p, h)) for fs, p, h in cfgs}
        best = max(cfgs, key=lambda c: np.mean(valacc[c]))
        mainc = int(np.bincount(y[ok]).argmax())
        ref = allres[f"{best[0]}|mean|{best[2]}"]
        sel = allres[f"{best[0]}|{best[1]}|{best[2]}"]

        def mrec(s):
            d = s["recall"].get(str(mainc)) or s["rare_classes"].get(str(mainc))
            return d["recall"] if d else float("nan")

        drop = mrec(ref) - mrec(sel)
        passed = (sel["delta"] >= 0.05 and sel["folds_positive"] >= 3
                  and drop <= 0.02)
        results[field] = {
            "n_cases": int(ok.sum()), "n_classes": k, "ordinal": ordinal,
            "class_counts": np.bincount(y[ok], minlength=k).tolist(),
            "selected_config": "|".join(best), "selected": sel,
            "mean_pool_reference": ref, "main_class": mainc,
            "main_class_recall_drop_vs_mean": drop, "passed": bool(passed),
            "all_configs": allres,
        }
        report[field] = (best, sel, passed, drop)
        print(f"{field}: {'|'.join(best)} acc={sel['accuracy']:.3f} "
              f"base={bacc:.3f} delta={sel['delta']:+.4f} "
              f"folds+={sel['folds_positive']}/5 pass={passed}", flush=True)

    # sanity check: max/topk3 should help sparse-event fields, not global ones
    sanity = {}
    for field in FIELDS:
        a = results[field]["all_configs"]
        fs = results[field]["selected_config"].split("|")[0]
        h = results[field]["selected_config"].split("|")[2]
        mn = a[f"{fs}|mean|{h}"]["accuracy"]
        sanity[field] = {
            "mean": mn,
            "max_minus_mean": a[f"{fs}|max|{h}"]["accuracy"] - mn,
            "topk3_minus_mean": a[f"{fs}|topk3_mean|{h}"]["accuracy"] - mn,
        }
    glob = [f for f in ("clarity", "blood_color")
            if max(sanity[f]["max_minus_mean"],
                   sanity[f]["topk3_minus_mean"]) >= 0.05]
    sparse = [f for f in ("microthrombus", "exudation")
              if max(sanity[f]["max_minus_mean"],
                     sanity[f]["topk3_minus_mean"]) >= 0.05]
    sanity["global_fields_where_max_wins"] = glob
    sanity["sparse_fields_where_max_wins"] = sparse
    sanity["passed"] = len(glob) == 0

    out = {"locked_cases_seen": 0, "seed": SEED, "n_dev_cases": 186,
           "n_frames": 1708, "pooling_ops": POOLS, "feature_sets": FSETS,
           "gate": "delta>=0.05 AND folds_positive>=3 AND main-class recall "
                   "drop vs mean <=0.02",
           "sanity_check": sanity, "fields": results}
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("SANITY", json.dumps(sanity, ensure_ascii=False))
    print("PASSED FIELDS", [f for f in results if results[f]["passed"]])


if __name__ == "__main__":
    main()
