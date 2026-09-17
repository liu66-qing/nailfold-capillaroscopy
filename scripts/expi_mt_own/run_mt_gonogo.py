"""exp_i: is `microthrombus` learnable from static images? (go/no-go)

Protocol (mandated):
  - manifest locked_evaluation_v1_reviewed.csv, dtype exam_case_id=str,
    ONLY evaluation_role == 'development'.  locked_test is NEVER read.
  - CV uses the existing development_fold column: test=fold, val=(fold+1)%5, train=rest.
  - conflicted cases recovered_archive2/180, recovered_archive3/263 excluded.
  - frozen DINOv2-base features precomputed; per-frame = CLS(768) + GatedPool(patch)(768).
  - class weights from the TRAINING FOLD ONLY.

Arms: static_cls (mean-pool frames), static_maxpool (max-pool frames), majority.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MANIFEST = "/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv"
RULES = "/root/nailfold/artifacts/labels/score_rules_v3.json"
FEATDIR = "/root/autodl-tmp/nailfold/exp_i_feats"
OUTDIR = "/root/nailfold/artifacts/experiments/exp_i_microthrombus"
CONFLICTED = ["recovered_archive2/180", "recovered_archive3/263"]
CLASSES = ["无", "1--2", ">2"]
SEEDS = [17, 42, 123]
JUNK_TO_NO = {"[未见]": "无", "个/min": "无"}


def load_cohort():
    man = pd.read_csv(MANIFEST, dtype={"exam_case_id": str})
    man = man[man.evaluation_role == "development"].copy()
    man = man[~man.exam_case_id.isin(CONFLICTED)]
    man["mt"] = man["microthrombus"].replace(JUNK_TO_NO)
    man = man[man["mt"].isin(CLASSES)]
    man = man[man["development_fold"].notna()]
    man["y"] = man["mt"].map({c: i for i, c in enumerate(CLASSES)})
    man["fold"] = man["development_fold"].astype(int)
    return man[["exam_case_id", "mt", "y", "fold"]].reset_index(drop=True)


class Feats:
    """Frozen features pinned on GPU in bf16 (no per-step PCIe traffic)."""

    def __init__(self, cases, dev):
        self.cache = {}
        nb = 0
        for c in cases:
            z = np.load(os.path.join(FEATDIR, c.replace("/", "__") + ".npz"))
            cls = torch.from_numpy(z["cls"]).to(dev, dtype=torch.bfloat16)
            patch = torch.from_numpy(z["patch"]).to(dev, dtype=torch.bfloat16)
            nb += (cls.numel() + patch.numel()) * 2
            self.cache[c] = (cls, patch)
        print("   feature store %.2f GB on %s" % (nb / 1e9, dev), flush=True)

    def get(self, c):
        return self.cache[c]


class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, patch):
        a = self.w(torch.tanh(self.v(patch)) * torch.sigmoid(self.u(patch)))
        return (torch.softmax(a, dim=1) * patch).sum(dim=1)


class Net(nn.Module):
    def __init__(self, agg="mean", dim=768, ncls=3, p=0.3):
        super().__init__()
        self.agg = agg
        self.pool = GatedPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Dropout(p),
            nn.Linear(2 * dim, 256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, ncls),
        )

    def forward(self, cls, patch):
        fr = torch.cat([cls, self.pool(patch)], dim=-1)
        case = fr.max(dim=0).values if self.agg == "max" else fr.mean(dim=0)
        return self.head(case.unsqueeze(0))


def confusion(y, p, k=3):
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def balanced_acc(y, p, k=3):
    r = [((p == c) & (y == c)).sum() / (y == c).sum() for c in range(k) if (y == c).sum()]
    return float(np.mean(r)) if r else float("nan")


def per_class_recall(y, p, k=3):
    out = {}
    for c in range(k):
        n = int((y == c).sum())
        out[CLASSES[c]] = float(((p == c) & (y == c)).sum() / n) if n else None
    return out


def macro_f1(y, p, k=3):
    fs = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        if tp + fp + fn:
            fs.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs)) if fs else float("nan")


def bootstrap_ba(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) >= 2:
            v.append(balanced_acc(y[i], p[i]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def score_mae(y, p, mp):
    sy = np.array([mp[CLASSES[c]] for c in y])
    sp = np.array([mp[CLASSES[c]] for c in p])
    return float(np.abs(sy - sp).mean())


def summarize(y, p, mp, tag, extra=None):
    lo, hi = bootstrap_ba(y, p)
    maj = int(np.bincount(y, minlength=3).argmax())
    d = {
        "arm": tag, "n": int(len(y)),
        "balanced_accuracy": balanced_acc(y, p), "ba_ci95": [lo, hi],
        "ba_ci95_excludes_random_third": bool(lo > 1.0 / 3.0),
        "macro_f1": macro_f1(y, p), "accuracy": float((y == p).mean()),
        "majority_class": CLASSES[maj], "majority_rate": float((y == maj).mean()),
        "per_class_recall": per_class_recall(y, p),
        "confusion_matrix": confusion(y, p).tolist(),
        "confusion_rows_true_cols_pred": CLASSES,
        "field_score_mae": score_mae(y, p, mp),
    }
    if extra:
        d.update(extra)
    return d


def run_fold(feats, tr, va, te, agg, seed, dev, epochs=60, lr=3e-4, patience=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(agg=agg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    cnt = np.bincount(tr["y"].values, minlength=3).astype(float)
    w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / 3.0, 0.0)
    cw = torch.tensor(w, dtype=torch.float32, device=dev)

    def ev(df):
        net.eval()
        ys, ps, pr = [], [], []
        with torch.no_grad():
            for _, r in df.iterrows():
                c, pt = feats.get(r["exam_case_id"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = net(c, pt)
                q = torch.softmax(lg.float(), -1).cpu().numpy()[0]
                pr.append(q)
                ps.append(int(q.argmax()))
                ys.append(int(r["y"]))
        return np.array(ys), np.array(ps), np.array(pr)

    best, state, bad = -1.0, None, 0
    order = np.arange(len(tr))
    for ep in range(epochs):
        net.train()
        np.random.shuffle(order)
        tot = 0.0
        opt.zero_grad()
        for k, i in enumerate(order):
            r = tr.iloc[i]
            c, pt = feats.get(r["exam_case_id"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(c, pt)
            loss = F.cross_entropy(lg.float(), torch.tensor([int(r["y"])], device=dev), weight=cw)
            (loss / 8).backward()
            tot += float(loss)
            if (k + 1) % 8 == 0 or k == len(order) - 1:
                opt.step()
                opt.zero_grad()
        yv, pv, _ = ev(va)
        vba = balanced_acc(yv, pv)
        if vba > best:
            best, bad = vba, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 15 == 0:
            print("      ep%02d loss=%.4f val_ba=%.4f best=%.4f" % (ep, tot / len(order), vba, best), flush=True)
        if bad >= patience:
            print("      early stop ep%d best_val_ba=%.4f" % (ep, best), flush=True)
            break
    if state:
        net.load_state_dict(state)
    return ev(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda:%d" % a.gpu
    os.makedirs(OUTDIR, exist_ok=True)

    coh = load_cohort()
    mp = json.load(open(RULES))["field_rules"]["microthrombus"]["mapping"]
    print("cohort n=%d" % len(coh), flush=True)
    print(coh["mt"].value_counts().to_string(), flush=True)
    print("folds:", coh["fold"].value_counts().sort_index().to_dict(), flush=True)
    feats = Feats(coh["exam_case_id"].tolist(), dev)

    y_all = coh["y"].values
    res = {}
    maj = int(np.bincount(y_all, minlength=3).argmax())
    res["majority"] = summarize(y_all, np.full_like(y_all, maj), mp, "majority")

    oof = []
    for agg, tag in (("mean", "static_cls"), ("max", "static_maxpool")):
        print("=" * 60, flush=True)
        print("ARM", tag, flush=True)
        seeds_s, acc = [], np.zeros((len(coh), 3))
        for sd in SEEDS:
            op = np.zeros(len(coh), dtype=int)
            opr = np.zeros((len(coh), 3))
            for f in sorted(coh["fold"].unique()):
                vf = (f + 1) % 5
                te, va = coh[coh.fold == f], coh[coh.fold == vf]
                tr = coh[~coh.fold.isin([f, vf])]
                print("   seed%d fold%d tr=%d va=%d te=%d" % (sd, f, len(tr), len(va), len(te)), flush=True)
                _, pt, prt = run_fold(feats, tr, va, te, agg, sd, dev)
                pos = coh.index[coh.fold == f]
                op[pos], opr[pos] = pt, prt
            s = summarize(y_all, op, mp, "%s_seed%d" % (tag, sd))
            print("   -> seed%d BA=%.4f acc=%.4f" % (sd, s["balanced_accuracy"], s["accuracy"]), flush=True)
            seeds_s.append(s)
            acc += opr
        ens = acc.argmax(1)
        bas = [s["balanced_accuracy"] for s in seeds_s]
        summ = summarize(y_all, ens, mp, tag, extra={
            "per_seed_ba": bas,
            "ba_mean_over_seeds": float(np.mean(bas)),
            "ba_std_over_seeds": float(np.std(bas)),
            "seed_details": seeds_s,
            "note": "headline BA is 3-seed softmax-ensemble OOF",
        })
        res[tag] = summ
        print("   ENSEMBLE BA=%.4f CI=%s" % (summ["balanced_accuracy"], summ["ba_ci95"]), flush=True)
        d = coh.copy()
        d["case"] = d["exam_case_id"]
        d["pred_class_idx"] = ens
        d["pred_class"] = [CLASSES[i] for i in ens]
        for j, c in enumerate(CLASSES):
            d["prob_%s" % c] = acc[:, j] / len(SEEDS)
        d["arm"] = tag
        oof.append(d)

    pd.concat(oof).to_csv(os.path.join(OUTDIR, "oof_predictions.csv"), index=False)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as fh:
        json.dump({
            "experiment": "exp_i_microthrombus",
            "question": "is microthrombus learnable from static images",
            "cohort_n": int(len(coh)),
            "class_counts": coh["mt"].value_counts().to_dict(),
            "junk_handling": "[未见] and 个/min -> 无, justified by flow_score additivity",
            "excluded_conflicted": CONFLICTED,
            "manifest": MANIFEST, "role_filter": "development",
            "seeds": SEEDS, "score_mapping": mp, "results": res,
        }, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUTDIR, flush=True)
    for k, v in res.items():
        print("%-16s BA=%.4f CI=[%.3f,%.3f] MAE=%.3f acc=%.3f" % (
            k, v["balanced_accuracy"], v["ba_ci95"][0], v["ba_ci95"][1],
            v["field_score_mae"], v["accuracy"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())

