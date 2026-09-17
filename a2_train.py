"""A2: end-to-end DINOv2 LoRA + case-level gated attention pooling, 7-way multiclass.

Key difference from E5 (frozen-feature attention MIL): gradients flow through the
backbone's LoRA adapters, so attention weights and features are learned jointly under
case-level supervision. Backbone itself stays frozen (E7 showed unfreezing overfits).
"""
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

FIELDS = ["exudation", "clarity", "blood_color", "subpapillary_venous_plexus",
          "microthrombus", "papilla", "capillary_count"]
NCLS = {"exudation": 3, "clarity": 2, "blood_color": 2, "subpapillary_venous_plexus": 3,
        "microthrombus": 3, "papilla": 3, "capillary_count": 3}
MAP = {
    "exudation": {"无": 0, "[无]": 0, "+": 1, "++": 1, "+++": 2},
    "clarity": {"清晰": 0, "不清": 1, "模糊": 1},
    "blood_color": {"暗红": 0, "暗紫": 0, "浅红": 1, "淡红": 1, "[淡红色]": 1},
    "subpapillary_venous_plexus": {"不见": 0, "[不见]": 0, "可见1排": 1, "可见2排": 1, ">2排,扩张": 2},
    "microthrombus": {"无": 0, "[无]": 0, "[未见]": 0, "1--2": 1, ">2": 2},
    "papilla": {"平坦": 0, "浅波纹状": 1, "波纹状": 2, "[波纹状]": 2},
    "capillary_count": {">=7": 0, "5--6": 1, "3--4": 2, "<1": 2},
}


def encode(row, field):
    """Return class index or -1 for missing. Status must be high_consensus-family."""
    st = row.get(field + "__status")
    if not isinstance(st, str) or not st.startswith("high_consensus"):
        return -1
    v = row.get(field)
    if not isinstance(v, str):
        return -1
    return MAP[field].get(v.strip(), -1)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class Cases(Dataset):
    """One item = one case: a stack of its frames plus the 7 case-level labels."""

    def __init__(self, case_ids, frames_by_case, labels, root, tf, max_frames=None, rng=None):
        self.ids = list(case_ids)
        self.fbc = frames_by_case
        self.labels = labels
        self.root = root
        self.tf = tf
        self.max_frames = max_frames
        self.rng = rng

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        paths = self.fbc[cid]
        keep = list(range(len(paths)))
        if self.max_frames and len(paths) > self.max_frames:
            r = np.random.RandomState(self.rng + i) if self.rng is not None else np.random
            keep = sorted(r.choice(len(paths), self.max_frames, replace=False).tolist())
        imgs = torch.stack([self.tf(Image.open(self.root / paths[k]).convert("RGB")) for k in keep])
        y = torch.tensor(self.labels[cid], dtype=torch.long)
        return imgs, y, cid, torch.tensor(keep, dtype=torch.long)


def collate(batch):
    return batch[0]


class GatedAttention(nn.Module):
    def __init__(self, dim=768, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, z):
        s = self.w(torch.tanh(self.V(z)) * torch.sigmoid(self.U(z)))
        return torch.softmax(s.squeeze(-1), dim=0)


class A2Model(nn.Module):
    """Per-field independent pooling branch + multiclass head."""

    def __init__(self, backbone, pooling="attention"):
        super().__init__()
        self.b = backbone
        self.pooling = pooling
        if pooling == "attention":
            self.pool = nn.ModuleDict({f: GatedAttention() for f in FIELDS})
        self.heads = nn.ModuleDict({f: nn.Linear(768, NCLS[f]) for f in FIELDS})

    def forward(self, x):
        z = self.b(x)  # [F, 768]
        out, att = {}, {}
        for f in FIELDS:
            if self.pooling == "attention":
                a = self.pool[f](z)
            else:
                a = torch.full((z.shape[0],), 1.0 / z.shape[0], device=z.device, dtype=z.dtype)
            att[f] = a
            out[f] = self.heads[f](a.unsqueeze(0) @ z).squeeze(0)
        return out, att


def build_model(weights, pooling):
    b = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    missing = b.load_state_dict(sd, strict=False)
    assert len(missing.unexpected_keys) < 20, missing.unexpected_keys[:5]
    for p in b.parameters():
        p.requires_grad = False
    for block in b.blocks[-4:]:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=8, alpha=8)
        block.attn.proj = LoRALinear(block.attn.proj, rank=8, alpha=8)
    return A2Model(b, pooling).cuda()


@torch.inference_mode()
def predict(m, loader):
    m.eval()
    preds, atts = {}, {}
    for imgs, y, cid, keep in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, a = m(imgs.cuda(non_blocking=True))
        preds[cid] = {f: int(o[f].float().argmax().item()) for f in FIELDS}
        atts[cid] = {f: a[f].float().cpu().numpy() for f in FIELDS}
    return preds, atts


def class_weights(ytr, ncls):
    cnt = np.bincount(ytr, minlength=ncls).astype(float)
    w = np.where(cnt > 0, np.sqrt(len(ytr) / np.maximum(cnt, 1)), 0.0)
    present = cnt > 0
    if present.any():
        w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32).cuda()


def run_arm(args, pooling, seed, frames_by_case, labels, dev, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = {"input_size": (3, 518, 518), "interpolation": "bicubic",
           "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
           "crop_pct": 1.0, "crop_mode": "center"}
    trtf = timm.data.create_transform(**cfg, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    evtf = timm.data.create_transform(**cfg, is_training=False)
    fold_of = dict(zip(dev.exam_case_id, dev.development_fold.astype(int)))
    oof, oof_att, selections = {}, {}, []
    for test in range(5):
        val = (test + 1) % 5
        tr_ids = [c for c in labels if fold_of[c] not in (test, val)]
        va_ids = [c for c in labels if fold_of[c] == val]
        te_ids = [c for c in labels if fold_of[c] == test]
        m = build_model(args.weights, pooling)
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)
        cw = {}
        for i, f in enumerate(FIELDS):
            yv = np.array([labels[c][i] for c in tr_ids])
            cw[f] = class_weights(yv[yv >= 0], NCLS[f])
        tl = DataLoader(Cases(tr_ids, frames_by_case, labels, args.image_root, trtf,
                              args.max_frames, rng=seed), batch_size=1, shuffle=True,
                        num_workers=4, collate_fn=collate, pin_memory=True)
        vl = DataLoader(Cases(va_ids, frames_by_case, labels, args.image_root, evtf,
                              args.max_frames), batch_size=1, num_workers=4, collate_fn=collate)
        tel = DataLoader(Cases(te_ids, frames_by_case, labels, args.image_root, evtf,
                               args.max_frames), batch_size=1, num_workers=4, collate_fn=collate)
        best = (-1.0, None, 0)
        for epoch in range(1, args.epochs + 1):
            m.train()
            opt.zero_grad(set_to_none=True)
            for step, (imgs, y, cid, keep) in enumerate(tl):
                imgs = imgs.cuda(non_blocking=True)
                y = y.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o, _ = m(imgs)
                    ls = [nn.functional.cross_entropy(
                            o[f].unsqueeze(0), y[i].unsqueeze(0), weight=cw[f],
                            label_smoothing=0.04)
                          for i, f in enumerate(FIELDS) if y[i] >= 0]
                    loss = torch.stack(ls).mean() / args.accum
                loss.backward()
                if (step + 1) % args.accum == 0:
                    nn.utils.clip_grad_norm_(trainable, 2.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            vp, _ = predict(m, vl)
            accs = []
            for i, f in enumerate(FIELDS):
                yt = [labels[c][i] for c in va_ids if labels[c][i] >= 0]
                pr = [vp[c][f] for c in va_ids if labels[c][i] >= 0]
                accs.append(accuracy_score(yt, pr) if yt else 0.0)
            obj = float(np.mean(accs))
            log(f"[{pooling} seed{seed}] fold{test} ep{epoch} val_mean_acc={obj:.4f}")
            if obj > best[0]:
                best = (obj, copy.deepcopy(m.state_dict()), epoch)
        m.load_state_dict(best[1])
        tp, ta = predict(m, tel)
        oof.update(tp)
        oof_att.update(ta)
        selections.append({"fold": test, "val_fold": val, "epoch": best[2],
                           "val_mean_acc": best[0]})
        del m, opt
        torch.cuda.empty_cache()
    return oof, oof_att, selections


def evaluate(oof, labels, dev, fold_of):
    """Per-field metrics. mode_baseline comes from the 3 TRAINING folds of each test fold."""
    res = {}
    for i, f in enumerate(FIELDS):
        ids = [c for c in labels if labels[c][i] >= 0]
        y = np.array([labels[c][i] for c in ids])
        p = np.array([oof[c][f] for c in ids])
        # per-fold mode baseline (train folds only), aggregated to an OOF baseline vector
        base = np.empty_like(p)
        per_fold = []
        for test in range(5):
            val = (test + 1) % 5
            tr = [c for c in labels if labels[c][i] >= 0 and fold_of[c] not in (test, val)]
            mode = int(np.bincount([labels[c][i] for c in tr]).argmax())
            sel = np.array([fold_of[c] == test for c in ids])
            base[sel] = mode
            if sel.any():
                per_fold.append({"fold": test, "n": int(sel.sum()),
                                 "acc": float(accuracy_score(y[sel], p[sel])),
                                 "mode_acc": float(accuracy_score(y[sel], base[sel])),
                                 "delta": float(accuracy_score(y[sel], p[sel])
                                                - accuracy_score(y[sel], base[sel]))})
        cnt = np.bincount(y, minlength=NCLS[f])
        rec_all = recall_score(y, p, labels=list(range(NCLS[f])), average=None, zero_division=0)
        recalls = {str(c): float(rec_all[c]) for c in range(NCLS[f]) if cnt[c] >= 10}
        rare = {str(c): {"n": int(cnt[c]), "recall": float(rec_all[c])}
                for c in range(NCLS[f]) if 0 < cnt[c] < 10}
        acc = float(accuracy_score(y, p))
        macc = float(accuracy_score(y, base))
        adj1 = float(np.mean(np.abs(y - p) <= 1))
        try:
            qwk = float(cohen_kappa_score(y, p, weights="quadratic",
                                          labels=list(range(NCLS[f]))))
        except Exception:
            qwk = float("nan")
        folds_same = sum(1 for d in per_fold if d["delta"] >= 0.05)
        res[f] = {"n": len(ids), "n_classes": NCLS[f], "class_counts": cnt.tolist(),
                  "accuracy": acc, "mode_baseline_acc": macc, "delta": acc - macc,
                  "recall": recalls, "rare_classes": rare, "adj1": adj1, "QWK": qwk,
                  "per_fold_delta": per_fold,
                  "folds_delta_ge_0.05": folds_same,
                  "passes_gate_vs_mode": bool(acc - macc >= 0.05 and folds_same >= 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    logf = open(a.output_dir / "train.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    df = pd.read_csv(a.labels)
    df.exam_case_id = df.exam_case_id.astype(str)
    dev = df[df.development_fold.notna()].copy()
    locked = set(df.loc[df.development_fold.isna(), "exam_case_id"])
    assert len(dev) == 186, len(dev)
    assert len(locked) == 47, len(locked)

    idx = pd.read_csv(a.index)
    idx.exam_case_id = idx.exam_case_id.astype(str)
    idx = idx[idx.exam_case_id.isin(set(dev.exam_case_id))]
    assert not set(idx.exam_case_id) & locked, "LOCKED LEAKAGE"

    frames_by_case = idx.groupby("exam_case_id").image_path.apply(list).to_dict()
    labels = {}
    for _, r in dev.iterrows():
        cid = r.exam_case_id
        if cid not in frames_by_case:
            continue
        labels[cid] = [encode(r, f) for f in FIELDS]
    assert not set(labels) & locked, "LOCKED LEAKAGE"
    fold_of = dict(zip(dev.exam_case_id, dev.development_fold.astype(int)))
    log(f"cases={len(labels)} frames={len(idx)} locked_excluded={len(locked)}")
    for i, f in enumerate(FIELDS):
        c = np.bincount([labels[x][i] for x in labels if labels[x][i] >= 0],
                        minlength=NCLS[f])
        log(f"  {f}: n={int(c.sum())} counts={c.tolist()}")

    out = {"schema_version": "a2-e2e-case-attention/1.0",
           "evaluation_role": "development_oof", "locked_cases_seen": 0,
           "n_dev_cases": len(labels), "n_frames": int(len(idx)),
           "config": {"backbone": "vit_base_patch14_dinov2.lvd142m", "lora_rank": 8,
                      "lora_lr": 1e-4, "lora_targets": "last4.attn.qkv+attn.proj",
                      "backbone_frozen": True, "input_size": 518, "epochs": a.epochs,
                      "accum_cases": a.accum, "grad_clip": 2.0, "label_smoothing": 0.04,
                      "class_weight": "sqrt(N/count) normalized", "max_frames": a.max_frames,
                      "seeds": a.seeds},
           "arms": {}}
    att_store = {}
    for pooling in ["attention", "mean"]:
        per_seed = []
        for seed in a.seeds:
            t0 = time.time()
            oof, oof_att, sel = run_arm(a, pooling, seed, frames_by_case, labels, dev, log)
            res = evaluate(oof, labels, dev, fold_of)
            per_seed.append({"seed": seed, "selections": sel, "fields": res,
                             "minutes": (time.time() - t0) / 60})
            for cid, d in oof_att.items():
                for f, w in d.items():
                    att_store[f"{pooling}/seed{seed}/{cid}/{f}"] = w
            log(f"[{pooling} seed{seed}] " + " ".join(
                f"{f}:d={res[f]['delta']:+.3f}" for f in FIELDS))
            json.dump(out | {"partial": True}, open(a.output_dir / "metrics_partial.json", "w"),
                      ensure_ascii=False, indent=2, default=float)
        agg = {}
        for f in FIELDS:
            for k in ["accuracy", "delta", "adj1", "QWK", "mode_baseline_acc"]:
                v = [s["fields"][f][k] for s in per_seed]
                agg.setdefault(f, {})[k + "_mean"] = float(np.mean(v))
                agg[f][k + "_std"] = float(np.std(v))
            agg[f]["folds_delta_ge_0.05_mean"] = float(
                np.mean([s["fields"][f]["folds_delta_ge_0.05"] for s in per_seed]))
        out["arms"][pooling] = {"per_seed": per_seed, "aggregate": agg}

    gate = {}
    for f in FIELDS:
        at = out["arms"]["attention"]["aggregate"][f]
        mn = out["arms"]["mean"]["aggregate"][f]
        gate[f] = {"delta_attention": at["delta_mean"], "delta_mean_pool": mn["delta_mean"],
                   "attention_minus_mean": at["delta_mean"] - mn["delta_mean"],
                   "folds_ge_3": at["folds_delta_ge_0.05_mean"] >= 3,
                   "pass": bool(at["delta_mean"] >= 0.05
                                and at["folds_delta_ge_0.05_mean"] >= 3
                                and at["delta_mean"] > mn["delta_mean"])}
    npassed = sum(1 for f in FIELDS if gate[f]["pass"])
    out["gate"] = {"per_field": gate, "n_passed": npassed, "required": 4,
                   "overall_pass": npassed >= 4}
    json.dump(out, open(a.output_dir / "metrics.json", "w"), ensure_ascii=False,
              indent=2, default=float)
    np.savez_compressed(a.output_dir / "attention_weights.npz", **att_store)
    log(f"GATE n_passed={npassed}/7 overall={out['gate']['overall_pass']}")


if __name__ == "__main__":
    main()
