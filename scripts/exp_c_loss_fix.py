"""EXP-C: is the EXP-10 regression collapse caused by loss-scale imbalance?

EXP-10 combined 0.5*MSE((pred - ts/20)^2) with 0.5*weighted_CE. The MSE term
sits at ~1e-2 while the CE term sits at ~1.6, so the shared trunk was driven
almost entirely by the classification head and the score head decayed toward
the mean (pred_std 2.11 vs true_std 3.60).

Arms (identical data, folds, seeds, backbone; only the objective differs):
  A  reg_only_raw   single-task regression, target = ts/20      (EXP-10's scale)
  B  reg_only_znorm single-task regression, target z-scored on the TRAIN fold
  C  multi_balanced both heads, but regression on z-scored target so the two
                    loss terms are comparable in magnitude

Falsifiable prediction: if imbalance is the cause, B/C recover pred_std toward
true_std and improve Spearman; if the image truly carries no ordinal signal,
all three stay near the constant predictor.

Protocol notes (deliberate):
  * Report label `overall_assessment` is authoritative; the 4 conflicted cases
    from EXP-A are EXCLUDED, not relabelled.
  * z-score statistics are computed on the TRAIN fold only (no leakage).
  * Level metrics use the inclusive '<=' cut, applied only to PREDICTIONS.
  * locked_test is never touched here.
"""
import argparse, json, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

SEEDS = [17, 42, 123, 456, 789]
SEV = ["正常", "大致正常", "轻度异常", "中度异常", "重度异常"]
SEV_MAP = {s: i for i, s in enumerate(SEV)}
CUTS = [1.0, 2.0, 4.0, 8.0]

# From EXP-A: report level and total_score disagree irreducibly on these.
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263",
              "recovered_archive2/182", "recovered_archive3/244"}

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
FRAMES = '/root/nailfold/artifacts/features/image_index.csv'
ROOT = '/root/nailfold/data'
WEIGHTS = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'


def cut_inclusive(s):
    for t, lab in zip(CUTS, SEV[:-1]):
        if s <= t:
            return lab
    return SEV[-1]


class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), dim=1)
        return (a * x).sum(1)


class Net(nn.Module):
    def __init__(self, backbone, with_cls, bottleneck=64):
        super().__init__()
        self.b = backbone
        self.patch_pool = GatedPool()
        self.trunk = nn.Sequential(nn.Linear(1536, bottleneck), nn.ReLU(), nn.Dropout(0.3))
        self.score_head = nn.Linear(bottleneck, 1)
        self.assess_head = nn.Linear(bottleneck, 5) if with_cls else None

    def forward(self, x):
        z = self.b.forward_features(x)
        z = torch.cat([z[:, 0], self.patch_pool(z[:, 1:])], 1)
        z = self.trunk(z)
        s = self.score_head(z).squeeze(-1)
        a = self.assess_head(z) if self.assess_head is not None else None
        return s, a


class Cases(Dataset):
    def __init__(self, ids, frames, labels, transform):
        self.ids = list(ids)
        sub = frames[frames.exam_case_id.isin(set(ids))]
        self.groups = {c: g for c, g in sub.groupby('exam_case_id')}
        self.labels = labels.set_index('exam_case_id')
        self.transform = transform
        self.root = Path(ROOT)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        rows = self.groups[cid]
        imgs = torch.stack([self.transform(Image.open(self.root / r.image_path).convert('RGB'))
                            for _, r in rows.iterrows()])
        row = self.labels.loc[cid]
        return imgs, float(row['ts']), int(row['sev']), cid


def collate(b):
    return ([x[0] for x in b], torch.tensor([x[1] for x in b], dtype=torch.float32),
            torch.tensor([x[2] for x in b]), [x[3] for x in b])


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build(device, with_cls):
    b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    b.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=False)
    for p in b.parameters():
        p.requires_grad = False
    return Net(b, with_cls).to(device)


ARMS = {
    'A_reg_only_raw':   dict(with_cls=False, znorm=False),
    'B_reg_only_znorm': dict(with_cls=False, znorm=True),
    'C_multi_balanced': dict(with_cls=True,  znorm=True),
}


def run_arm(arm, cfg, dev, frames, tr, ev, device, epochs, oa_w):
    import time
    oof = {}
    t_start = time.time()
    for seed in SEEDS:
        seed_all(seed)
        for fold in range(5):
            vf = (fold + 1) % 5
            tr_ids = dev.loc[~dev.development_fold.isin([fold, vf]), 'exam_case_id'].tolist()
            va_ids = dev.loc[dev.development_fold.eq(vf), 'exam_case_id'].tolist()
            te_ids = dev.loc[dev.development_fold.eq(fold), 'exam_case_id'].tolist()

            # target scaling fitted on TRAIN fold only
            tr_ts = dev.loc[dev.exam_case_id.isin(tr_ids), 'ts'].values
            if cfg['znorm']:
                mu, sd = float(tr_ts.mean()), float(tr_ts.std() + 1e-6)
            else:
                mu, sd = 0.0, 20.0

            dl = lambda ids, t, sh: DataLoader(Cases(ids, frames, dev, t), batch_size=4,
                                              shuffle=sh, collate_fn=collate, num_workers=2)
            train_dl, val_dl, test_dl = dl(tr_ids, tr, True), dl(va_ids, ev, False), dl(te_ids, ev, False)

            model = build(device, cfg['with_cls'])
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, weight_decay=0.05)
            sched = CosineAnnealingLR(opt, T_max=epochs)
            best = (1e9, None, 0)

            for ep in range(epochs):
                model.train()
                for imgs_l, ts_b, sev_b, _ in train_dl:
                    loss, nc = 0., 0
                    for imgs, ts, sev in zip(imgs_l, ts_b, sev_b):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            s, a = model(imgs.to(device))
                        s = s.float().mean()
                        tgt = (ts.item() - mu) / sd
                        l = (s - torch.tensor(tgt, device=device)) ** 2
                        if cfg['with_cls']:
                            ce = nn.functional.cross_entropy(
                                a.float().mean(0, keepdim=True),
                                sev.unsqueeze(0).to(device), weight=oa_w.to(device))
                            l = 0.5 * l + 0.5 * ce
                        loss = loss + l
                        nc += 1
                    (loss / max(nc, 1)).backward()
                    opt.step()
                    opt.zero_grad()
                sched.step()

                model.eval()
                p, t = [], []
                with torch.inference_mode():
                    for imgs_l, ts_b, _, _ in val_dl:
                        for imgs, ts in zip(imgs_l, ts_b):
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                s, _ = model(imgs.to(device))
                            p.append(s.float().mean().item() * sd + mu)
                            t.append(ts.item())
                vmae = float(np.mean(np.abs(np.array(p) - np.array(t))))
                if vmae < best[0]:
                    best = (vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep)
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"    [{arm}] s{seed} f{fold} ep{ep+1}/{epochs} "
                          f"val_MAE={vmae:.3f} best={best[0]:.3f} "
                          f"pred_sd={np.std(p):.2f} ({time.time()-t_start:.0f}s)", flush=True)

            model.load_state_dict(best[1])
            model.eval()
            with torch.inference_mode():
                for imgs_l, _, _, ids in test_dl:
                    for imgs, cid in zip(imgs_l, ids):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            s, a = model(imgs.to(device))
                        rec = {'score': s.float().mean().item() * sd + mu}
                        if a is not None:
                            rec['assess'] = int(a.float().mean(0).argmax().item())
                        oof.setdefault(cid, []).append(rec)
            del model, opt, sched
            torch.cuda.empty_cache()
        print(f"  [{arm}] seed {seed} done", flush=True)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--arms', default='A_reg_only_raw,B_reg_only_znorm,C_multi_balanced')
    args = ap.parse_args()

    out = Path('/root/nailfold/artifacts/experiments/exp_c_loss_fix')
    out.mkdir(parents=True, exist_ok=True)
    device = f'cuda:{args.gpu}'

    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    L['ts'] = pd.to_numeric(L.total_score, errors='coerce')
    dev = L[(L.evaluation_role == 'development') & L.ts.notna()
            & L.overall_assessment.isin(SEV)].copy()
    n0 = len(dev)
    dev = dev[~dev.exam_case_id.isin(CONFLICTED)].copy()
    dev['sev'] = dev.overall_assessment.map(SEV_MAP)
    dev['development_fold'] = dev.development_fold.astype(int)
    print(f"dev cases: {n0} -> {len(dev)} after excluding {n0-len(dev)} conflicted", flush=True)
    print(f"true_ts: mean={dev.ts.mean():.3f} std={dev.ts.std():.3f} median={dev.ts.median():.3f}", flush=True)

    frames = pd.read_csv(FRAMES, dtype={'exam_case_id': str})
    frames = frames[frames.exam_case_id.isin(set(dev.exam_case_id))]

    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto)
    del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    cnt = dev.sev.value_counts()
    oa_w = torch.tensor([min(math.sqrt(len(dev) / (5 * max(cnt.get(i, 1), 1))), 5.0)
                         for i in range(5)], dtype=torch.float32)

    all_summary = {}
    for arm in args.arms.split(','):
        cfg = ARMS[arm]
        print(f"\n{'='*60}\nARM {arm}  {cfg}\n{'='*60}", flush=True)
        oof = run_arm(arm, cfg, dev, frames, tr, ev, device, args.epochs, oa_w)

        idx = dev.set_index('exam_case_id')
        rows = []
        for cid, es in oof.items():
            if cid not in idx.index:
                continue
            pred = float(np.median([e['score'] for e in es]))
            r = {'case': cid, 'true_ts': float(idx.at[cid, 'ts']), 'pred_ts': pred,
                 'true_level': idx.at[cid, 'overall_assessment'],
                 'pred_level_from_score': cut_inclusive(pred)}
            if 'assess' in es[0]:
                r['pred_level_classify'] = SEV[Counter([e['assess'] for e in es]).most_common(1)[0][0]]
            rows.append(r)
        r = pd.DataFrame(rows)
        r.to_csv(out / f'{arm}_oof.csv', index=False)

        mae = float(np.mean(np.abs(r.true_ts - r.pred_ts)))
        sp = float(spearmanr(r.true_ts, r.pred_ts).statistic)
        const = float(np.median(r.true_ts))
        s = {'arm': arm, 'n': len(r), 'mae': mae, 'spearman': sp,
             'pred_std': float(r.pred_ts.std()), 'true_std': float(r.true_ts.std()),
             'pred_mean': float(r.pred_ts.mean()), 'true_mean': float(r.true_ts.mean()),
             'std_ratio': float(r.pred_ts.std() / r.true_ts.std()),
             'constant_mae_insample': float(np.mean(np.abs(r.true_ts - const)))}
        all_summary[arm] = s
        print(json.dumps(s, indent=2), flush=True)
        (out / 'summary.json').write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + '\n')

    print("\n" + "=" * 60)
    print(f"{'arm':20s} {'MAE':>7s} {'const':>7s} {'rho':>7s} {'pred_sd':>8s} {'sd_ratio':>9s}")
    for a, s in all_summary.items():
        print(f"{a:20s} {s['mae']:7.3f} {s['constant_mae_insample']:7.3f} "
              f"{s['spearman']:7.3f} {s['pred_std']:8.3f} {s['std_ratio']:9.3f}")
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
