"""EXP-G: per-field specialist classifiers for the image-observable nailfold fields.

One INDEPENDENT model per field (no shared trunk, no shared heads). DINOv2 ViT-B/14
stays FROZEN, so patch features are precomputed ONCE into a view cache and each
specialist trains only a small MIL head on top. This makes 3 seeds x 5 folds x N
fields cheap, which is what lets us actually compare aggregation/loss choices.

Design decisions (see rationale in the accompanying report):
  * Aspect-correct 518x686 input (source images are 1024x768; timm's default
    518x518 squashes a 4:3 image, and RandomResizedCrop(scale=(0.08,1.0)) destroys
    count-based labels). Augmentation is restricted to scale=(0.70,1.0), hflip,
    mild photometric jitter -- no vflip (papilla / subpapillary orientation is
    semantically up-down).
  * Label space merged per field so every class survives a 5-fold split; merged
    classes get a population-weighted score so score MAE stays well defined.
  * Hybrid loss: CE (class-weighted) + lambda * MSE on the score value. Score-MSE
    was the one component that helped in exp6 (c3: 3.133 vs 3.412 baseline) and it
    directly optimises the metric that composes into total_score.
  * Frame aggregation is per-field: attention MIL for spot/presence fields
    (microthrombus, exudation), mean for density/global fields (capillary_count,
    papilla). Case label broadcast to frames is a weak assumption for spot fields.
  * Three score point-estimates are reported (argmax, expected, median). MAE is
    minimised by the MEDIAN of the predictive distribution, not the mean or argmax.

Protocol is fixed and must not be changed:
  development rows only, 2 conflicted cases excluded, folds from development_fold
  (test=fold, val=(fold+1)%5), seeds [17,42,123], bf16 autocast, logits .float().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
IMAGE_INDEX = '/root/nailfold/artifacts/features/image_index.csv'
IMAGE_ROOT = '/root/nailfold/data'
WEIGHTS = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'
RULES = '/root/autodl-tmp/nailfold/artifacts/labels/score_rules_v3.json'
OUT_DIR = '/root/nailfold/artifacts/experiments/exp_g_perfield'
CACHE_DIR = '/root/autodl-tmp/nailfold/cache/exp_g_feat'

EXCLUDE_CASES = ['recovered_archive2/180', 'recovered_archive3/263']
SEEDS = [17, 42, 123]
N_FOLDS = 5
MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'
GRID_H, GRID_W = 3, 4          # region grid pooled from the 37x49 patch map
EMB = 768

# ─────────────────────────── field configuration ────────────────────────────
# label_map : raw report string -> class index (-1 = drop this case for this field)
# class_score: class index -> score contribution. For merged classes this is the
#              population-weighted mean of the merged members' scores, computed on
#              development rows only, so it is a well-defined expected score.
# agg       : frame aggregation ('attn' = gated attention MIL, 'mean')
FIELD_CFG = {
    'microthrombus': dict(
        label_map={'无': 0, '[未见]': 0, '1--2': 1, '>2': 2, '个/min': -1},
        agg='attn',
        note='spot/presence field, 3 healthy classes (106/45/29), no merge needed',
    ),
    'exudation': dict(
        label_map={'无': 0, '+': 1, '++': 2, '+++': 2},
        agg='attn',
        note='+++ has only 4 dev cases -> merged into ++ (weighted score)',
    ),
    'capillary_count': dict(
        label_map={'>=7': 0, '5--6': 1, '3--4': 2, '1--2': 2, '<1': 2, '条/mm': -1},
        agg='mean',
        note='<1 n=1 and 1--2 n=0 in dev -> merged into 3--4 as "<=4" (weighted score)',
    ),
    'papilla': dict(
        label_map={'波纹状': 0, '浅波纹状': 1, '平坦': 2},
        agg='mean',
        note='global morphology, 3 healthy classes (66/76/41)',
    ),
    'subpapillary_venous_plexus': dict(
        label_map={'不见': 0, '可见1排': 1, '可见2排': 2, '>2排,扩张': 3},
        agg='mean', note='4 healthy classes'),
    'blood_color': dict(
        label_map={'淡红': 0, '浅红': 1, '暗红': 2, '暗紫': 2},
        agg='mean', note='暗紫 n=2 -> merged into 暗红'),
    'clarity': dict(
        label_map={'清晰': 0, '不清': 1, '模糊': 2},
        agg='mean', note='3 healthy classes'),
}
DEFAULT_FIELDS = ['microthrombus', 'exudation', 'capillary_count', 'papilla']


def log(*a):
    print(*a, flush=True)


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ───────────────────────────── score bookkeeping ─────────────────────────────
def build_class_scores(dev: pd.DataFrame, fields):
    """class index -> score, using population-weighted means for merged classes.

    Returns {field: {'scores': np.ndarray[n_cls], 'members': {cls: [(raw, score, n)]}}}
    """
    rules = json.load(open(RULES))['field_rules']
    out = {}
    for f in fields:
        mapping = rules[f]['mapping']
        lm = FIELD_CFG[f]['label_map']
        n_cls = max(v for v in lm.values()) + 1
        counts = dev[f].value_counts()
        members = {c: [] for c in range(n_cls)}
        for raw, ci in lm.items():
            if ci < 0 or raw not in mapping:
                continue
            members[ci].append((raw, float(mapping[raw]), int(counts.get(raw, 0))))
        scores = np.zeros(n_cls, dtype=np.float64)
        for ci, mem in members.items():
            if not mem:
                raise RuntimeError(f'{f}: class {ci} has no scored member')
            wsum = sum(m[2] for m in mem)
            if wsum == 0:                       # no dev support -> unweighted mean
                scores[ci] = float(np.mean([m[1] for m in mem]))
            else:
                scores[ci] = sum(m[1] * m[2] for m in mem) / wsum
        out[f] = dict(scores=scores, n_cls=n_cls, members=members)
    return out


def true_field_score(dev: pd.DataFrame, f: str) -> pd.Series:
    """Ground-truth score for a field straight from the raw report value."""
    mapping = json.load(open(RULES))['field_rules'][f]['mapping']
    return dev.set_index('exam_case_id')[f].map(mapping)


# ───────────────────────── feature cache (frozen trunk) ──────────────────────
class ViewSet(Dataset):
    """One deterministic augmented view of every frame, at fixed resolution."""

    def __init__(self, rows, root, size, view_seed, train_view: bool):
        self.rows = rows
        self.root = Path(root)
        self.h, self.w = size
        self.view_seed = view_seed
        self.train_view = train_view

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cid, rel = self.rows[i]
        img = Image.open(self.root / rel).convert('RGB')
        rng = random.Random(hash((self.view_seed, rel)) & 0xFFFFFFFF)
        if self.train_view:
            # Conservative crop: keep >=70% of area so counts/density survive.
            s = rng.uniform(0.70, 1.0)
            ar = rng.uniform(0.92, 1.08)
            W, H = img.size
            cw = min(W, int(round(W * math.sqrt(s * ar))))
            ch = min(H, int(round(H * math.sqrt(s / ar))))
            x0 = rng.randint(0, W - cw)
            y0 = rng.randint(0, H - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            img = img.resize((self.w, self.h), Image.BICUBIC)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            a = np.asarray(img, dtype=np.float32) / 255.0
            # mild photometric jitter: microscope illumination varies per session
            a = np.clip((a - 0.5) * rng.uniform(0.9, 1.1) + 0.5
                        + rng.uniform(-0.06, 0.06), 0, 1)
        else:
            img = img.resize((self.w, self.h), Image.BICUBIC)
            a = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        a = (a - mean) / std
        return torch.from_numpy(a.transpose(2, 0, 1)), i


def build_backbone(device):
    b = torch.timm_model = __import__('timm').create_model(
        MODEL_NAME, pretrained=False, num_classes=0, dynamic_img_size=True)
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    r = b.load_state_dict(sd, strict=False)
    log(f'  backbone loaded: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}')
    for p in b.parameters():
        p.requires_grad = False
    return b.to(device).eval()


@torch.inference_mode()
def encode_views(rows, size, n_views, device, batch=8, workers=6):
    """Precompute frozen features for every frame under n_views augmentations.

    Returns float16 array [n_views, n_frames, D] where D = EMB*(1+GRID_H*GRID_W):
    CLS token plus a GRID_H x GRID_W average-pooled region grid. The grid preserves
    coarse spatial layout so the head can still localise; full patch tokens would be
    1814*768 per frame which is far too large to cache.
    """
    tag = f'{size[0]}x{size[1]}_g{GRID_H}x{GRID_W}_v{n_views}_n{len(rows)}'
    cache = Path(CACHE_DIR) / f'feat_{tag}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    D = EMB * (1 + GRID_H * GRID_W)
    if cache.exists():
        # load fully into RAM (float16, ~0.2 GB at 5 views) -- mmap random access
        # per case dominated runtime in timing tests
        arr = np.load(cache)
        if arr.shape == (n_views, len(rows), D):
            log(f'  feature cache hit: {cache} {arr.shape}')
            return arr
        log('  cache shape mismatch, recomputing')
    bb = build_backbone(device)
    out = np.zeros((n_views, len(rows), D), dtype=np.float16)
    for v in range(n_views):
        ds = ViewSet(rows, IMAGE_ROOT, size, view_seed=1000 + v, train_view=(v > 0))
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=True)
        t0 = time.time()
        done = 0
        for x, idxs in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = bb.forward_features(x)
            f = f.float()
            cls = f[:, 0]
            pt = f[:, bb.num_prefix_tokens:]
            gh, gw = size[0] // 14, size[1] // 14
            pt = pt.transpose(1, 2).reshape(pt.shape[0], EMB, gh, gw)
            reg = F.adaptive_avg_pool2d(pt, (GRID_H, GRID_W)).flatten(1)
            feats = torch.cat([cls, reg], 1).cpu().numpy().astype(np.float16)
            out[v, idxs.numpy()] = feats
            done += len(idxs)
            if done % 480 == 0:
                log(f'    view {v}: {done}/{len(rows)} frames '
                    f'({time.time() - t0:.0f}s)')
        log(f'  view {v} done in {time.time() - t0:.0f}s')
    del bb
    torch.cuda.empty_cache()
    np.save(cache, out)
    log(f'  feature cache written: {cache} {out.shape} '
        f'({out.nbytes / 1e9:.2f} GB)')
    return out


# ──────────────────────────────── specialist head ────────────────────────────
class Specialist(nn.Module):
    """Small per-field MIL head over frozen frame features.

    frame_proj shrinks the concatenated CLS+grid vector to `hid`, then frames are
    aggregated to a case vector, then one linear classifier. Deliberately tiny:
    at n=183 cases anything larger overfits (see exp7 unfreezing disaster).
    """

    def __init__(self, in_dim, n_cls, agg='mean', hid=192, p_drop=0.3):
        super().__init__()
        self.agg = agg
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(p_drop),
            nn.Linear(in_dim, hid), nn.GELU())
        if agg == 'attn':
            self.att_v = nn.Linear(hid, 64)
            self.att_u = nn.Linear(hid, 64)
            self.att_w = nn.Linear(64, 1)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hid, n_cls))

    def forward(self, x):
        """x: [n_frames, in_dim] for ONE case -> (case_logits[1,C], frame_logits[n,C], attn[n])"""
        h = self.frame_proj(x)
        frame_logits = self.head(h)
        if self.agg == 'attn':
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))
            a = torch.softmax(a, dim=0)
            case_vec = (a * h).sum(0, keepdim=True)
            attn = a.squeeze(-1)
        else:
            case_vec = h.mean(0, keepdim=True)
            attn = torch.full((h.shape[0],), 1.0 / h.shape[0], device=h.device)
        return self.head(case_vec), frame_logits, attn


# ────────────────────────────── train / eval one fold ────────────────────────
def run_fold(field, cfg, cls_scores, feats, frame_slices, labels, ids_by_fold,
             fold, seed, device, epochs, args):
    """Train one specialist on one fold. Returns per-case OOF predictions."""
    n_cls = cls_scores['n_cls']
    scores_t = torch.tensor(cls_scores['scores'], dtype=torch.float32, device=device)
    test_ids = ids_by_fold[fold]
    val_ids = ids_by_fold[(fold + 1) % N_FOLDS]
    train_ids = [c for k in range(N_FOLDS) if k not in (fold, (fold + 1) % N_FOLDS)
                 for c in ids_by_fold[k]]
    train_ids = [c for c in train_ids if labels[c] >= 0]
    val_ids = [c for c in val_ids if labels[c] >= 0]
    test_ids = [c for c in test_ids if labels[c] >= 0]

    n_views = feats.shape[0]
    in_dim = feats.shape[2]

    # class weights: sqrt inverse frequency, capped -- full inverse frequency
    # over-corrects at n~110 train cases and destabilises the minority class.
    cnt = np.bincount([labels[c] for c in train_ids], minlength=n_cls).astype(float)
    w = np.sqrt(cnt.sum() / (n_cls * np.maximum(cnt, 1)))
    w = np.minimum(w, 4.0)
    w[cnt == 0] = 0.0
    cw = torch.tensor(w, dtype=torch.float32, device=device)

    model = Specialist(in_dim, n_cls, agg=cfg['agg'], hid=args.hid,
                       p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Pre-stage every case's features on the GPU once per fold. Total is tiny
    # (~0.2 GB fp16) and removes a host->device copy from every training step,
    # which was the dominant cost in timing tests.
    if not hasattr(run_fold, '_gpu_cache') or run_fold._gpu_key != id(feats):
        run_fold._gpu_cache = {}
        run_fold._gpu_key = id(feats)
        for cid, (s, e) in frame_slices.items():
            run_fold._gpu_cache[cid] = torch.from_numpy(
                np.ascontiguousarray(feats[:, s:e])).to(device).float()
    gpu_cache = run_fold._gpu_cache

    def case_feats(cid, view):
        return gpu_cache[cid][view]

    def evaluate(ids, view=0):
        model.eval()
        probs, ys = {}, {}
        with torch.inference_mode():
            for cid in ids:
                cl, _, _ = model(case_feats(cid, view))
                probs[cid] = torch.softmax(cl.float(), -1)[0].cpu().numpy()
                ys[cid] = labels[cid]
        return probs, ys

    def val_objective(probs, ys):
        """Selection metric: -score MAE via median estimator, BA as tiebreak.
        Score MAE is what composes into total_score, so select on it directly."""
        ids = list(probs)
        if not ids:
            return -1e9, 0.0, 0.0
        yt = np.array([ys[c] for c in ids])
        P = np.stack([probs[c] for c in ids])
        pred_s = np.array([median_score(p, cls_scores['scores']) for p in P])
        true_s = cls_scores['scores'][yt]
        mae = float(np.abs(pred_s - true_s).mean())
        ba = balanced_accuracy_score(yt, P.argmax(1)) if len(set(yt)) > 1 else 0.0
        return -mae + 0.05 * ba, mae, ba

    best = (-1e9, None, 0)
    order = list(train_ids)
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(order)
        tot, nb = 0.0, 0
        for bstart in range(0, len(order), args.accum):
            chunk = order[bstart:bstart + args.accum]
            opt.zero_grad()
            loss_sum = 0.0
            for cid in chunk:
                v = random.randrange(1, n_views) if n_views > 1 else 0
                x = case_feats(cid, v)
                y = torch.tensor([labels[cid]], device=device)
                cl, fl, _ = model(x)
                cl, fl = cl.float(), fl.float()
                ce_case = F.cross_entropy(cl, y, weight=cw)
                ce_frame = F.cross_entropy(fl, y.expand(fl.shape[0]), weight=cw)
                # score MSE on the case-level expected score: directly targets the
                # metric, and gives the ordinal structure to the model for free.
                exp_score = (torch.softmax(cl, -1) * scores_t).sum()
                mse = (exp_score - scores_t[labels[cid]]).pow(2)
                rng_sq = max(float(scores_t.max() - scores_t.min()) ** 2, 1e-6)
                loss = ((1 - args.frame_w) * ce_case + args.frame_w * ce_frame
                        + args.lam_score * mse / rng_sq)
                loss_sum = loss_sum + loss
            (loss_sum / len(chunk)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss_sum.detach()) / len(chunk)
            nb += 1
        sched.step()
        vp, vy = evaluate(val_ids)
        obj, vmae, vba = val_objective(vp, vy)
        if obj > best[0]:
            best = (obj, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if ep % args.log_every == 0 or ep == 1 or ep == epochs:
            log(f'    [{field} s{seed} f{fold}] ep {ep:3d}/{epochs} '
                f'loss {tot / max(nb,1):.4f} val_sMAE {vmae:.4f} val_BA {vba:.3f} '
                f'best_ep {best[2]}')
    if best[1] is not None:
        model.load_state_dict(best[1])
    # test-time averaging over the clean view only (view 0) -- augmented-view TTA
    # gave no gain in earlier work and doubles cost.
    tp, ty = evaluate(test_ids, view=0)
    rows = []
    for cid in test_ids:
        p = tp[cid]
        rows.append(dict(
            case=cid, field=field, seed=seed, test_fold=fold,
            true_class=int(ty[cid]), pred_class=int(p.argmax()),
            true_score=float(cls_scores['scores'][ty[cid]]),
            pred_score_argmax=float(cls_scores['scores'][int(p.argmax())]),
            pred_score_expected=float((p * cls_scores['scores']).sum()),
            pred_score_median=float(median_score(p, cls_scores['scores'])),
            best_epoch=best[2],
            **{f'prob_{i}': float(p[i]) for i in range(n_cls)}))
    return rows


def median_score(prob, scores):
    """Weighted median of the predictive distribution over score values.
    The median minimises expected L1 (= MAE); the mean minimises L2. Since the
    reported metric is MAE, the median is the correct point estimate."""
    o = np.argsort(scores)
    c = np.cumsum(prob[o])
    k = int(np.searchsorted(c, 0.5))
    return float(scores[o[min(k, len(o) - 1)]])


# ────────────────────────────────── metrics ──────────────────────────────────
def field_metrics(df_f, cls_scores, dev, field):
    """Per-field metrics, averaged over seeds after per-seed computation."""
    scores = cls_scores['scores']
    n_cls = cls_scores['n_cls']
    per_seed = []
    for seed, g in df_f.groupby('seed'):
        y = g.true_class.values
        p = g.pred_class.values
        maj = float(pd.Series(y).value_counts(normalize=True).max())
        ba = balanced_accuracy_score(y, p) if len(set(y)) > 1 else float('nan')
        m = dict(
            seed=int(seed), n=int(len(g)),
            balanced_accuracy=float(ba), accuracy=float((y == p).mean()),
            majority_class_rate=maj, random_baseline=1.0 / n_cls,
            lift_over_random=float(ba - 1.0 / n_cls),
            score_mae_argmax=float((g.pred_score_argmax - g.true_score).abs().mean()),
            score_mae_expected=float((g.pred_score_expected - g.true_score).abs().mean()),
            score_mae_median=float((g.pred_score_median - g.true_score).abs().mean()),
        )
        # mode-fill baseline on the SAME cases -- the number the model must beat
        train_mode = int(pd.Series(y).value_counts().idxmax())
        m['score_mae_mode_baseline'] = float(
            np.abs(scores[train_mode] - g.true_score.values).mean())
        m['beats_mode_by'] = m['score_mae_mode_baseline'] - m['score_mae_median']
        per_seed.append(m)
    keys = [k for k in per_seed[0] if k != 'seed']
    mean = {k: float(np.nanmean([s[k] for s in per_seed])) for k in keys}
    std = {k: float(np.nanstd([s[k] for s in per_seed])) for k in keys}
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, pr in zip(df_f.true_class, df_f.pred_class):
        cm[t, pr] += 1
    return dict(mean=mean, std=std, per_seed=per_seed,
                confusion_all_seeds=cm.tolist(),
                class_scores=[float(x) for x in scores],
                class_members={str(k): v for k, v in cls_scores['members'].items()},
                n_classes=n_cls, agg=FIELD_CFG[field]['agg'],
                note=FIELD_CFG[field]['note'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--fields', type=str, default=','.join(DEFAULT_FIELDS))
    ap.add_argument('--seeds', type=str, default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--folds', type=str, default='0,1,2,3,4')
    ap.add_argument('--height', type=int, default=518)
    ap.add_argument('--width', type=int, default=686)
    ap.add_argument('--n_views', type=int, default=5,
                    help='1 clean + (n-1) augmented cached views')
    ap.add_argument('--hid', type=int, default=192)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--accum', type=int, default=8, help='cases per optimiser step')
    ap.add_argument('--frame_w', type=float, default=0.25)
    ap.add_argument('--lam_score', type=float, default=0.5)
    ap.add_argument('--log_every', type=int, default=5)
    ap.add_argument('--out', type=str, default=OUT_DIR)
    ap.add_argument('--tag', type=str, default='')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'
    fields = [f.strip() for f in args.fields.split(',') if f.strip()]
    seeds = [int(s) for s in args.seeds.split(',')]
    folds = [int(f) for f in args.folds.split(',')]
    for f in fields:
        if f not in FIELD_CFG:
            raise SystemExit(f'unknown field {f}; known: {list(FIELD_CFG)}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f'=== EXP-G per-field specialists ===')
    log(f'fields={fields} seeds={seeds} folds={folds} epochs={args.epochs} '
        f'res={args.height}x{args.width} views={args.n_views} gpu={args.gpu}')

    # ---- data ----
    df = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    dev = df[(df.evaluation_role == 'development')
             & (~df.exam_case_id.isin(EXCLUDE_CASES))].copy()
    log(f'development cases after excluding {EXCLUDE_CASES}: {len(dev)}')
    assert dev.development_fold.notna().all(), 'null development_fold'

    idx = pd.read_csv(IMAGE_INDEX, dtype={'exam_case_id': str})
    idx = idx[idx.exam_case_id.isin(set(dev.exam_case_id))].copy()
    idx = idx.sort_values(['exam_case_id', 'image_path']).reset_index(drop=True)
    rows = list(zip(idx.exam_case_id, idx.image_path))
    frame_slices = {}
    for cid, g in idx.groupby('exam_case_id', sort=False):
        frame_slices[cid] = (int(g.index[0]), int(g.index[-1]) + 1)
    log(f'frames: {len(rows)} over {len(frame_slices)} cases '
        f'(mean {len(rows)/len(frame_slices):.2f}/case)')

    ids_by_fold = {k: sorted(dev.loc[dev.development_fold == k, 'exam_case_id'])
                   for k in range(N_FOLDS)}
    for k in range(N_FOLDS):
        ids_by_fold[k] = [c for c in ids_by_fold[k] if c in frame_slices]
    log('fold sizes: ' + str({k: len(v) for k, v in ids_by_fold.items()}))

    cls_scores_all = build_class_scores(dev, fields)
    for f in fields:
        cs = cls_scores_all[f]
        log(f'  {f}: n_cls={cs["n_cls"]} scores={np.round(cs["scores"],4).tolist()}')
        for ci, mem in cs['members'].items():
            log(f'      class {ci}: ' + ', '.join(
                f'{r}(score {s}, n={n})' for r, s, n in mem))

    # ---- frozen features ----
    t0 = time.time()
    feats = encode_views(rows, (args.height, args.width), args.n_views, device)
    log(f'features ready in {time.time() - t0:.0f}s, dim={feats.shape[2]}')

    # ---- per-field training ----
    summary = {'config': vars(args), 'n_dev_cases': int(len(dev)),
               'n_frames': len(rows), 'fields': {}}
    all_rows = []
    for field in fields:
        cfg = FIELD_CFG[field]
        lm = cfg['label_map']
        raw = dev.set_index('exam_case_id')[field]
        labels = {}
        n_drop = 0
        for cid in frame_slices:
            v = raw.get(cid)
            ci = lm.get(v, -1) if isinstance(v, str) else -1
            labels[cid] = ci
            if ci < 0:
                n_drop += 1
        usable = sum(1 for v in labels.values() if v >= 0)
        log(f'\n--- FIELD {field} (agg={cfg["agg"]}) usable={usable} '
            f'dropped={n_drop} ---')
        log(f'    class counts: ' + str(
            pd.Series([v for v in labels.values() if v >= 0]).value_counts().sort_index().to_dict()))
        t1 = time.time()
        for seed in seeds:
            for fold in folds:
                seed_all(seed * 100 + fold)
                all_rows += run_fold(field, cfg, cls_scores_all[field], feats,
                                     frame_slices, labels, ids_by_fold, fold,
                                     seed, device, args.epochs, args)
        log(f'  {field} done in {time.time() - t1:.0f}s')

    oof = pd.DataFrame(all_rows)
    suffix = f'_{args.tag}' if args.tag else ''
    oof_path = out / f'oof_predictions{suffix}.csv'
    oof.to_csv(oof_path, index=False)
    log(f'\nOOF written: {oof_path} ({len(oof)} rows)')

    log('\n' + '=' * 100)
    log('%-28s %6s %6s %6s %6s %6s | %8s %8s %8s %8s' % (
        'field', 'BA', 'acc', 'maj', 'rand', 'lift',
        'sMAEarg', 'sMAEexp', 'sMAEmed', 'modeBase'))
    for field in fields:
        d = oof[oof.field == field]
        if not len(d):
            continue
        m = field_metrics(d, cls_scores_all[field], dev, field)
        summary['fields'][field] = m
        a = m['mean']
        log('%-28s %6.3f %6.3f %6.3f %6.3f %+6.3f | %8.4f %8.4f %8.4f %8.4f' % (
            field, a['balanced_accuracy'], a['accuracy'], a['majority_class_rate'],
            a['random_baseline'], a['lift_over_random'], a['score_mae_argmax'],
            a['score_mae_expected'], a['score_mae_median'],
            a['score_mae_mode_baseline']))
    log('=' * 100)
    for field in fields:
        if field in summary['fields']:
            b = summary['fields'][field]['mean']['beats_mode_by']
            log(f'  {field}: median-estimator beats mode baseline by {b:+.4f} '
                f'score MAE {"(GOOD)" if b > 0 else "(NO GAIN)"}')
    sum_path = out / f'summary{suffix}.json'
    json.dump(summary, open(sum_path, 'w'), indent=2, ensure_ascii=False)
    log(f'summary written: {sum_path}')


if __name__ == '__main__':
    main()
