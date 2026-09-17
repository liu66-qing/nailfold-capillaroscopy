"""EXP-B: does temporal information add anything for flow fields?

EXP-A killed the naive design. Video is NOT a random subset: of the 95 dev
cases that have video, 93 are 中度/重度 (正常=0, 大致正常=1, 轻度=1), and the
negative classes of the flow fields are systematically missing
(rbc_aggregation 无: 31 -> 2; microthrombus 无: 108 -> 29). Video was saved
selectively when the clinician saw something abnormal. So "full dev static vs
video subset + temporal" confounds temporal information with sampling bias.

Corrected design - self-controlled, one fixed cohort:
  cohort  = the 95 dev cases that HAVE video (minus conflicted)
  arm S   = static CAPorg frames only          (same cohort)
  arm T   = video frames only, temporally spread (same cohort)
  arm ST  = both                                (same cohort)
Any S->T/ST difference is attributable to the input, not the cohort.

Target: flow_state only. Inside the video cohort it retains 4 classes with
>=5 cases (粒流 39, 粒线流 31, 线粒流 12, 粒缓流 11); everything rarer is
dropped rather than merged, and the dropped count is reported.

Frames are decoded once to JPEG and cached; 66.75 GB of msmpeg4v1/rawvideo AVI
is not decoded on the fly.
"""
import argparse, json, math, random, subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

SEEDS = [17, 42, 123]
DATA = Path('/root/nailfold/data')
CACHE = Path('/root/autodl-tmp/nailfold/video_frames')
MANIFEST = '/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
FRAMES = '/root/nailfold/artifacts/features/image_index.csv'
WEIGHTS = '/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth'
OUT = Path('/root/nailfold/artifacts/experiments/exp_b_video_flow')

TARGET = 'flow_state'
N_PER_VIDEO = 8          # frames sampled evenly across each clip
MIN_CLASS = 5            # classes rarer than this are dropped, not merged
CONFLICTED = {"recovered_archive2/180", "recovered_archive3/263"}


def extract_frames(force=False):
    """Decode N_PER_VIDEO evenly spaced frames per clip into CACHE."""
    CACHE.mkdir(parents=True, exist_ok=True)
    vids = sorted(DATA.glob('recovered_archive*/*/*.avi'))
    print(f"found {len(vids)} clips", flush=True)
    rows = []
    for i, v in enumerate(vids, 1):
        case = f"{v.parent.parent.name}/{v.parent.name}"
        stem = v.stem
        odir = CACHE / v.parent.parent.name / v.parent.name
        odir.mkdir(parents=True, exist_ok=True)
        done = sorted(odir.glob(f'{stem}_f*.jpg'))
        if len(done) < N_PER_VIDEO or force:
            # nb_frames is unreliable on these files; count via ffprobe packets
            try:
                n = int(subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                     '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(v)],
                    capture_output=True, text=True, timeout=120).stdout.strip() or 0)
            except Exception:
                n = 0
            if n < 2:
                continue
            idxs = np.linspace(0, n - 1, N_PER_VIDEO).astype(int)
            sel = '+'.join(f'eq(n\\,{k})' for k in idxs)
            subprocess.run(
                ['ffmpeg', '-v', 'error', '-y', '-i', str(v), '-vf',
                 f"select='{sel}'", '-vsync', '0', '-q:v', '3',
                 str(odir / f'{stem}_f%02d.jpg')],
                capture_output=True, timeout=600)
            done = sorted(odir.glob(f'{stem}_f*.jpg'))
        for f in done:
            rows.append({'exam_case_id': case,
                         'image_path': str(f.relative_to(CACHE)), 'src': 'video'})
        if i % 25 == 0:
            print(f"  {i}/{len(vids)} clips, {len(rows)} frames", flush=True)
    idx = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    idx.to_csv(OUT / 'video_frame_index.csv', index=False)
    print(f"extracted {len(idx)} frames over {idx.exam_case_id.nunique()} cases", flush=True)
    return idx


class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__()
        self.v, self.u = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))), 1)
        return (a * x).sum(1)


class Net(nn.Module):
    def __init__(self, backbone, n_cls):
        super().__init__()
        self.b = backbone
        self.pool = GatedPool()
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(1536, n_cls)

    def forward(self, x):
        z = self.b.forward_features(x)
        z = torch.cat([z[:, 0], self.pool(z[:, 1:])], 1)
        return self.head(self.drop(z))


class Cases(Dataset):
    def __init__(self, ids, frames, labels, transform, roots):
        self.ids = list(ids)
        sub = frames[frames.exam_case_id.isin(set(ids))]
        self.groups = {c: g for c, g in sub.groupby('exam_case_id')}
        self.ids = [i for i in self.ids if i in self.groups]
        self.labels = labels.set_index('exam_case_id')
        self.t = transform
        self.roots = roots

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        g = self.groups[cid]
        if len(g) > 16:
            g = g.sample(16, random_state=0)
        imgs = torch.stack([self.t(Image.open(self.roots[r.src] / r.image_path).convert('RGB'))
                            for _, r in g.iterrows()])
        return imgs, int(self.labels.loc[cid, 'y']), cid


def collate(b):
    return [x[0] for x in b], torch.tensor([x[1] for x in b]), [x[2] for x in b]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(arm, frames, dev, n_cls, w, tr, ev, roots, device, epochs):
    oof = {}
    for seed in SEEDS:
        seed_all(seed)
        for fold in range(5):
            vf = (fold + 1) % 5
            tri = dev.loc[~dev.development_fold.isin([fold, vf]), 'exam_case_id'].tolist()
            vai = dev.loc[dev.development_fold.eq(vf), 'exam_case_id'].tolist()
            tei = dev.loc[dev.development_fold.eq(fold), 'exam_case_id'].tolist()
            mk = lambda ids, t, sh: DataLoader(Cases(ids, frames, dev, t, roots), batch_size=2,
                                              shuffle=sh, collate_fn=collate, num_workers=2)
            train_dl, val_dl, test_dl = mk(tri, tr, True), mk(vai, ev, False), mk(tei, ev, False)
            if len(train_dl.dataset) == 0 or len(test_dl.dataset) == 0:
                continue

            b = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
            b.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=False)
            for p in b.parameters():
                p.requires_grad = False
            model = Net(b, n_cls).to(device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=1e-3, weight_decay=0.05)
            sched = CosineAnnealingLR(opt, T_max=epochs)
            best = (-1, None)

            for _ in range(epochs):
                model.train()
                for imgs_l, y_b, _ in train_dl:
                    loss, nc = 0., 0
                    for imgs, y in zip(imgs_l, y_b):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            lg = model(imgs.to(device))
                        lg = lg.float()
                        yd = y.unsqueeze(0).to(device)
                        loss = loss + 0.75 * nn.functional.cross_entropy(
                            lg.mean(0, keepdim=True), yd, weight=w.to(device)) \
                            + 0.25 * nn.functional.cross_entropy(
                            lg, y.expand(len(imgs)).to(device), weight=w.to(device))
                        nc += 1
                    (loss / max(nc, 1)).backward()
                    opt.step(); opt.zero_grad()
                sched.step()

                model.eval()
                P, T = [], []
                with torch.inference_mode():
                    for imgs_l, y_b, _ in val_dl:
                        for imgs, y in zip(imgs_l, y_b):
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                lg = model(imgs.to(device))
                            P.append(int(lg.float().mean(0).argmax())); T.append(int(y))
                ba = balanced_accuracy_score(T, P) if len(set(T)) > 1 else 0.
                if ba > best[0]:
                    best = (ba, {k: v.cpu().clone() for k, v in model.state_dict().items()})

            if best[1]:
                model.load_state_dict(best[1])
            model.eval()
            with torch.inference_mode():
                for imgs_l, _, ids in test_dl:
                    for imgs, cid in zip(imgs_l, ids):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            lg = model(imgs.to(device))
                        oof.setdefault(cid, []).append(int(lg.float().mean(0).argmax()))
            del model, opt, sched
            torch.cuda.empty_cache()
        print(f"  [{arm}] seed {seed} done", flush=True)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--arms', default='S_static,T_video,ST_both')
    ap.add_argument('--extract-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    vidx_p = OUT / 'video_frame_index.csv'
    vidx = pd.read_csv(vidx_p) if vidx_p.exists() else extract_frames()
    if args.extract_only:
        return

    L = pd.read_csv(MANIFEST, dtype={'exam_case_id': str})
    dev = L[(L.evaluation_role == 'development') & L[TARGET].notna()].copy()
    dev = dev[~dev.exam_case_id.isin(CONFLICTED)]

    cohort = set(vidx.exam_case_id) & set(dev.exam_case_id)
    dev = dev[dev.exam_case_id.isin(cohort)].copy()
    vc = dev[TARGET].value_counts()
    keep = vc[vc >= MIN_CLASS].index.tolist()
    dropped = int(len(dev) - dev[TARGET].isin(keep).sum())
    dev = dev[dev[TARGET].isin(keep)].copy()
    classes = sorted(keep)
    dev['y'] = dev[TARGET].map({c: i for i, c in enumerate(classes)})
    dev['development_fold'] = dev.development_fold.astype(int)

    print(f"video cohort: {len(cohort)} cases; usable after class filter: {len(dev)} "
          f"(dropped {dropped} in rare classes)", flush=True)
    print(f"classes kept: {dict(dev[TARGET].value_counts())}", flush=True)
    print("cohort level mix:", dict(dev.overall_assessment.value_counts()), flush=True)

    static = pd.read_csv(FRAMES, dtype={'exam_case_id': str})
    static['src'] = 'static'
    vidx['src'] = 'video'
    roots = {'static': DATA, 'video': CACHE}
    sets = {'S_static': static, 'T_video': vidx,
            'ST_both': pd.concat([static, vidx], ignore_index=True)}

    proto = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(proto); del proto
    tr = timm.data.create_transform(**dc, is_training=True, color_jitter=0, hflip=0.5, vflip=0)
    ev = timm.data.create_transform(**dc, is_training=False)

    cnt = dev.y.value_counts()
    n_cls = len(classes)
    w = torch.tensor([min(math.sqrt(len(dev) / (n_cls * max(cnt.get(i, 1), 1))), 5.0)
                      for i in range(n_cls)], dtype=torch.float32)

    summ = {}
    for arm in args.arms.split(','):
        fr = sets[arm]
        fr = fr[fr.exam_case_id.isin(set(dev.exam_case_id))]
        print(f"\n{'='*60}\nARM {arm}: {len(fr)} frames / {fr.exam_case_id.nunique()} cases\n{'='*60}", flush=True)
        oof = run(arm, fr, dev, n_cls, w, tr, ev, roots, f'cuda:{args.gpu}', args.epochs)
        idx = dev.set_index('exam_case_id')
        rows = [{'case': c, 'true': int(idx.at[c, 'y']),
                 'pred': Counter(v).most_common(1)[0][0]}
                for c, v in oof.items() if c in idx.index]
        r = pd.DataFrame(rows)
        r.to_csv(OUT / f'{arm}_oof.csv', index=False)
        ba = float(balanced_accuracy_score(r.true, r.pred))
        maj = float((r.true == r.true.mode()[0]).mean())
        summ[arm] = {'n': len(r), 'balanced_acc': ba, 'acc': float((r.true == r.pred).mean()),
                     'majority_rate': maj, 'random_ba': 1.0 / n_cls,
                     'lift_over_random': ba - 1.0 / n_cls, 'classes': classes}
        print(json.dumps(summ[arm], ensure_ascii=False, indent=2), flush=True)
        (OUT / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=2) + '\n')

    print("\n" + "=" * 60)
    print(f"{'arm':12s} {'n':>4s} {'BA':>7s} {'lift':>7s} {'acc':>7s} {'majority':>9s}")
    for a, s in summ.items():
        print(f"{a:12s} {s['n']:4d} {s['balanced_acc']:7.3f} {s['lift_over_random']:7.3f} "
              f"{s['acc']:7.3f} {s['majority_rate']:9.3f}")


if __name__ == '__main__':
    main()
