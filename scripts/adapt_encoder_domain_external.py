"""Unlabelled domain adaptation of a vision encoder on EXTERNAL nailfold images.

Arms A1 / M1 / R1 start here. The encoder is further self-supervised on external
nailfold photographs, then frozen again and used exactly like the deployed encoder.

Why this cannot leak a validation fold
  Only EXTERNAL images are used. No local image, label, fold, or case id is read
  by this script at all. The adapted encoder is therefore fold-independent, and one
  checkpoint is legitimately shared by every outer fold -- there is no path by
  which a local validation case could influence it.

Method: DINO-style self-distillation (two global views, EMA teacher, centred and
sharpened cross-entropy over a projection head). Chosen because it is the same
objective family the base checkpoint was trained with, so adaptation continues the
pretraining rather than fighting it. The head is discarded afterwards; only the
backbone is kept.

Pool: the frozen external pool. Mendeley is EXCLUDED from adaptation by default --
every one of its 576 images carries device caliper strokes and contour overlays, so
adapting on it would teach the encoder to read another device's annotations. Pass
--include-mendeley only for an explicit sensitivity run.

Adaptation resolution is 224, not the deployed 518x686: at 8.5 GB the deployed
geometry does not fit two views through a ViT-L. Recorded as a limitation, not
hidden -- extraction afterwards still runs at the arm's own deployed geometry.

  PYTHONIOENCODING=utf-8 python scripts/adapt_encoder_domain_external.py \
      --encoder dinov2b --epochs 8
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "artifacts" / "experiments" / "rescue_external_20260922" / "frozen"
OUT_ROOT = ROOT / "artifacts" / "experiments" / "rescue_external_20260922" / "adapted"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SEED = 20260917

ENCODERS = {
    # local weight files, identical to the ones the already-run arms load, so an
    # adapted arm starts from a byte-identical base and the hub is never needed
    "dinov2b": dict(kind="timm", model="vit_base_patch14_dinov2.lvd142m",
                    weights="weights/dinov2/b/model.safetensors",
                    norm=IMAGENET, patch=14,
                    base_arm="anchor_dinov2b_deployed"),
    "dinov2l": dict(kind="timm", model="vit_large_patch14_dinov2.lvd142m",
                    weights="weights/dinov2/l/model.safetensors",
                    norm=IMAGENET, patch=14,
                    base_arm="dinov2l_deployed_geometry"),
    "retfound": dict(kind="retfound", model="vit_large_patch14_dinov2.lvd142m",
                     weights="weights/RETFound_dinov2_meh/RETFound_dinov2_meh.pth",
                     norm=IMAGENET, patch=14, base_arm="retfound_dinov2_meh"),
    "medsiglip": dict(kind="medsiglip", model="siglip_vision",
                      weights="weights/medsiglip-448/model.safetensors",
                      norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), patch=14,
                      base_arm="medsiglip_medical"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class TwoView(Dataset):
    """Two augmented global views. No colour jitter on hue/saturation: blood_color
    and clarity are downstream targets, so teaching colour invariance would destroy
    exactly the signal those fields need."""

    def __init__(self, paths, size, norm):
        self.paths, self.size, self.norm = paths, size, norm

    def __len__(self):
        return len(self.paths)

    def _view(self, im):
        import torchvision.transforms as T
        tf = T.Compose([
            T.RandomResizedCrop(self.size, scale=(0.4, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(*self.norm)])
        return tf(im)

    def __getitem__(self, i):
        try:
            im = Image.open(ROOT / self.paths[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (self.size, self.size), (128, 128, 128))
        return self._view(im), self._view(im)


class Head(nn.Module):
    def __init__(self, dim, out=4096, hidden=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 256))
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(256, out, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1, p=2))


def build_backbone(cfg, device):
    if cfg["kind"] == "medsiglip":
        from transformers import SiglipVisionModel
        w = ROOT / cfg["weights"]
        m = SiglipVisionModel.from_pretrained(str(w.parent), dtype=torch.float32)
        return m.to(device), int(m.config.hidden_size), sha256(w)
    import timm
    if cfg["kind"] == "retfound":
        m = timm.create_model(cfg["model"], pretrained=False, num_classes=0,
                              img_size=518, dynamic_img_size=True)
        w = ROOT / cfg["weights"]
        blob = torch.load(str(w), map_location="cpu", weights_only=False)
        sd = {k[len("backbone."):]: v for k, v in blob["teacher"].items()
              if k.startswith("backbone.")}
        sd.pop("mask_token", None)
        m.load_state_dict(sd, strict=False)
        return m.to(device), m.num_features, sha256(w)
    from safetensors.torch import load_file
    w = ROOT / cfg["weights"]
    m = timm.create_model(cfg["model"], pretrained=False, num_classes=0,
                          img_size=518, dynamic_img_size=True)
    sd = load_file(str(w))
    sd.pop("mask_token", None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "mask_token" not in k]
    unexpected = [k for k in unexpected if "mask_token" not in k]
    if missing or unexpected:
        raise RuntimeError("state mismatch missing=%s unexpected=%s"
                           % (missing[:6], unexpected[:6]))
    return m.to(device), m.num_features, sha256(w)


def embed(m, x, kind):
    if kind == "medsiglip":
        return m(pixel_values=x).pooler_output
    return m.forward_features(x)[:, 0]


def freeze_all_but_last(model, kind, n_blocks):
    """Adapt only the last n transformer blocks plus the final norm.

    Returns the names of the trainable blocks so the manifest records exactly what
    moved. Applied identically to every encoder, so the arms stay comparable.
    """
    for p in model.parameters():
        p.requires_grad = False
    blocks = (model.vision_model.encoder.layers if kind == "medsiglip"
              else model.blocks)
    keep = list(range(max(0, len(blocks) - n_blocks), len(blocks)))
    names = []
    for i in keep:
        for p in blocks[i].parameters():
            p.requires_grad = True
        names.append("block.%d" % i)
    tail = (model.vision_model.post_layernorm if kind == "medsiglip" else model.norm)
    for p in tail.parameters():
        p.requires_grad = True
    names.append("final_norm")
    return dict(total_blocks=len(blocks), adapted=names,
                trainable_params=int(sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)),
                total_params=int(sum(p.numel() for p in model.parameters())))


def load_pool(include_mendeley: bool):
    hf = pd.read_csv(FROZEN / "external_pool_hf.csv")
    paths = hf.rel_path.tolist()
    src = {"hf_classification_original": len(paths)}
    box = pd.read_csv(FROZEN / "external_pool_hf_boxes.csv")
    paths += box.rel_path.tolist()
    src["hf_morphology_images"] = len(box)
    if include_mendeley:
        pm = FROZEN / "mendeley_pathmap.local.csv"
        mp = pd.read_csv(pm)
        keep = pd.read_csv(FROZEN / "external_pool_mendeley.csv")
        ok = set(keep.loc[keep.appearance_training_allowed, "image_id"])
        add = mp[mp.image_id.isin(ok)].rel_path.tolist()
        paths += add
        src["mendeley_marker_free"] = len(add)
    return paths, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--include-mendeley", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    # Uniform adaptation budget. Full-backbone adaptation of a ViT-L does not fit
    # in 8.5 GB at a usable speed (measured: 15.2 min / 100 steps, i.e. ~49 h for
    # 8 epochs), so every encoder adapts its last --unfreeze-blocks transformer
    # blocks plus the final norm, for the same --max-steps. Identical budget for
    # every arm is what keeps the arms comparable; it is a smaller intervention
    # than full adaptation and is recorded as such.
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="write a resumable snapshot every N steps; 0 disables")
    ap.add_argument("--resume", action="store_true",
                    help="continue from resume.pth if one is present")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg = ENCODERS[a.encoder]
    tag = a.encoder + ("_da_mend" if a.include_mendeley else "_da")
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    paths, src = load_pool(a.include_mendeley)
    if any("recovered_archive" in p for p in paths):
        raise RuntimeError("a local archive image reached the adaptation pool")
    ds = TwoView(paths, a.size, cfg["norm"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True)

    student, dim, wsha = build_backbone(cfg, device)
    teacher, _, _ = build_backbone(cfg, device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    trainable = freeze_all_but_last(student, cfg["kind"], a.unfreeze_blocks)
    sh, th = Head(dim).to(device), Head(dim).to(device)
    th.load_state_dict(sh.state_dict())
    for p in th.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad] + list(sh.parameters()),
        lr=a.lr, weight_decay=0.04)
    scaler = torch.amp.GradScaler("cuda")
    centre = torch.zeros(1, 4096, device=device)
    t0, step, losses, stop = time.time(), 0, [], False
    first100 = None

    resumed = None
    rp = out / "resume.pth"
    if a.resume and rp.exists():
        snap = torch.load(str(rp), map_location=device, weights_only=False)
        teacher.load_state_dict(snap["state_dict"])
        student.load_state_dict(snap["student_state_dict"])
        sh.load_state_dict(snap["student_head"])
        th.load_state_dict(snap["teacher_head"])
        opt.load_state_dict(snap["optimizer"])
        centre = snap["centre"].to(device)
        step = int(snap["step"])
        losses = list(snap.get("losses", []))
        first100 = snap.get("first_100_loss")
        # The sampler is not restored, so the resumed half draws a fresh shuffle of
        # the same external pool rather than the exact tail of the original order.
        # The total step budget is unchanged and no local data is involved, so this
        # does not affect comparability across arms, but it means the run is not
        # bit-identical to an uninterrupted one.
        resumed = dict(from_step=step,
                       minutes_before_interruption=snap.get("elapsed_minutes"),
                       sampler_state_restored=False,
                       reason="machine shutdown interrupted the original run")
        print("%s resumed from step %d" % (tag, step), flush=True)

    for ep in range(a.epochs):
        if stop:
            break
        for v1, v2 in dl:
            if step >= a.max_steps:
                stop = True
                break
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t = torch.cat([th(embed(teacher, v1, cfg["kind"])),
                                   th(embed(teacher, v2, cfg["kind"]))])
                    tp = F.softmax((t - centre) / a.teacher_temp, dim=-1)
                s = torch.cat([sh(embed(student, v1, cfg["kind"])),
                               sh(embed(student, v2, cfg["kind"]))])
                sl = F.log_softmax(s / a.student_temp, dim=-1)
                n = v1.shape[0]
                # cross view: teacher view1 supervises student view2 and vice versa
                loss = -(tp[:n] * sl[n:]).sum(-1).mean() \
                       - (tp[n:] * sl[:n]).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss / 2).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                centre = 0.9 * centre + 0.1 * t.float().mean(0, keepdim=True)
                # frozen student parameters are identical to the teacher's, so the
                # EMA only moves where adaptation actually happened
                for q, p in zip(teacher.parameters(), student.parameters()):
                    if p.requires_grad:
                        q.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
                for q, p in zip(th.parameters(), sh.parameters()):
                    q.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
            losses.append(float(loss.item()) / 2)
            step += 1
            if step % 100 == 0:
                print("%s ep%d step%d loss %.4f (%.1f min)"
                      % (tag, ep, step, float(np.mean(losses[-100:])),
                         (time.time() - t0) / 60), flush=True)
            # A machine shutdown during MedSigLIP's 2.5 h run cost the whole
            # budget, because the checkpoint was only written at the end. Save a
            # resumable snapshot periodically. It is written to a temporary name
            # and renamed, so an interrupted save cannot leave a truncated file
            # where a later run would read it as valid.
            if a.ckpt_every and step % a.ckpt_every == 0:
                tmp = out / "resume.pth.tmp"
                torch.save(dict(state_dict=teacher.state_dict(),
                                student_state_dict=student.state_dict(),
                                student_head=sh.state_dict(),
                                teacher_head=th.state_dict(),
                                optimizer=opt.state_dict(),
                                centre=centre.cpu(), step=step, epoch=ep,
                                losses=losses[-100:],
                                first_100_loss=(
                                    first100 if first100 is not None
                                    else (round(float(np.mean(losses[:100])), 4)
                                          if len(losses) >= 100 else None)),
                                elapsed_minutes=round((time.time() - t0) / 60, 1),
                                encoder=a.encoder, kind=cfg["kind"],
                                timm_model=cfg["model"], dim=dim), tmp)
                tmp.replace(out / "resume.pth")

    ck = out / "backbone.pth"
    torch.save(dict(state_dict=teacher.state_dict(), encoder=a.encoder,
                    kind=cfg["kind"], timm_model=cfg["model"], dim=dim), ck)
    meta = dict(
        run="external_unlabelled_domain_adaptation", arm_tag=tag,
        encoder=a.encoder, base_arm=cfg["base_arm"], kind=cfg["kind"],
        base_weights=cfg["weights"], base_weights_sha256=wsha,
        checkpoint=str(ck.relative_to(ROOT)).replace("\\", "/"),
        checkpoint_sha256=sha256(ck), feature_dim=int(dim),
        objective="DINO-style self-distillation, two global views, EMA teacher, "
                  "centred + sharpened cross-entropy; head discarded",
        hyperparameters=dict(epochs=a.epochs, steps=step, max_steps=a.max_steps,
                             size=a.size, batch_size=a.batch_size, lr=a.lr,
                             ema=a.ema, teacher_temp=a.teacher_temp,
                             student_temp=a.student_temp,
                             unfreeze_blocks=a.unfreeze_blocks,
                             seed=SEED, precision="bf16 autocast"),
        adapted_parameters=trainable,
        pool=dict(images=len(paths), by_source=src,
                  mendeley_included=bool(a.include_mendeley),
                  mendeley_default_excluded_because="all 576 images carry device "
                      "caliper strokes / contour overlays, so adapting on them "
                      "would teach another device's annotation marks"),
        loss=dict(first_100=(first100 if first100 is not None
                             else round(float(np.mean(losses[:100])), 4)),
                  last_100=round(float(np.mean(losses[-100:])), 4),
                  first_100_is_from_this_process=resumed is None),
        resumed=resumed,
        leakage=dict(local_images_used=0, local_labels_used=0, folds_read=0,
                     locked_cases_seen=0,
                     why_fold_independent="only external images are used, so the "
                         "adapted encoder cannot carry information about any local "
                         "validation case and one checkpoint is valid for all folds"),
        limitations=[
            "adaptation runs at %d square; the deployed geometry does not fit two "
            "views in 8.5 GB. Extraction afterwards still uses the arm's own "
            "deployed geometry, so adaptation and readout resolutions differ."
            % a.size,
            "PARTIAL adaptation: only the last %d blocks plus the final norm move "
            "(%d of %d parameters). Full-backbone adaptation of a ViT-L was "
            "measured at 15.2 min / 100 steps on this 8.5 GB GPU, i.e. ~49 h for "
            "8 epochs, so it is out of budget. A null result is therefore a null "
            "for THIS budget, not for domain adaptation in general."
            % (a.unfreeze_blocks, trainable["trainable_params"],
               trainable["total_params"]),
            "the step budget (%d) is fixed and identical for every encoder so the "
            "arms are comparable; it is not tuned per encoder" % a.max_steps,
            "no hue/saturation jitter was applied, because blood_color and clarity "
            "are downstream targets and colour invariance would remove their signal",
            "external subjects are not our patients; this transfers appearance "
            "statistics only and carries no label information",
        ] + ([] if resumed is None else [
            "this run was RESUMED from step %d after a machine shutdown. The total "
            "step budget matches the other arms, but the data sampler order was not "
            "restored, so the run is not bit-identical to an uninterrupted one and "
            "the reported first_100 loss comes from the original process."
            % resumed["from_step"]]),
        runtime_minutes=round((time.time() - t0) / 60, 1),
        runtime_minutes_this_process_only=resumed is not None)
    (out / "adaptation.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                    indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
