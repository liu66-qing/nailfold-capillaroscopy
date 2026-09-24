#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract frozen features for the medical-encoder transfer experiment.

Four arms, one preprocessing rule, one readout rule (protocol.yaml):
  anchor_dinov2b_deployed  general DINOv2 ViT-B/14   -- reproduces the deployment
  dinov2b_newloader        same weights, this loader -- loader control
  dinov2l_capacity         general DINOv2 ViT-L/14   -- capacity control
  biomedclip_medical       BiomedCLIP ViT-B/16 image tower -- medical pretraining

Why a loader control. The deployed features were produced by a different script on
a different machine. Without an arm that runs the SAME weights through THIS code,
any difference could be the loader rather than the pretraining, and the comparison
would be uninterpretable.

Geometry is identical for every arm: aspect-preserving resize, then pad to the
encoder's own supported square with that encoder's normalisation-neutral value.
Padding is placed bottom-right and the padded region is recorded so pad patches can
be excluded from patch pooling. No colour or sharpness transform is applied at all,
because blood_color and clarity are prediction targets.

DEVELOPMENT ONLY. Cases come from development_fold non-NaN; the run aborts if a
locked-47 id appears. Report scans (rep_*) are never encoder inputs.

Run (local RTX 5070, ~8 GB):
  PYTHONIOENCODING=utf-8 python scripts/extract_medical_encoders.py --arm biomedclip_medical
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from numpy.lib.format import open_memmap
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
INDEX = ROOT / ".tmp_probe" / "feat" / "dinov2" / "index.csv"
LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
IMAGE_ROOT = ROOT / "data"

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073),
               (0.26862954, 0.26130258, 0.27577711))

ARMS = {
    # geometry="deployed" reproduces extract_dinov2_spatial.py's `native` preset:
    # a direct resize to 518x686, no padding. Without this the anchor and the
    # loader control would be byte-identical and the control would test nothing.
    "anchor_dinov2b_deployed": dict(
        kind="timm", model="vit_base_patch14_dinov2.lvd142m",
        weights="weights/dinov2/b/model.safetensors",
        size=518, patch=14, norm=IMAGENET, role="anchor",
        geometry="deployed", deployed_res=(518, 686),
        pretraining="general LVD-142M"),
    "dinov2b_newloader": dict(
        kind="timm", model="vit_base_patch14_dinov2.lvd142m",
        weights="weights/dinov2/b/model.safetensors",
        size=518, patch=14, norm=IMAGENET, role="loader_control",
        pretraining="general LVD-142M"),
    "dinov2l_capacity": dict(
        kind="timm", model="vit_large_patch14_dinov2.lvd142m",
        weights="weights/dinov2/l/model.safetensors",
        size=518, patch=14, norm=IMAGENET, role="capacity_control",
        pretraining="general LVD-142M"),
    # Round 1b: ViT-L at the DEPLOYED geometry, so a re-check under the shipped
    # readout (5 poolings, fixed C=0.03, PCA 64) varies capacity alone against
    # anchor_dinov2b_deployed. Without this the comparison would still carry the
    # loader change that geometry_only showed to be null but not proven harmless.
    "dinov2l_deployed_geometry": dict(
        kind="timm", model="vit_large_patch14_dinov2.lvd142m",
        weights="weights/dinov2/l/model.safetensors",
        size=518, patch=14, norm=IMAGENET, role="capacity_control_deployed_geom",
        geometry="deployed", deployed_res=(518, 686),
        pretraining="general LVD-142M"),
    "biomedclip_medical": dict(
        kind="biomedclip", model="vit_base_patch16_224",
        weights="甲劈微循环hf_biomedclip/open_clip_pytorch_model.bin",
        size=224, patch=16, norm=OPENAI_CLIP, role="medical_candidate",
        pretraining="medical PMC-15M"),
    # Round two, unblocked 2026-09-23 after the user accepted the licences on
    # both gated pages. Before this the two repos returned 403 "not in the
    # authorized list", which is blocked_access, not a negative result.
    #
    # RETFound at the DEPLOYED geometry. Its config.json claims patch 16 /
    # image 224, but the published weights say otherwise: patch_embed.proj is
    # (1024,3,14,14) and pos_embed is (1,1370,1024) = 1 + 37x37, i.e. DINOv2
    # ViT-L/14 at 518. The card is wrong; the tensors decide. That it is ViT-L/14
    # at 518 is convenient rather than lucky: it means RETFound can be compared
    # against dinov2l_deployed_geometry with capacity, patch size and input
    # geometry all held fixed, so the only difference left is the pretraining
    # data (retinal CFP vs LVD-142M).
    "retfound_dinov2_meh": dict(
        kind="retfound", model="vit_large_patch14_dinov2.lvd142m",
        weights="weights/RETFound_dinov2_meh/RETFound_dinov2_meh.pth",
        size=518, patch=14, norm=IMAGENET, role="medical_candidate",
        geometry="deployed", deployed_res=(518, 686),
        pretraining="retinal CFP, MEH AlzEye, DINOv2 SSL",
        licence="CC BY-NC 4.0 (non-commercial: research-period control only)"),
    # MedSigLIP vision tower only. The text tower is downloaded but never used:
    # the standing constraint is image embeddings only, no zero-shot text
    # prediction. SigLIP has no CLS token and a fixed 1024-position embedding
    # (32x32 patches at 448), so it cannot run at the deployed 518x686; it is
    # evaluated at its own native square, and that difference is recorded rather
    # than hidden, because it means this arm is not a pure pretraining contrast.
    "medsiglip_medical": dict(
        kind="medsiglip", model="siglip_vision",
        weights="weights/medsiglip-448/model.safetensors",
        size=448, patch=14, norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        role="medical_candidate",
        pretraining="medical multimodal (HAI-DEF MedSigLIP)",
        licence="health-ai-developer-foundations",
        note="vision tower only; no text tower, no zero-shot"),

    # ---- rescue round: arms whose backbone was domain-adapted on EXTERNAL
    # unlabelled nailfold images (adapt_encoder_domain_external.py). The base
    # weights, geometry, readout and pooling set are unchanged, so the only
    # difference against the matching un-adapted arm is the adaptation itself.
    # Adaptation read no local image, label or fold, so one checkpoint is valid
    # for every outer fold and cannot carry validation-case information.
    "dinov2b_da_external": dict(
        kind="adapted", model="vit_base_patch14_dinov2.lvd142m",
        weights="artifacts/experiments/rescue_external_20260922/adapted/"
                "dinov2b_da/backbone.pth",
        size=518, patch=14, norm=IMAGENET, role="domain_adapted",
        geometry="deployed", deployed_res=(518, 686),
        base_arm="anchor_dinov2b_deployed",
        pretraining="general LVD-142M + external nailfold SSL adaptation"),
    "dinov2l_da_external": dict(
        kind="adapted", model="vit_large_patch14_dinov2.lvd142m",
        weights="artifacts/experiments/rescue_external_20260922/adapted/"
                "dinov2l_da/backbone.pth",
        size=518, patch=14, norm=IMAGENET, role="domain_adapted",
        geometry="deployed", deployed_res=(518, 686),
        base_arm="dinov2l_deployed_geometry",
        pretraining="general LVD-142M + external nailfold SSL adaptation"),
    "retfound_da_external": dict(
        kind="adapted", model="vit_large_patch14_dinov2.lvd142m",
        weights="artifacts/experiments/rescue_external_20260922/adapted/"
                "retfound_da/backbone.pth",
        size=518, patch=14, norm=IMAGENET, role="domain_adapted",
        geometry="deployed", deployed_res=(518, 686),
        base_arm="retfound_dinov2_meh",
        licence="CC BY-NC 4.0 (non-commercial: research-period control only)",
        pretraining="retinal CFP DINOv2 + external nailfold SSL adaptation"),
    "medsiglip_da_external": dict(
        kind="adapted_siglip", model="siglip_vision",
        weights="artifacts/experiments/rescue_external_20260922/adapted/"
                "medsiglip_da/backbone.pth",
        size=448, patch=14, norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        role="domain_adapted", base_arm="medsiglip_medical",
        licence="health-ai-developer-foundations",
        pretraining="MedSigLIP + external nailfold SSL adaptation",
        note="vision tower only; no text tower, no zero-shot"),
}
POOLINGS = ["cls", "mean", "max", "topk_mean", "std"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(path: Path, size: int, norm, deployed_res=None) -> tuple:
    """Aspect-preserving resize, then pad to size x size with the neutral value.

    Neutral means the value that maps to 0 after normalisation, so the padding
    contributes no signal. Returns the tensor and the occupied (h, w) so pad
    patches can be dropped from patch statistics.

    deployed_res reproduces the shipped `native` preset instead: a direct resize
    to (518, 686) with no padding, so the anchor arm matches deployment and the
    loader control isolates the geometry change rather than duplicating it.
    """
    if deployed_res is not None:
        with Image.open(path) as im:
            im = im.convert("RGB")
            t = TF.resize(im, list(deployed_res),
                          interpolation=TF.InterpolationMode.BICUBIC)
            t = TF.normalize(TF.to_tensor(t), *norm)
        return t, (deployed_res[0], deployed_res[1])
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        t = TF.resize(im, [nh, nw], interpolation=TF.InterpolationMode.BICUBIC)
        t = TF.normalize(TF.to_tensor(t), *norm)
    canvas = torch.zeros(3, size, size)          # zero == neutral after normalise
    canvas[:, :nh, :nw] = t
    return canvas, (nh, nw)


def build(cfg: dict, device: torch.device, img_size=None):
    import timm
    from safetensors.torch import load_file
    wpath = ROOT / cfg["weights"]
    if cfg["kind"] == "timm":
        # Always instantiate at the checkpoint's own img_size so pos_embed loads
        # without a shape mismatch. dynamic_img_size=True then interpolates
        # pos_embed at forward time, which is how the deployed extractor reaches
        # 518x686 as well; the grid is therefore computed from the actual input.
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=0,
                                  img_size=cfg["size"],
                                  dynamic_img_size=True)
        sd = load_file(str(wpath))
        sd.pop("mask_token", None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    elif cfg["kind"] == "retfound":
        # DINOv2-SSL checkpoint: the usable encoder is teacher.backbone.*, next
        # to dino_head/ibot_head which are SSL projection heads and are dropped.
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=0,
                                  img_size=cfg["size"], dynamic_img_size=True)
        blob = torch.load(str(wpath), map_location="cpu", weights_only=False)
        t = blob["teacher"]
        pre = "backbone."
        sd = {k[len(pre):]: v for k, v in t.items() if k.startswith(pre)}
        if not sd:
            raise RuntimeError("no teacher.backbone.* keys in %s" % wpath)
        sd.pop("mask_token", None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    elif cfg["kind"] == "medsiglip":
        from transformers import SiglipVisionModel
        # Vision tower only, loaded from the local snapshot. The text tower is
        # present in the file and deliberately left unloaded.
        model = SiglipVisionModel.from_pretrained(
            str(wpath.parent), dtype=torch.float32)
        return model.to(device).eval(), sha256(wpath)
    elif cfg["kind"] == "adapted":
        # domain-adapted backbone: the EMA teacher saved by the adaptation run.
        # Its architecture and geometry are identical to the base arm, so the
        # comparison isolates the adaptation.
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=0,
                                  img_size=cfg["size"], dynamic_img_size=True)
        blob = torch.load(str(wpath), map_location="cpu", weights_only=False)
        sd = blob["state_dict"]
        sd.pop("mask_token", None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    elif cfg["kind"] == "adapted_siglip":
        from transformers import SiglipVisionModel
        base = ROOT / "weights" / "medsiglip-448"
        model = SiglipVisionModel.from_pretrained(str(base), dtype=torch.float32)
        blob = torch.load(str(wpath), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError("adapted siglip mismatch missing=%s unexpected=%s"
                               % (missing[:6], unexpected[:6]))
        return model.to(device).eval(), sha256(wpath)
    else:
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=0)
        blob = torch.load(str(wpath), map_location="cpu", weights_only=True)
        pre = "visual.trunk."
        sd = {k[len(pre):]: v for k, v in blob.items() if k.startswith(pre)}
        if not sd:
            raise RuntimeError("no visual.trunk.* keys in %s" % wpath)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "mask_token" not in m]
    unexpected = [u for u in unexpected if "mask_token" not in u]
    if missing or unexpected:
        raise RuntimeError("state mismatch missing=%s unexpected=%s"
                           % (missing[:6], unexpected[:6]))
    return model.to(device).eval(), sha256(wpath)
def load_index() -> pd.DataFrame:
    """Development cases only, with a hard abort if a locked id appears."""
    ix = pd.read_csv(INDEX, dtype={"exam_case_id": str})
    lab = pd.read_csv(LABELS, dtype={"exam_case_id": str})
    dev = set(lab.loc[lab.development_fold.notna(), "exam_case_id"])
    locked = set(lab.loc[lab.development_fold.isna(), "exam_case_id"])
    if len(locked) != 47:
        raise RuntimeError("expected 47 locked cases, got %d" % len(locked))
    leaked = set(ix.exam_case_id) & locked
    if leaked:
        raise RuntimeError("locked case in extraction index: %s" % sorted(leaked)[:5])
    ix = ix[ix.exam_case_id.isin(dev)].reset_index(drop=True)
    if ix.exam_case_id.nunique() != 186:
        raise RuntimeError("expected 186 development cases, got %d"
                           % ix.exam_case_id.nunique())
    bad = [p for p in ix.image_path if Path(p).name.lower().startswith("rep")]
    if bad:
        raise RuntimeError("report scan reached the encoder index: %s" % bad[:3])
    return ix


def pool(tokens: torch.Tensor, npre: int, grid: tuple, occ: list,
         patch: int, topk: int) -> dict:
    """Global vector plus patch statistics, with padded patches excluded.

    Round one uses the global vector as the protocol's readout; the patch
    statistics are stored so the existing five-pooling anchor stays comparable
    without a second forward pass.
    """
    cls = tokens[:, 0].float()
    pt = tokens[:, npre:].float()
    gh, gw = grid
    out = {"cls": cls}
    means, maxs, topks, stds = [], [], [], []
    for i in range(pt.shape[0]):
        ph = max(1, min(gh, int(np.ceil(occ[i][0] / patch))))
        pw = max(1, min(gw, int(np.ceil(occ[i][1] / patch))))
        grid_i = pt[i].reshape(gh, gw, -1)[:ph, :pw].reshape(-1, pt.shape[-1])
        kk = min(topk, grid_i.shape[0])
        means.append(grid_i.mean(0))
        maxs.append(grid_i.max(0).values)
        topks.append(grid_i.topk(kk, dim=0).values.mean(0))
        stds.append(grid_i.std(0) if grid_i.shape[0] > 1
                    else torch.zeros_like(grid_i[0]))
    out["mean"] = torch.stack(means)
    out["max"] = torch.stack(maxs)
    out["topk_mean"] = torch.stack(topks)
    out["std"] = torch.stack(stds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    cfg = ARMS[a.arm]
    ix = load_index()
    out = OUT_ROOT / "features" / a.arm
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    dep = cfg.get("deployed_res")
    model, wsha = build(cfg, device, img_size=list(dep) if dep else None)
    if cfg["kind"] in ("medsiglip", "adapted_siglip"):
        # SigLIP exposes no num_features / patch_embed.grid_size and has no CLS
        # token: its position embedding is a fixed 1024 = 32x32 grid at 448, so
        # it runs only at its own native square.
        dim = int(model.config.hidden_size)
        g = cfg["size"] // cfg["patch"]
        grid = (g, g)
    else:
        dim = model.num_features
        # With dynamic_img_size the token grid follows the actual input, not the
        # checkpoint's square default, so derive it from this arm's geometry.
        grid = ((dep[0] // cfg["patch"], dep[1] // cfg["patch"]) if dep
                else tuple(model.patch_embed.grid_size))

    n = len(ix)
    sinks = {k: open_memmap(out / ("features_%s.npy" % k), mode="w+",
                            dtype=np.float16, shape=(n, dim)) for k in POOLINGS}

    with torch.inference_mode():
        for s in range(0, n, a.batch_size):
            rows = ix.image_path.iloc[s:s + a.batch_size].tolist()
            prepped = [preprocess(IMAGE_ROOT / r, cfg["size"], cfg["norm"], dep)
                       for r in rows]
            x = torch.stack([p[0] for p in prepped]).to(device)
            occ = [p[1] for p in prepped]
            if cfg["kind"] in ("medsiglip", "adapted_siglip"):
                # No CLS token, so npre=0 and the "cls" pooling slot is filled
                # by SigLIP's own pooled output (attention pooling head), which
                # is that family's equivalent global vector. Recorded in the
                # metadata so the slot is not mistaken for a real CLS token.
                o = model(pixel_values=x)
                tok = o.last_hidden_state
                vals = pool(tok, 0, grid, occ, cfg["patch"], a.topk)
                vals["cls"] = o.pooler_output
            else:
                tok = model.forward_features(x)
                vals = pool(tok, model.num_prefix_tokens, grid, occ,
                            cfg["patch"], a.topk)
            e = s + len(rows)
            for k, v in vals.items():
                sinks[k][s:e] = v.cpu().numpy().astype(np.float16)
            if (s // a.batch_size) % 25 == 0:
                print("%s %d/%d" % (a.arm, e, n), flush=True)
    for v in sinks.values():
        v.flush()

    ix[["exam_case_id", "image_path"]].to_csv(out / "index.csv", index=False)
    meta = dict(
        schema_version="medical-encoder-features/1.0",
        arm=a.arm, role=cfg["role"], pretraining=cfg["pretraining"],
        timm_model=cfg["model"], input_size=cfg["size"], patch=cfg["patch"],
        patch_grid=list(grid), feature_dim=int(dim),
        normalisation=dict(mean=list(cfg["norm"][0]), std=list(cfg["norm"][1])),
        geometry=("direct resize to %s, no padding (reproduces the deployed "
                  "native preset)" % list(dep) if dep
                  else "aspect-preserving resize then neutral pad, bottom-right"),
        pad_patches_excluded_from_patch_stats=True,
        poolings=POOLINGS, topk=a.topk,
        weights=cfg["weights"], weights_sha256=wsha,
        cases=int(ix.exam_case_id.nunique()), images=n,
        locked_cases_seen=0,
        licence=cfg.get("licence"),
        arm_notes=cfg.get("note"),
        limitations=[
            "development features only; this run did not read locked-47",
            "frozen encoder statistics, not learned detectors",
            "readout differs across model families, so this is a system "
            "comparison and does not isolate pretraining as a cause",
            "no accuracy claim is made by this file",
        ] + ([
            "RETFound's config.json states patch 16 / image 224, but the "
            "published tensors are patch 14 with pos_embed 1+37x37, i.e. "
            "ViT-L/14 at 518. The tensors were followed, not the card.",
            "CC BY-NC 4.0: non-commercial. Research-period control only; it "
            "cannot ship in a product.",
            "this checkpoint is a CONTINUED DINOv2 SSL run, not independent "
            "pretraining: cosine(patch_embed, the DINOv2-L this project "
            "extracts with) = 0.955, and a shuffle control gives -0.0017, so "
            "the ancestry is established. It does NOT follow that the encoder "
            "is nearly unchanged -- relative Frobenius distance on that same "
            "tensor is 0.495 and attention cosine falls to 0.165 by block 23. "
            "See encoder_distance/retfound_vs_dinov2l.json. Consequence for "
            "attribution: this arm is a domain-of-continuation contrast, not a "
            "medical-versus-general-pretraining contrast.",
        ] if cfg["kind"] == "retfound" else []) + ([
            "this backbone was domain-adapted on EXTERNAL unlabelled nailfold "
            "images only. No local image, label or fold was read during "
            "adaptation, so one checkpoint is valid for every outer fold.",
            "adaptation ran at a smaller square than this extraction geometry; "
            "see the arm's adaptation.json for the exact resolution and steps.",
            "the base arm is %s; only the adaptation differs, so a difference "
            "against that arm is attributable to the adaptation"
            % cfg.get("base_arm"),
        ] if cfg["kind"] in ("adapted", "adapted_siglip") else []) + ([
            "RETFound lineage: CC BY-NC 4.0, non-commercial. Research-period "
            "control only; it cannot ship in a product.",
        ] if cfg.get("base_arm") == "retfound_dinov2_meh" else []) + ([
            "SigLIP has no CLS token; the 'cls' pooling slot holds its "
            "attention-pooled output instead, which is a different object.",
            "fixed 1024-position embedding (32x32 at 448) means this arm "
            "cannot run at the deployed 518x686, so it is NOT a pure "
            "pretraining contrast -- geometry differs too.",
            "vision tower only; the text tower is never loaded and no "
            "zero-shot text prediction is made.",
        ] if cfg["kind"] in ("medsiglip", "adapted_siglip") else []))
    (out / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
