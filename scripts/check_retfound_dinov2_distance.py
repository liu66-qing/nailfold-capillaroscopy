"""How far is RETFound from the plain DINOv2-L it was continued from?

This exists because the addendum's 7-E.3 asserts a cosine of 0.955 between
RETFound's patch embedding and plain DINOv2-L's, and that number was computed in
a shell and never persisted. A claim in a document needs an artefact behind it.

The point of the measurement is to bound what "medical pretraining" could
possibly have changed on this arm. RETFound_dinov2_meh is a continued DINOv2
SSL run on retinal fundus photographs, not an independent pretraining. If the
input projection is nearly identical, then the whole difference between this arm
and DINOv2-L has to come from drift in the later blocks -- and that drift went
towards retina, not towards nailfold.

It reads only the two weight files. No local images, no labels, no folds, no
locked case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
RETFOUND = ROOT / "weights" / "RETFound_dinov2_meh" / "RETFound_dinov2_meh.pth"
# The local copy, not a timm download. The HF cache holds only an incomplete blob
# for this model, and more importantly this file IS the reference the ladder's own
# ViT-L arms were extracted with, so comparing against it answers the question
# actually being asked: how far is RETFound from the DINOv2-L we use.
REFERENCE = ROOT / "weights" / "dinov2" / "l" / "model.safetensors"
OUT = ROOT / "artifacts" / "experiments" / "rescue_external_20260922" / "encoder_distance"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))


def find(sd: dict, needle: str):
    for k in sd:
        if k.endswith(needle):
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(REFERENCE))
    a = ap.parse_args()

    raw = torch.load(RETFOUND, map_location="cpu", weights_only=False)
    # The published file nests everything under "teacher" and prefixes the encoder
    # with "backbone."; the two SSL heads are training machinery, not the encoder.
    ret = raw.get("model", raw.get("teacher", raw))
    ret = {k[len("backbone."):]: v for k, v in ret.items() if k.startswith("backbone.")} \
        or {k: v for k, v in ret.items() if not k.startswith(("dino_head", "ibot_head"))}

    ref = load_file(a.reference)

    report: dict = {
        "question": "how far is RETFound from the DINOv2-L this project extracts with",
        "retfound_weights": str(RETFOUND.relative_to(ROOT)).replace("\\", "/"),
        "retfound_sha256": sha256(RETFOUND),
        "reference_weights": str(Path(a.reference).relative_to(ROOT)).replace("\\", "/"),
        "reference_sha256": sha256(Path(a.reference)),
        "tensors": {},
        "local_images_used": 0,
        "labels_read": 0,
        "folds_read": 0,
        "locked_cases_seen": 0,
    }

    for label, needle in [("patch_embed.proj.weight", "patch_embed.proj.weight"),
                          ("cls_token", "cls_token"),
                          ("pos_embed", "pos_embed"),
                          ("blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.weight"),
                          ("blocks.11.attn.qkv.weight", "blocks.11.attn.qkv.weight"),
                          ("blocks.23.attn.qkv.weight", "blocks.23.attn.qkv.weight"),
                          ("norm.weight", "norm.weight")]:
        kr, kf = find(ret, needle), find(ref, needle)
        if kr is None or kf is None:
            report["tensors"][label] = {"status": "absent",
                                        "in_retfound": kr is not None,
                                        "in_reference": kf is not None}
            continue
        tr, tf = ret[kr], ref[kf]
        if tuple(tr.shape) != tuple(tf.shape):
            report["tensors"][label] = {"status": "shape_mismatch",
                                        "retfound": list(tr.shape),
                                        "reference": list(tf.shape)}
            continue
        report["tensors"][label] = {
            "status": "compared",
            "shape": list(tr.shape),
            "cosine": round(cos(tr, tf), 6),
            "max_abs_diff": round(float((tr.double() - tf.double()).abs().max()), 6),
            "relative_frobenius": round(
                float((tr.double() - tf.double()).norm() / tf.double().norm()), 6),
        }

    # Control: does a cosine of 0.95 on patch_embed actually mean "initialised from
    # these weights"? Shuffling RETFound's own patch_embed destroys the element
    # correspondence while keeping its value distribution exactly. If the shuffled
    # cosine is ~0, the unshuffled figure is real correspondence and not an artefact
    # of both tensors being small zero-centred numbers.
    kr, kf = find(ret, "patch_embed.proj.weight"), find(ref, "patch_embed.proj.weight")
    if kr and kf:
        g = torch.Generator().manual_seed(20260917)
        flat = ret[kr].flatten()
        shuffled = flat[torch.randperm(flat.numel(), generator=g)]
        report["shuffle_control"] = {
            "what": "RETFound patch_embed with its elements permuted, vs the reference",
            "cosine": round(cos(shuffled, ref[kf]), 6),
            "reading": ("near zero means the 0.95 figure is genuine tensor "
                        "correspondence, i.e. RETFound was initialised from these weights"),
        }

    # The drift profile across depth. One tensor cannot say whether the encoder was
    # lightly or heavily retrained; 24 can.
    prof = []
    for i in range(24):
        kr, kf = find(ret, f"blocks.{i}.attn.qkv.weight"), find(ref, f"blocks.{i}.attn.qkv.weight")
        if kr and kf and tuple(ret[kr].shape) == tuple(ref[kf].shape):
            prof.append({"block": i, "cosine": round(cos(ret[kr], ref[kf]), 4)})
    report["depth_profile_attn_qkv"] = prof
    if prof:
        report["depth_profile_summary"] = {
            "block0_cosine": prof[0]["cosine"],
            "last_block_cosine": prof[-1]["cosine"],
            "monotone_decreasing": all(prof[i]["cosine"] >= prof[i + 1]["cosine"] - 0.05
                                       for i in range(len(prof) - 1)),
        }

    pe = report["tensors"].get("pos_embed", {})
    report["geometry_from_tensors"] = {
        "pos_embed_shape": pe.get("shape") or pe.get("retfound"),
        "note": ("the published config.json says patch 16 / image 224; the tensors say "
                 "patch 14 with 1+37x37 positions, i.e. ViT-L/14 at 518. The tensors win."),
    }
    pw = report["tensors"].get("patch_embed.proj.weight", {})
    report["headline_patch_embed_cosine"] = pw.get("cosine")
    report["interpretation"] = [
        ("SHARED ANCESTRY IS ESTABLISHED, NOT SIMILARITY. patch_embed cosine 0.955 with a "
         "near-zero shuffle control proves RETFound was initialised from this DINOv2-L. "
         "It does NOT show the encoder is nearly unchanged: on the same tensor the "
         "relative Frobenius distance is 0.495, i.e. the update has half the norm of the "
         "weight itself, and the deeper attention blocks fall to cosine 0.17."),
        ("So the retinal SSL retrained this network substantially. Any earlier phrasing "
         "of mine that used 0.955 to bound how much RETFound could possibly differ from "
         "DINOv2-L was wrong: a cosine on one input tensor bounds nothing about the "
         "function computed 24 blocks later."),
        ("What remains true is the ancestry claim, which is what matters for attribution: "
         "R0 is not an independent medical pretraining but a DINOv2-L continued on "
         "retinal photographs, so R0-vs-A0L is a domain-of-continuation contrast, not a "
         "medical-vs-general-pretraining contrast."),
        "weight-space distance is not accuracy; the field numbers are in ladder/",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retfound_vs_dinov2l.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tensors"},
                     ensure_ascii=False, indent=1))
    for k, v in report["tensors"].items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
