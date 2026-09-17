"""Stage-1 SAM/MedSAM comparison with shared vascular prompts.

This is an audit runner, not a medical segmentation benchmark: there are no
pixel masks.  It freezes 50 development images, keeps multiple loop-level
instances, and records failure signals for later geometric routing.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

FIELDS = ["candidate_count", "kept_count", "coverage", "components", "largest_component_px", "seconds"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def choose_samples(manifest: Path, files: Path, out: Path, n: int = 50) -> list[dict]:
    folds = pd.read_csv(manifest)
    # Development only: locked_evaluation_v1 is deliberately not used here.
    folds = folds[folds["evaluation_role"].eq("development")].sort_values(["fold", "exam_case_id"])
    fm = pd.read_csv(files)
    fm = fm[(fm.role == "cap_image") & (~fm.is_black_placeholder.astype(bool))]
    rows = []
    for _, c in folds.iterrows():
        x = fm[fm.exam_case_id.eq(c.exam_case_id)].sort_values("filename")
        if len(x):
            row = x.iloc[len(x) // 2]
            rows.append({"exam_case_id": c.exam_case_id, "fold": int(c.development_fold), "duplicate_group": c.duplicate_group, "image_path": row.path})
    if n >= len(rows):
        pd.DataFrame(rows).to_csv(out, index=False)
        return rows
    # exactly ten case groups per fold where available, deterministic.
    selected = []
    for fold in sorted({r["fold"] for r in rows}):
        seen = set()
        for r in (r for r in rows if r["fold"] == fold):
            if r["duplicate_group"] in seen: continue
            selected.append(r); seen.add(r["duplicate_group"])
            if len(seen) >= max(1, n // 5): break
    selected = selected[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(out, index=False)
    return selected

def vascular_prompts(rgb: np.ndarray, max_prompts: int = 12) -> tuple[list[list[int]], np.ndarray]:
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.float32); g = rgb[..., 1].astype(np.float32); b = rgb[..., 2].astype(np.float32)
    red = r - 0.5 * (g + b)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    base = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 1]).astype(np.float32)
    ridges = []
    for sigma in (1.0, 2.0, 3.5, 5.0):
        sm = cv2.GaussianBlur(base, (0, 0), sigma)
        xx = cv2.Sobel(sm, cv2.CV_32F, 2, 0); yy = cv2.Sobel(sm, cv2.CV_32F, 0, 2); xy = cv2.Sobel(sm, cv2.CV_32F, 1, 1)
        tr = xx + yy; det = xx * yy - xy * xy; d = np.sqrt(np.maximum(tr * tr * .25 - det, 0))
        l1 = tr * .5 - d; l2 = tr * .5 + d
        ridges.append(np.maximum(-l1, 0) * (np.abs(l1) >= np.abs(l2)))
    vessel = np.max(np.stack(ridges), 0); vessel /= vessel.max() + 1e-6
    source = .55 * (red - cv2.GaussianBlur(red, (0, 0), 19)) + .45 * vessel * 40
    mask = (source >= np.quantile(source, .86)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    n, labc, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in sorted(range(1, n), key=lambda j: int(stats[j, cv2.CC_STAT_AREA]), reverse=True):
        x, y, bw, bh, area = stats[i]
        if area < max(24, int(h * w * .00015)) or bh < int(h * .025): continue
        px, py = max(8, int(bw * .25)), max(8, int(bh * .22))
        boxes.append([max(0, int(x-px)), max(0, int(y-py)), min(w-1, int(x+bw+px)), min(h-1, int(y+bh+py))])
        if len(boxes) >= max_prompts: break
    if not boxes: boxes = [[int(.05*w), int(.15*h), int(.95*w), int(.9*h)]]
    return boxes, mask.astype(bool)

def component_summary(mask: np.ndarray) -> dict:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([], dtype=np.int64)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-5)
    widths = 2. * dt[peaks]
    # These are pixel-domain audit features only. Calibration is intentionally deferred.
    return {"coverage": float(mask.mean()), "components": int((areas >= 16).sum()), "largest_component_px": int(areas.max()) if len(areas) else 0,
            "area_px": int(mask.sum()), "perimeter_px": float(sum(cv2.arcLength(c, True) for c in contours)),
            "width_px_median": float(np.median(widths)) if len(widths) else 0., "width_px_p90": float(np.quantile(widths, .9)) if len(widths) else 0.}

def candidate_score(mask: np.ndarray, rgb: np.ndarray, quality: float, stability: float) -> float:
    if not mask.any(): return -1e9
    red = rgb[..., 0].astype(np.float32) - .5 * (rgb[..., 1].astype(np.float32) + rgb[..., 2].astype(np.float32))
    cov = float(mask.mean()); contrast = float(red[mask].mean() - red.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = int((stats[1:, cv2.CC_STAT_AREA] >= 16).sum()) if n > 1 else 0
    # Tight loop proposals: penalize image-wide fills and fragmented masks.
    return float(quality * max(stability, .01) + .02 * contrast - 1.5 * max(0., cov-.12) - .01 * max(0, comps-4))

def nms_keep(cands: list[dict], rgb: np.ndarray, limit: int = 24) -> list[dict]:
    accepted = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        m = c["mask"]
        if float(m.mean()) < .0001 or float(m.mean()) > .25: continue
        overlap = max((float((m & q["mask"]).sum()) / max(1, int((m | q["mask"]).sum())) for q in accepted), default=0.)
        if overlap > .78: continue
        accepted.append(c)
        if len(accepted) >= limit: break
    return accepted

def overlay(rgb: np.ndarray, masks: list[np.ndarray], color=(0, 255, 128)) -> np.ndarray:
    out = rgb.copy()
    for m in masks:
        out[m] = (.45 * out[m] + .55 * np.asarray(color)).astype(np.uint8)
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, (255, 255, 0), 1)
    return out

def run_sam2(name: str, cfg: str, ckpt: Path, samples: list[dict], image_root: Path, out: Path) -> dict:
    sys.path.insert(0, "/root/autodl-tmp/sam2_source_2b90b9f5")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg, str(ckpt), device="cuda").eval(); pred = SAM2ImagePredictor(model)
    return run_predictor(name, pred, samples, image_root, out, mode="sam2")

def run_predictor(name: str, pred, samples: list[dict], image_root: Path, out: Path, mode: str) -> dict:
    (out / name / "masks").mkdir(parents=True, exist_ok=True); (out / name / "overlays").mkdir(parents=True, exist_ok=True); (out / name / "instances").mkdir(parents=True, exist_ok=True)
    rows=[]; start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for s in samples:
        path=image_root / s["image_path"]; bgr=cv2.imread(str(path)); rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); boxes, tradition=vascular_prompts(rgb)
        pred.set_image(rgb); cands=[]; t=time.perf_counter()
        for box in boxes:
            x1,y1,x2,y2=box; pt=np.array([[(x1+x2)/2,(y1+y2)/2]], np.float32); labels=np.ones(1,np.int32)
            if mode == "sam2": masks, scores, logits = pred.predict(point_coords=pt, point_labels=labels, box=np.asarray(box,np.float32), multimask_output=True)
            else: masks, scores, logits = pred.predict(point_coords=pt, point_labels=labels, box=np.asarray(box,np.float32), multimask_output=False)
            for i,m in enumerate(masks):
                m=np.asarray(m).astype(bool); q=float(scores[i]) if np.ndim(scores) else float(scores)
                stab=float(np.mean((logits[i] > 0) == (logits[i] > .05))) if np.ndim(logits) else 1.0
                cands.append({"mask":m,"score":candidate_score(m,rgb,q,stab),"quality":q,"stability":stab,"box":box,"candidate_index":i})
        kept=nms_keep(cands,rgb); union=np.zeros(rgb.shape[:2],bool)
        for c in kept: union |= c["mask"]
        stem=s["exam_case_id"].replace('/','__')+'__'+Path(s["image_path"]).stem
        np.save(out/name/"masks"/(stem+".npy"),union); cv2.imwrite(str(out/name/"overlays"/(stem+".jpg")),cv2.cvtColor(overlay(rgb,[c["mask"] for c in kept]),cv2.COLOR_RGB2BGR))
        # Bit-packed masks preserve the multi-instance set without storing 24 full planes per image.
        planes = np.stack([c["mask"] for c in kept], axis=0) if kept else np.zeros((0, *union.shape), dtype=bool)
        np.savez_compressed(out/name/"instances"/(stem+".npz"), masks=np.packbits(planes, axis=2), height=union.shape[0], width=union.shape[1],
                            quality=np.asarray([c["quality"] for c in kept]), stability=np.asarray([c["stability"] for c in kept]),
                            score=np.asarray([c["score"] for c in kept]), boxes=np.asarray([c["box"] for c in kept], dtype=np.int16))
        rows.append({**s,"method":name,"prompt_count":len(boxes),"candidate_count":len(cands),"kept_count":len(kept),"seconds":time.perf_counter()-t,**component_summary(union)})
    pd.DataFrame(rows).to_csv(out/name/"metrics.csv",index=False)
    return {"method":name,"samples":len(rows),"seconds":time.perf_counter()-start,"peak_gpu_mb":round(torch.cuda.max_memory_allocated()/2**20,1)}

def run_traditional(samples, image_root, out):
    (out/"traditional"/"overlays").mkdir(parents=True,exist_ok=True); (out/"traditional"/"masks").mkdir(parents=True,exist_ok=True); rows=[]
    for s in samples:
        path=image_root/s["image_path"]; rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB); boxes,m=vascular_prompts(rgb); masks=[]
        n, lab, stats, _=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]>=16: masks.append(lab==i)
        stem=s["exam_case_id"].replace('/','__')+'__'+Path(s["image_path"]).stem; np.save(out/"traditional"/"masks"/(stem+".npy"),m)
        cv2.imwrite(str(out/"traditional"/"overlays"/(stem+".jpg")),cv2.cvtColor(overlay(rgb,masks,(0,128,255)),cv2.COLOR_RGB2BGR)); rows.append({**s,"method":"traditional","prompt_count":len(boxes),"candidate_count":len(masks),"kept_count":len(masks),**component_summary(m)})
    pd.DataFrame(rows).to_csv(out/"traditional"/"metrics.csv",index=False)

def build_audit(samples, out: Path) -> None:
    """A browser-friendly, one-row-per-development-image review sheet."""
    rows = ["<html><head><meta charset='utf-8'><style>body{font-family:Arial;background:#eee} .r{display:grid;grid-template-columns:180px repeat(5,256px);gap:6px;margin:10px;padding:8px;background:#fff}img{width:256px;height:192px;object-fit:contain;background:#222}.id{font-size:12px}</style></head><body><h1>Stage 1 segmentation audit: development images only</h1>"]
    labels = ["Original", "Traditional", "SAM2 Tiny", "SAM2 Base+", "MedSAM"]
    for s in samples:
        stem=s["exam_case_id"].replace('/','__')+'__'+Path(s["image_path"]).stem
        original="../../../data768/"+s["image_path"]
        imgs=[original, f"traditional/overlays/{stem}.jpg", f"sam2_tiny/overlays/{stem}.jpg", f"sam2_base_plus/overlays/{stem}.jpg", f"medsam/overlays/{stem}.jpg"]
        cells="".join(f"<div><small>{lab}</small><br><img src='{src}'></div>" for lab,src in zip(labels,imgs))
        rows.append(f"<section class='r'><div class='id'><b>{s['exam_case_id']}</b><br>fold {s['fold']}<br>{s['image_path']}</div>{cells}</section>")
    rows.append("</body></html>"); (out/"audit_index.html").write_text("\n".join(rows),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/nailfold")); ap.add_argument("--output",type=Path,default=None); ap.add_argument("--manifest",type=Path,default=None); ap.add_argument("--count",type=int,default=50); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    root=args.root; out=args.output or (root/"artifacts/experiments/sam_stage2"); out.mkdir(parents=True,exist_ok=True)
    manifest= args.manifest or (out/f"sample_manifest_{args.count}.csv");
    if not manifest.exists(): choose_samples(root/"code_latest/artifacts/manifest/locked_evaluation_v1.csv",root/"code_latest/artifacts/manifest/files.csv",manifest,args.count)
    samples=pd.read_csv(manifest).to_dict("records"); samples=samples[:12] if args.smoke else samples
    (out/"run_input_manifest.csv").write_text(pd.DataFrame(samples).to_csv(index=False),encoding="utf-8")
    image_root = root / "data768"
    summaries=[]; run_traditional(samples,image_root,out)
    base= root/"artifacts/experiments/sam_round1/checkpoints/sam2.1_hiera_base_plus.official.pt"; tiny=root/"artifacts/experiments/sam_stage2/checkpoints/sam2.1_hiera_tiny.pt"
    # The official Tiny checkpoint is roughly 150 MiB; do not load a live partial download.
    if tiny.exists() and tiny.stat().st_size > 150_000_000:
        summaries.append(run_sam2("sam2_tiny","configs/sam2.1/sam2.1_hiera_t.yaml",tiny,samples,image_root,out))
    summaries.append(run_sam2("sam2_base_plus","configs/sam2.1/sam2.1_hiera_b+.yaml",base,samples,image_root,out))
    # MedSAM uses the same boxes; its official source is isolated from SAM2 import.
    sys.path.insert(0,"/root/autodl-tmp/medsam_source_d71e8a1a"); from segment_anything import sam_model_registry
    m=sam_model_registry["vit_b"](checkpoint=str(root/"artifacts/experiments/sam_round1/checkpoints/medsam/medsam_vit_b.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    summaries.append(run_predictor("medsam",SamPredictor(m),samples,image_root,out,mode="medsam"))
    build_audit(samples, out)
    (out/"run_metadata.json").write_text(json.dumps({
        "source_sam2": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "source_medsam": "d71e8a1a99ad751840a22a7fa3ecfb4166fb1488",
        "sam2_tiny_sha256": sha256(tiny), "sam2_base_plus_sha256": sha256(base),
        "medsam_sha256": sha256(root/"artifacts/experiments/sam_round1/checkpoints/medsam/medsam_vit_b.pth"),
        "prompt_strategy": "deterministic red-excess plus multiscale ridge components; up to 12 tight boxes with a center positive point",
        "candidate_filter": "score = predicted quality * stability + red contrast; retain 0.01%-25% coverage and NMS IoU <= 0.78, maximum 24 instances",
        "instance_encoding": "np.savez_compressed; masks are np.packbits along width, restore with np.unpackbits(..., axis=2)[:, :, :width]",
        "samples": len(samples), "sample_manifest": str(manifest), "summaries": summaries,
        "input_reports": False, "locked_set_used": False,
        "reproduce": "venv/bin/python artifacts/experiments/sam_stage2/stage1_segmentation_audit.py"
    }, indent=2),encoding="utf-8")

if __name__ == "__main__": main()
