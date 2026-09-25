"""Extract the locked-47 features the frozen candidate needs. THE LOCKED READ.

This is the one authorised read. It runs the frozen encoder and the frozen external
detector over the locked cases' capillary images and writes features. It computes no
metric, fits nothing, and selects nothing -- scoring is a separate script, so this
step cannot be rerun with a different configuration in response to a number.

Preconditions asserted before any image is opened:
  * the final heads, output contract and RAG rules are already frozen and committed
  * the encoder geometry matches the development anchor's metadata exactly
  * the detector weights hash matches the frozen candidate
  * only CAPorg*.jpg is read; rep_* report scans and videos are excluded by name

Budget honesty: locked-47 has already been consumed seven times, with model selection
on it in three runs. This read does not restore it to a clean test set and the scoring
artefact must not name it one.

  PYTHONIOENCODING=utf-8 PYTHONPATH=scripts \
    /c/Users/liujunqing/anaconda3/envs/pytorch_gpu/python.exe \
    scripts/extract_locked_a2x_features.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

LABELS = ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv"
IMAGE_ROOT = ROOT / "data"
FILES = ROOT / "artifacts" / "manifest" / "files.csv"
DEV_ANCHOR = (ROOT / "artifacts" / "experiments" / "medical_encoder_transfer_20260921"
              / "features" / "anchor_dinov2b_deployed")
FROZEN = (ROOT / "artifacts" / "experiments" / "rescue_external_20260922"
          / "frozen_candidate_malformation_a2x")
HEADS = (ROOT / "artifacts" / "experiments" / "product_contract_20260924"
         / "final_heads" / "manifest.json")
OUT = (ROOT / "artifacts" / "experiments" / "locked_consumed_20260924"
       / "features_locked")

POOLINGS = ["mean", "topk_mean", "max", "cls", "std"]
TOPK = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_index() -> pd.DataFrame:
    """The 47 locked cases' capillary images. Report scans are excluded by name.

    The inverse of every other index builder in this project: here a DEVELOPMENT id
    appearing is the error, because mixing the two would make the read unreadable.
    """
    lab = pd.read_csv(LABELS, dtype={"exam_case_id": str})
    locked = sorted(lab.loc[lab.development_fold.isna(), "exam_case_id"])
    dev = set(lab.loc[lab.development_fold.notna(), "exam_case_id"])
    if len(locked) != 47 or len(dev) != 186:
        raise RuntimeError("expected 47 locked / 186 dev, got %d / %d"
                           % (len(locked), len(dev)))
    rows = []
    for case in locked:
        d = IMAGE_ROOT / case
        if not d.is_dir():
            raise RuntimeError("locked case directory missing: %s" % case)
        for p in sorted(d.iterdir()):
            n = p.name.lower()
            if not n.startswith("caporg") or not n.endswith(".jpg"):
                continue
            rows.append(dict(exam_case_id=case,
                             image_path=str(p.relative_to(IMAGE_ROOT)).replace("\\", "/")))
    ix = pd.DataFrame(rows)
    if set(ix.exam_case_id) & dev:
        raise RuntimeError("a development case reached the locked index")
    bad = [p for p in ix.image_path if Path(p).name.lower().startswith("rep")]
    if bad:
        raise RuntimeError("report scan reached the encoder index: %s" % bad[:3])
    # Placeholder images are all on the development side (36 files, 25 cases); assert
    # rather than assume, so a future manifest change surfaces here.
    files = pd.read_csv(FILES, dtype=str)
    black = set(files.loc[files.is_black_placeholder == "True", "path"].fillna(""))
    hit = [p for p in ix.image_path if p in black]
    if hit:
        raise RuntimeError("placeholder image in the locked index: %s" % hit[:3])
    if ix.exam_case_id.nunique() != 47:
        raise RuntimeError("expected images for 47 locked cases, got %d"
                           % ix.exam_case_id.nunique())
    return ix.reset_index(drop=True)


def check_preconditions() -> dict:
    """Refuse to read locked-47 unless the downstream decisions are already frozen."""
    if not HEADS.exists():
        raise SystemExit("final heads are not frozen; run freeze_final_heads.py first")
    with open(HEADS, encoding="utf-8") as fh:
        heads = json.load(fh)
    with open(FROZEN / "frozen_candidate.json", encoding="utf-8") as fh:
        frozen = json.load(fh)
    det = FROZEN / "weights" / "external_detector_best.pt"
    got = sha256(det)
    if got != frozen["detector"]["weights_sha256"]:
        raise RuntimeError("detector weights changed since the freeze: %s" % got)
    for rel in heads["also_frozen_before_locked"]:
        if not (ROOT / rel).exists():
            raise RuntimeError("declared-frozen artefact missing: %s" % rel)
    return dict(heads=heads, frozen=frozen, detector_sha256=got)


def encode(ix: pd.DataFrame, dev_meta: dict) -> tuple[dict, dict]:
    """Run the frozen encoder at the development anchor's exact geometry.

    build/preprocess/pool are imported from the development extractor rather than
    reimplemented: a reimplementation that drifted by one interpolation mode would
    make the locked features incomparable while still producing plausible numbers.
    """
    import torch
    from extract_medical_encoders import ARMS, build, pool, preprocess

    cfg = ARMS["anchor_dinov2b_deployed"]
    dep = cfg["deployed_res"]
    if list(dev_meta["patch_grid"]) != [dep[0] // cfg["patch"], dep[1] // cfg["patch"]]:
        raise RuntimeError("geometry mismatch vs the development anchor")
    if dev_meta["normalisation"]["mean"] != list(cfg["norm"][0]):
        raise RuntimeError("normalisation mismatch vs the development anchor")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, wsha = build(cfg, device, img_size=list(dep))
    if wsha != dev_meta["weights_sha256"]:
        raise RuntimeError("encoder weights differ from the development anchor: %s"
                           % wsha)
    dim, grid = model.num_features, (dep[0] // cfg["patch"], dep[1] // cfg["patch"])
    if dim != dev_meta["feature_dim"]:
        raise RuntimeError("feature dim %d != development %d"
                           % (dim, dev_meta["feature_dim"]))

    n = len(ix)
    outs = {k: np.zeros((n, dim), np.float16) for k in POOLINGS}
    with torch.inference_mode():
        for s in range(0, n, 8):
            rows = ix.image_path.iloc[s:s + 8].tolist()
            prepped = [preprocess(IMAGE_ROOT / r, cfg["size"], cfg["norm"], dep)
                       for r in rows]
            x = torch.stack([p[0] for p in prepped]).to(device)
            tok = model.forward_features(x)
            vals = pool(tok, model.num_prefix_tokens, grid,
                        [p[1] for p in prepped], cfg["patch"], TOPK)
            for k, v in vals.items():
                outs[k][s:s + len(rows)] = v.cpu().numpy().astype(np.float16)
            if (s // 8) % 10 == 0:
                print("  encode %d/%d" % (s + len(rows), n), flush=True)
    del model
    torch.cuda.empty_cache()
    return outs, dict(weights_sha256=wsha, feature_dim=int(dim),
                      patch_grid=list(grid), geometry=dev_meta["geometry"])


def detect(ix: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Run the frozen external detector and aggregate to case level.

    per_image_stats and aggregate come from the development extractor, and the class
    map comes from the freeze rather than this file -- the external checkpoint emits
    bushy/crossing/hairpin/tortuous, and filing its class 0 under a vessel column
    would produce a table that looks entirely normal and means something else.
    """
    from ultralytics import YOLO

    from extract_detector_local_features import aggregate, per_image_stats

    cls_map = {int(k): v for k, v in frozen["preprocessing"]["class_map"].items()}
    conf = frozen["preprocessing"]["conf_threshold"]
    imgsz = frozen["preprocessing"]["detector_imgsz"]
    model = YOLO(str(FROZEN / "weights" / "external_detector_best.pt"))

    rows = []
    for s in range(0, len(ix), 16):
        chunk = ix.iloc[s:s + 16]
        paths = [str(IMAGE_ROOT / p) for p in chunk.image_path]
        # workers is not a predict argument, but batch size is kept small for the
        # same reason: this box has 8.5 GB and orphaned workers have OOMed it before.
        res = model.predict(paths, imgsz=imgsz, conf=conf, verbose=False,
                           device=0, stream=False)
        for (_, r), out in zip(chunk.iterrows(), res):
            st = per_image_stats(out, conf, cls_map)
            st.update(exam_case_id=r.exam_case_id, image_path=r.image_path)
            rows.append(st)
        if (s // 16) % 5 == 0:
            print("  detect %d/%d" % (min(s + 16, len(ix)), len(ix)), flush=True)
    per = pd.DataFrame(rows)
    case = aggregate(per)
    return per, case


def main() -> None:
    pre = check_preconditions()
    with open(DEV_ANCHOR / "metadata.json", encoding="utf-8") as fh:
        dev_meta = json.load(fh)

    ix = locked_index()
    print("locked read: %d cases, %d images" % (ix.exam_case_id.nunique(), len(ix)))

    feats, enc_meta = encode(ix, dev_meta)
    per, case = detect(ix, pre["frozen"])

    # Column parity with the development table, or the frozen head cannot be applied.
    dev_case = pd.read_csv(FROZEN / "case_features.csv",
                           dtype={"exam_case_id": str}).set_index("exam_case_id")
    missing = [c for c in dev_case.columns if c not in case.columns]
    extra = [c for c in case.columns if c not in dev_case.columns]
    if missing or extra:
        raise RuntimeError("local column mismatch missing=%s extra=%s"
                           % (missing[:6], extra[:6]))
    case = case[list(dev_case.columns)]
    if len(case) != 47:
        raise RuntimeError("expected 47 locked case rows, got %d" % len(case))

    OUT.mkdir(parents=True, exist_ok=True)
    ix.to_csv(OUT / "index.csv", index=False)
    for k, v in feats.items():
        np.save(OUT / ("features_%s.npy" % k), v)
    per.to_csv(OUT / "per_image.csv", index=False)
    case.to_csv(OUT / "case_features.csv")

    meta = dict(
        run="locked_consumed_20260924_features",
        scope="locked-47 only",
        cases=int(ix.exam_case_id.nunique()), images=int(len(ix)),
        encoder=dict(arm="anchor_dinov2b_deployed", **enc_meta),
        detector=dict(source="external only",
                      weights_sha256=pre["detector_sha256"],
                      conf=pre["frozen"]["preprocessing"]["conf_threshold"],
                      imgsz=pre["frozen"]["preprocessing"]["detector_imgsz"],
                      class_map=pre["frozen"]["preprocessing"]["class_map"]),
        heads_frozen_sha256=pre["heads"]["head_sha256"],
        excluded_by_name=["rep_* report scans", "videos", "non-CAPorg files"],
        placeholder_images_in_locked=0,
        placeholder_note=("all 36 all-black placeholders are development-side "
                          "(25 cases); asserted, not assumed"),
        this_run_read_locked=True,
        budget_note=("locked-47 was already consumed seven times with model "
                     "selection on it in three runs. This read does not restore it "
                     "to a clean test set; the scoring artefact is named a consumed "
                     "internal holdout descriptive evaluation for that reason."),
        computes_no_metric=("features only. Scoring is a separate script so this "
                            "step cannot be rerun differently in response to a "
                            "number."),
    )
    with open(OUT / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
