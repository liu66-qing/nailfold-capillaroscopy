"""Final-release inference with expert_router_v1.

Per binary target: p = mean of four expert heads
  A0   DINOv2-B image features, 5 poolings, image-level head, case mean
  COL  colour / sharpness statistics (case mean)
  SEG  S0 segmenter vessel geometry (case mean)
  DET  Capillary-Dataset detector counts / box geometry (case aggregate)
then q = sigmoid(a * logit(p) + b), the Platt calibrator fitted on 5-fold
out-of-fold p. Rows are then composed per final_registry.FIELDS.
Deterministic: no device re-centring and no state kept between calls.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .final_registry import (BUNDLE_SHA256, CONF_WORDS, DET_SHA256, ENCODER_SHA256,
                             FIELDS, RELEASE_ID, SCHEMA_VERSION, SEG_SHA256, confidence,
                             derived_ratio)

MODEL_DIR = "artifacts/models/expert_router_v1"


class AssetMismatch(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path, want: str) -> Path:
    got = _sha(path)
    if got != want:
        raise AssetMismatch("%s sha256 %s != release %s" % (path, got, want))
    return path


def load_bundle(root: Path, path: Path | None = None) -> dict:
    import joblib
    p = path or Path(root) / MODEL_DIR / "bundle.joblib"
    return joblib.load(_check(p, BUNDLE_SHA256))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _case_head(h: dict, x: pd.Series) -> float:
    v = x.reindex(h["columns"]).astype(float).to_numpy()
    v = np.where(np.isfinite(v), v, h["median"])
    return float(h["pipe"].predict_proba(v[None])[0, list(h["pipe"].classes_).index(1)])


def target_probs(bundle: dict, feats: dict[str, Any]) -> dict[str, dict[str, float]]:
    """feats: A0 -> {pooling: (n_images, 768)}, COL/SEG/DET -> pd.Series."""
    out = {}
    for t, spec in bundle["targets"].items():
        a0 = np.mean([spec["heads"]["A0"][k].predict_proba(feats["A0"][k])[:, 1].mean()
                      for k in bundle["poolings"]])
        parts = dict(A0=float(a0))
        for k in ("COL", "SEG", "DET"):
            parts[k] = _case_head(spec["heads"][k], feats[k])
        p = float(np.mean(list(parts.values())))
        z = spec["cal_coef"] * _logit(p) + spec["cal_intercept"]
        out[t] = dict(p=p, q=float(1 / (1 + np.exp(-z))), parts=parts)
    return out


def fields_from_probs(tp: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    band_of: dict[str, int] = {}
    for f, name, kind, spec, ref in FIELDS:
        if kind == "binary":
            t, words, flag = spec
            q = tp[t]["q"]
            k = int(q > 0.5)
            c = q if k else 1 - q
            probs = [1 - q, q]
            deviates = flag is not None and k == flag
        elif kind in ("band3", "papilla"):
            lo_t, hi_t, words = spec
            # papilla: "lo" = wavy head (class 0), "hi" = flat head (class 2)
            plo, phi = tp[lo_t]["q"], tp[hi_t]["q"]
            pref = max(0.0, 1 - plo - phi)
            probs = np.array([plo, pref, phi])
            probs = probs / probs.sum()
            k = int(np.argmax(probs))
            c = float(probs[k])
            deviates = kind == "band3" and k != 1
            band_of[f] = k
            probs = probs.tolist()
        elif kind == "fixed":
            out[f] = dict(item=name, value=spec, reference=ref, kind="fixed",
                          deviates=False, confidence=None, confidence_word=None)
            continue
        else:
            continue
        lvl = confidence(c, "binary" if kind == "binary" else "band3")
        out[f] = dict(item=name, value=words[k], reference=ref, kind=kind, class_id=k,
                      deviates=bool(deviates), probability=round(float(c), 3),
                      confidence=lvl, confidence_word=CONF_WORDS[lvl],
                      class_probabilities=[round(float(x), 3) for x in probs])
    ea, aa = band_of["efferent_diameter"], band_of["afferent_diameter"]
    out["output_input_ratio"] = dict(
        item="输出/输入枝", value=derived_ratio(ea, aa), reference="输出枝略粗于输入枝",
        kind="derived", deviates=False,
        confidence=min(out["efferent_diameter"]["confidence"],
                       out["afferent_diameter"]["confidence"]),
        confidence_word=None)
    out["output_input_ratio"]["confidence_word"] = \
        CONF_WORDS[out["output_input_ratio"]["confidence"]]
    return {f[0]: out[f[0]] for f in FIELDS}


class FinalReportPredictor:
    def __init__(self, root: Path, device: str | None = None) -> None:
        from ultralytics import YOLO
        from .rag_inference import RagFieldPredictor
        self.root = Path(root)
        self.bundle = load_bundle(self.root)
        md = self.root / MODEL_DIR
        self.seg = YOLO(str(_check(md / "seg_s0.pt", SEG_SHA256)))
        self.det = YOLO(str(_check(md / "det_capillary.pt", DET_SHA256)))
        self.device = device or "cpu"
        self._enc = RagFieldPredictor(self.root, bundle_path=self.root /
                                      "artifacts/models/rag_heads_v1/bundle.joblib",
                                      device=device)
        if _sha(self.root / self._enc.bundle["encoder"]["weights"]) != ENCODER_SHA256:
            raise AssetMismatch("encoder weights sha256 mismatch")

    def features(self, paths: list[Path]) -> dict[str, Any]:
        from .expert_features import col_features, det_features, read_bgr, seg_features
        ims = [read_bgr(p) for p in paths]
        return dict(A0=self._enc.encode([Path(p) for p in paths]), COL=col_features(ims),
                    SEG=seg_features(self.seg, ims, self.device),
                    DET=det_features(self.det, ims, self.device))

    def predict(self, paths: list[Path], *, exam_id: str,
                device_id: str = "unknown") -> dict[str, Any]:
        if not paths:
            raise ValueError("no images")
        tp = target_probs(self.bundle, self.features(paths))
        return build_response(exam_id, device_id, len(paths), fields_from_probs(tp), tp)


def build_response(exam_id: str, device_id: str, n_images: int,
                   fields: dict[str, Any], tp: dict[str, Any]) -> dict[str, Any]:
    from .final_advice import compose_advice
    return dict(schema_version=SCHEMA_VERSION, exam_id=str(exam_id), release_id=RELEASE_ID,
                assets=dict(bundle_sha256=BUNDLE_SHA256, encoder_sha256=ENCODER_SHA256,
                            seg_sha256=SEG_SHA256, det_sha256=DET_SHA256),
                device=dict(id=str(device_id)), images=int(n_images), fields=fields,
                advice=compose_advice(fields),
                audit={t: dict(p=round(v["p"], 4), q=round(v["q"], 4),
                               parts={k: round(x, 4) for k, x in v["parts"].items()})
                       for t, v in tp.items()})
