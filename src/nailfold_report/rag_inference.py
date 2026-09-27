"""Field outputs for the RAG layer, from the frozen rag_heads_v2 bundle (dev 186 + locked 47; loop_length withheld).

Contract (every field is binary and categorical; no number is ever emitted):
  {"field": {"value": <wording> | None, "status": "answered" | "abstained",
             "p": float}}
"abstained" means the model is not confident enough for this person and the
RAG layer must treat it as unknown, never as negative.

Device handling: a new, unknown device shifts the encoder features, which
collapses several heads onto one answer. DeviceCalibrator accumulates image
features per device id, one entry per case; once it holds enough cases and
their centroid is farther from the development reference than 99% of
development batches with the same number of cases, features from that device are re-centred onto the
reference mean/std before the heads run. Below that the features are left
alone, because re-centring also costs a little on the same device.

Fields not in the bundle (flow_state, rbc_aggregation, crossing_ratio, the
three diameters, ...) are reported as "not_modelled" so the RAG layer cannot
mistake a constant for a finding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

NOT_MODELLED = (
    "flow_state", "rbc_aggregation", "crossing_ratio", "afferent_diameter",
    "efferent_diameter", "apex_diameter", "capillary_count", "papilla",
    "hemorrhage", "sweat_duct",
)


def _pool(tokens, npre: int, topk: int) -> dict[str, np.ndarray]:
    import torch
    cls = tokens[:, 0].float()
    pt = tokens[:, npre:].float()
    k = min(topk, pt.shape[1])
    out = {"cls": cls, "mean": pt.mean(1), "max": pt.max(1).values,
           "topk_mean": pt.topk(k, dim=1).values.mean(1), "std": pt.std(1)}
    return {n: v.cpu().numpy().astype(np.float32) for n, v in out.items()
            if isinstance(v, torch.Tensor)}


class DeviceCalibrator:
    """Per-device running feature store. Not thread-safe."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.ref = bundle["reference"]
        self.need = int(bundle["device"]["cases"])
        self.q99 = float(bundle["device"]["null_q99"])
        self.store: dict[str, dict[str, list[np.ndarray]]] = {}

    def add(self, device_id: str, feats: dict[str, np.ndarray]) -> None:
        s = self.store.setdefault(device_id, {k: [] for k in feats})
        for k, v in feats.items():
            s[k].append(v)

    def state(self, device_id: str) -> dict[str, Any]:
        s = self.store.get(device_id)
        n = 0 if s is None else len(s["cls"])
        if n < self.need:
            return dict(cases=n, distance=None, shifted=None, recentre=False)
        X = np.concatenate(s["cls"])
        d = float(np.linalg.norm((X.mean(0) - self.ref["cls"]["mean"])
                                 / self.ref["cls"]["std"]) / np.sqrt(X.shape[1]))
        shifted = d > self.q99
        return dict(cases=n, distance=round(d, 4), shifted=bool(shifted),
                    recentre=bool(shifted))

    def transform(self, device_id: str, feats: dict[str, np.ndarray]):
        st = self.state(device_id)
        if not st["recentre"]:
            return feats, st
        s = self.store[device_id]
        out = {}
        for k, v in feats.items():
            X = np.concatenate(s[k])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            out[k] = ((v - mu) / sd * self.ref[k]["std"] + self.ref[k]["mean"]
                      ).astype(np.float32)
        return out, st


class RagFieldPredictor:
    def __init__(self, root: Path, bundle_path: Path | None = None,
                 device: str | None = None) -> None:
        import joblib
        import timm
        import torch
        from safetensors.torch import load_file

        self.root = Path(root)
        self.bundle = joblib.load(bundle_path or self.root / "artifacts" / "models"
                                  / "rag_heads_v2" / "bundle.joblib")
        enc = self.bundle["encoder"]
        self.torch_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = timm.create_model(enc["model"], pretrained=False, num_classes=0,
                                  img_size=518, dynamic_img_size=True)
        sd = load_file(str(self.root / enc["weights"]))
        sd.pop("mask_token", None)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "mask_token" not in m]
        if missing or unexpected:
            raise RuntimeError("encoder weights do not match %s" % enc["model"])
        self.model = model.to(self.torch_device).eval()
        self.calibrator = DeviceCalibrator(self.bundle)

    def encode(self, paths: list[Path], batch: int = 8) -> dict[str, np.ndarray]:
        import torch
        import torchvision.transforms.functional as TF
        enc = self.bundle["encoder"]
        parts: dict[str, list[np.ndarray]] = {}
        with torch.inference_mode():
            for i in range(0, len(paths), batch):
                xs = []
                for p in paths[i:i + batch]:
                    with Image.open(p) as im:
                        t = TF.resize(im.convert("RGB"), list(enc["resize"]),
                                      interpolation=TF.InterpolationMode.BICUBIC)
                    xs.append(TF.normalize(TF.to_tensor(t), *enc["norm"]))
                tok = self.model.forward_features(
                    torch.stack(xs).to(self.torch_device))
                for k, v in _pool(tok, self.model.num_prefix_tokens,
                                  enc["topk"]).items():
                    parts.setdefault(k, []).append(v)
        return {k: np.concatenate(v) for k, v in parts.items()}

    def raw_predictions(self, feats: dict[str, np.ndarray]) -> dict[str, Any]:
        """Every fitted head, including withheld ones. For the shadow log only;
        never hand this to the RAG layer."""
        out: dict[str, Any] = {}
        for f, spec in self.bundle["fields"].items():
            p = float(np.mean([spec["heads"][k].predict_proba(feats[k])[:, 1].mean()
                               for k in self.bundle["poolings"]]))
            margin = spec["abstain_margin"]
            role = spec.get("role")
            if abs(p - 0.5) < margin:
                out[f] = dict(value=None, status="abstained", p=round(p, 4), role=role)
            else:
                out[f] = dict(value=spec["labels"][int(p >= 0.5)],
                              status="answered", p=round(p, 4), role=role)
        return out

    def predict_features(self, feats: dict[str, np.ndarray]) -> dict[str, Any]:
        out = self.raw_predictions(feats)
        # fields a bundle deliberately withholds (failed a gate) are unmodelled too
        for f in tuple(NOT_MODELLED) + tuple(self.bundle.get("withheld", {})):
            out[f] = dict(value=None, status="not_modelled", p=None, role=None)
        return out

    def predict(self, paths: list[Path], device_id: str = "default") -> dict[str, Any]:
        if not paths:
            raise ValueError("no images")
        feats = self.encode(paths)
        self.calibrator.add(device_id, feats)
        used, st = self.calibrator.transform(device_id, feats)
        fields = self.predict_features(used)
        self.last_raw = self.raw_predictions(used)
        warnings = []
        if st["cases"] < self.calibrator.need:
            warnings.append("该设备累计病例不足 %d 例，尚未做设备校准"
                            % self.calibrator.need)
        elif st["shifted"]:
            warnings.append("检测到与参考设备不同的成像特性，已做设备校准；"
                            "新设备结果未经医生标注验证")
        return dict(schema_version="nailfold-rag/1.0",
                    model_version=self.bundle["version"], fields=fields,
                    device=dict(id=device_id, **st), images=len(paths),
                    warnings=warnings)


class ShadowLog:
    """Append-only prediction log for the prospective temporal evaluation.

    One JSON line per examination, written at inference time, before any
    report exists. The log never reads a report and never rewrites a line;
    pairing with report labels happens later in a separate script. Each line
    carries the bundle sha256 and the git commit so predictions from a model
    other than the frozen one can be rejected at scoring time.
    """

    def __init__(self, path: Path, bundle_path: Path, commit: str) -> None:
        import hashlib
        self.path = Path(path)
        self.bundle_sha256 = hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()
        self.commit = commit

    def record(self, exam_id: str, subject_key: str, exam_date: str,
               result: dict[str, Any] | None, raw: dict[str, Any] | None,
               error: str | None = None,
               device_id: str | None = None) -> dict[str, Any]:
        """subject_key: patient id if one exists, otherwise a stable pseudonym
        of the examined person; used to keep only one exam per person.
        A failed inference is recorded with error set, never skipped;
        device_id keeps a failure in the right (same/cross-device) queue."""
        import datetime
        import json
        row = dict(exam_id=exam_id, subject_key=subject_key, exam_date=exam_date,
                   logged_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   bundle_sha256=self.bundle_sha256, commit=self.commit,
                   model_version=None if result is None else result["model_version"],
                   device=None if result is None else result["device"],
                   images=None if result is None else result["images"],
                   rag_fields=None if result is None else result["fields"],
                   shadow_fields=raw, error=error, device_id_requested=device_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        return row
