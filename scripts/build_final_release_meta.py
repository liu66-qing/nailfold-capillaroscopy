"""Write release/final_v1/ metadata for expert_router_v1:
validation.json (internal CV numbers per printed row, from the saved OOF),
schemas/report.schema.json and release_manifest.json (sha256 of every asset).

    python scripts/build_final_release_meta.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from nailfold_report.final_registry import (BUNDLE_SHA256, DET_SHA256,  # noqa: E402
                                            ENCODER_SHA256, FIELD_IDS, FIELDS,
                                            FORBIDDEN, RELEASE_ID, SCHEMA_VERSION,
                                            SEG_SHA256)

OUT = ROOT / "release" / "final_v1"
EXP = ROOT / "artifacts/experiments/expert_routes_20260928"
MD = "artifacts/models/expert_router_v1/"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def validation() -> dict:
    cal = json.loads((EXP / "calibrated_router.json").read_text(encoding="utf-8"))
    out = {}
    for f, name, kind, spec, _ref in FIELDS:
        if kind == "binary":
            s = cal["binary"][spec[0]]["acc_rule"]
        elif kind in ("band3", "papilla"):
            s = cal["three_band"][spec[0].split("_")[0]]
        else:
            out[f] = dict(item=name, kind=kind, evidence="固定答案，不使用图像")
            continue
        lo, mid, hi = s["acc_minus_mode"]
        out[f] = dict(item=name, kind=kind, n=s["n"], accuracy=s["acc"],
                      always_common_answer=s["mode_acc"], gain_ci95=[lo, hi],
                      beats_common_answer=bool(lo > 0),
                      protocol="233 例病人级 5 折交叉验证（反复探索后的内部估计，无外部验证）")
    out["output_input_ratio"] = dict(item="输出/输入枝", kind="derived",
                                     evidence="由两枝档位组合")
    return out


def schema() -> dict:
    field = {"type": "object", "required": ["item", "value", "reference", "kind", "deviates"],
             "properties": {"value": {"type": "string", "minLength": 1},
                            "kind": {"enum": ["binary", "band3", "papilla", "fixed",
                                              "derived"]},
                            "deviates": {"type": "boolean"}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "nailfold report", "type": "object",
            "required": ["schema_version", "exam_id", "release_id", "assets", "device",
                         "images", "fields", "advice"],
            "properties": {
                "schema_version": {"const": SCHEMA_VERSION},
                "release_id": {"const": RELEASE_ID},
                "exam_id": {"type": "string", "minLength": 1},
                "images": {"type": "integer", "minimum": 1},
                "fields": {"type": "object", "required": list(FIELD_IDS),
                           "properties": {f: field for f in FIELD_IDS},
                           "additionalProperties": False},
                "advice": {"type": "object",
                           "required": ["headline", "sections", "how_to_read", "see_doctor",
                                        "disclaimer"]}},
            "not": {"anyOf": [{"required": [k]} for k in
                              ("score", "total_score", "grade", "severity", "diagnosis")]}}


def main() -> None:
    dump(OUT / "validation.json", validation())
    dump(OUT / "schemas" / "report.schema.json", schema())
    assets = {MD + "bundle.joblib": BUNDLE_SHA256, MD + "seg_s0.pt": SEG_SHA256,
              MD + "det_capillary.pt": DET_SHA256,
              "weights/dinov2/b/model.safetensors": ENCODER_SHA256}
    for rel, want in assets.items():
        assert sha(ROOT / rel) == want, rel
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*.json"))
             if p.name != "release_manifest.json" and "_superseded" not in p.as_posix()}
    dump(OUT / "release_manifest.json", dict(
        release_id=RELEASE_ID, schema_version=SCHEMA_VERSION, model_assets=assets,
        also_loaded=["artifacts/models/rag_heads_v1/bundle.joblib (encoder config only)"],
        release_files=files, forbidden_phrases=list(FORBIDDEN),
        code=["src/nailfold_report/" + m for m in
              ("final_registry.py", "final_inference.py", "expert_features.py",
               "final_advice.py", "final_render.py", "final_api.py")],
        download=("https://github.com/liu66-qing/nailfold-capillaroscopy/releases/tag/"
                  + RELEASE_ID)))
    print("wrote", len(files) + 1, "files")


if __name__ == "__main__":
    main()
