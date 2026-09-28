"""End-to-end check and examples for the final release.

1. runs FinalReportPredictor on the raw images of a few cases and checks that
   the live q of every target matches q computed from the saved training
   feature tables (|dq| < 0.01);
2. writes example report JSON + printable HTML to release/final_v1/examples/.
Example cases are drawn from development cases only (no locked-47 image is
rendered into a shipped file).

    python scripts/make_final_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_v1_v2_training as C  # noqa: E402
import eval_field_experts as E  # noqa: E402
from nailfold_report.final_inference import FinalReportPredictor, target_probs  # noqa: E402
from nailfold_report.final_render import render_html, thumbnail_bytes  # noqa: E402

OUT = ROOT / "release/final_v1/examples"
PATIENTS = ({"name": "示例一", "sex": "女", "age": "52"},
            {"name": "示例二", "sex": "男", "age": "38"},
            {"name": "示例三", "sex": "女", "age": "61"},
            {"name": "示例四", "sex": "男", "age": "45"},
            {"name": "示例五", "sex": "女", "age": "29"},
            {"name": "示例六", "sex": "男", "age": "67"},
            {"name": "示例七", "sex": "女", "age": "48"},
            {"name": "示例八", "sex": "男", "age": "56"})


def main() -> None:
    man, ixa, Fa, source, _ = C.load()
    routes = E.load_case_routes(ixa)
    pred = FinalReportPredictor(ROOT, device="cpu")   # GPU is left to training jobs
    dev = sorted(c for c in set(ixa.exam_case_id) if source[c] == "dev")
    rng = np.random.default_rng(20260928)
    pick = list(rng.choice(dev, len(PATIENTS), replace=False))
    OUT.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for n, c in enumerate(pick, 1):
        idx = np.where(ixa.exam_case_id.to_numpy() == c)[0]
        paths = [ROOT / "data" / ixa.image_path[i] for i in idx]
        saved = dict(A0={k: Fa[k][idx] for k in Fa}, COL=routes["COL"].loc[c],
                     SEG=routes["SEG"].loc[c], DET=routes["DET"].loc[c])
        ref = target_probs(pred.bundle, saved)
        r = pred.predict(paths, exam_id="EX-%03d" % n, device_id="示例设备")
        worst = max(worst, max(abs(r["audit"][t]["q"] - ref[t]["q"]) for t in ref))
        r.pop("audit")
        (OUT / ("example_%d.json" % n)).write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        render_html(r, OUT / ("example_%d.html" % n),
                    patient=dict(PATIENTS[n - 1], exam_date="2026-09-28", capture_time="上午",
                                 finger=("左手无名指", "右手中指")[n % 2], room_temp=str(22 + n % 3)),
                    images=thumbnail_bytes(paths))
        dv = [d["item"] + d["value"] for d in r["fields"].values() if d["deviates"]]
        print(n, len(paths), "images; deviating:", dv)
    print("max |q live - q from saved tables|: %.4f" % worst)
    assert worst < 0.01, "live features do not reproduce the evaluated pipeline"


if __name__ == "__main__":
    main()
