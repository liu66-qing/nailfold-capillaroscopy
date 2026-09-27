"""Shadow-log entry point for the prospective v2 evaluation, until the new
product inference entry exists.

  python scripts/shadow_infer.py EXAM_DIR [EXAM_DIR ...] [--archive NAME]
                                 [--exam-date YYYY-MM-DD] [--device default]

EXAM_DIR is one examination folder (the numbered folder, e.g. .../archive4/12).
Only CAPorg*.jpg images are read; report files (rep*, prn*, *.rtf) in the same
folder are never opened. Names are removed upstream, so subject_key is
"<archive>/<folder number>" (user decision 2026-09-28).

exam_date: --exam-date if given, else the earliest modification date of the
CAPorg images. Examinations dated before 2026-09-29 are logged but never scored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from nailfold_report.rag_inference import RagFieldPredictor, ShadowLog  # noqa: E402

BUNDLE = ROOT / "artifacts" / "models" / "rag_heads_v2" / "bundle.joblib"
LOG = ROOT / "artifacts" / "prospective_eval_v2" / "shadow_log.jsonl"


def exam_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.name.lower().startswith("caporg")
                  and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exam_dirs", nargs="+", type=Path)
    ap.add_argument("--archive", default=None,
                    help="archive name for subject_key; default: parent folder name")
    ap.add_argument("--exam-date", default=None)
    ap.add_argument("--device", default="default")
    ap.add_argument("--log", type=Path, default=LOG)
    a = ap.parse_args(argv)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    log = ShadowLog(a.log, BUNDLE, commit)
    pred = RagFieldPredictor(ROOT, BUNDLE)
    for d in a.exam_dirs:
        d = d.resolve()
        archive = a.archive or d.parent.name
        key = "%s/%s" % (archive, d.name)
        imgs = exam_images(d)
        date = a.exam_date or (min(dt.date.fromtimestamp(p.stat().st_mtime) for p in imgs).isoformat()
                               if imgs else None)
        try:
            if not imgs:
                raise ValueError("no CAPorg image")
            res = pred.predict(imgs, device_id=a.device)
            log.record(key, key, date, res, pred.last_raw, device_id=a.device)
            print(key, date, "ok", len(imgs))
        except Exception as e:  # a failure is logged, never skipped
            log.record(key, key, date, None, None,
                       error="%s: %s" % (type(e).__name__, e), device_id=a.device)
            print(key, date, "FAILED", e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
