"""OCR the per-case report images (rep*.jpg) and compare against the RTF text.

Two questions:
  Q1 Do the report images carry any patient-level diagnosis that the RTF lacks?
  Q2 For the 43 case folders that have NO rtf, does the report image still
     contain the table + advice (i.e. recoverable labels)?

Read-only w.r.t. data. Writes OCR text to the evidence dir.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_case_folder_documents import (  # noqa: E402
    DISEASE_WORDS,
    HISTORY_MARKERS,
    cjk_len,
    decode_document,
)

# Anchors that prove the image is the standard measurement report.
TABLE_ANCHORS = ["测量项目", "正常值", "积分", "清晰度", "管袢", "血色", "乳头"]
GRADE_ANCHORS = ["综合判断", "总积分"]
ADVICE_ANCHOR = "建议"

GRADE_LEVELS = ["大致正常", "轻度异常", "中度异常", "重度异常", "正常"]


def norm(s: str) -> str:
    return re.sub(r"[\s\.。，、；：\[\]（）()%]+", "", s)


def pick_rep_images(case_dir: str) -> list[str]:
    """One representative image per distinct content hash (they are duplicated)."""
    import hashlib

    paths = sorted(glob.glob(os.path.join(case_dir, "rep*.jpg")))
    seen: dict[str, str] = {}
    for p in paths:
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        seen.setdefault(h, p)
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument(
        "--archives",
        nargs="+",
        default=["recovered_archive1", "recovered_archive2", "recovered_archive3"],
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--resume", action="store_true", help="skip cases already in ocr_rows.jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()

    # Resume support: the full pass is slow and has been killed mid-run before,
    # so every case is appended to a jsonl as soon as it is done.
    jsonl_path = os.path.join(args.out_dir, "ocr_rows.jsonl")
    rows = []
    done_keys: set[tuple[str, str]] = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                done_keys.add((r["archive"], r["case"]))
        print(f"resuming: {len(done_keys)} cases already done")
    sink = open(jsonl_path, "a", encoding="utf-8")

    n_done = 0
    for arch in args.archives:
        adir = os.path.join(args.root, arch)
        if not os.path.isdir(adir):
            continue
        for case in sorted(os.listdir(adir), key=lambda x: (not x.isdigit(), x)):
            if not case.isdigit():
                continue
            cdir = os.path.join(adir, case)
            if not os.path.isdir(cdir):
                continue
            if args.limit and n_done >= args.limit:
                break
            if (arch, case) in done_keys:
                continue

            rtfs = sorted(glob.glob(os.path.join(cdir, "*.rtf")))
            rtf_text = ""
            for r in rtfs:
                try:
                    rtf_text += decode_document(r) + "\n"
                except Exception:
                    pass

            imgs = pick_rep_images(cdir)
            img_text = ""
            for p in imgs:
                try:
                    res, _ = ocr(p)
                except Exception as exc:
                    img_text += f"<<OCR ERROR {type(exc).__name__}>>"
                    continue
                if res:
                    img_text += " ".join(r[1] for r in res) + "\n"

            ni, nr = norm(img_text), norm(rtf_text)
            # advice段: text after the last 建议 anchor in the OCR
            advice_ocr = ""
            if ADVICE_ANCHOR in img_text:
                advice_ocr = img_text.split(ADVICE_ANCHOR)[-1]

            grade = [g for g in GRADE_LEVELS if g in ni]
            sim = (
                difflib.SequenceMatcher(None, norm(advice_ocr), nr).ratio()
                if advice_ocr and nr
                else None
            )

            row = {
                    "archive": arch,
                    "case": case,
                    "n_rtf": len(rtfs),
                    "rtf_cjk": cjk_len(rtf_text),
                    "n_rep_images_distinct": len(imgs),
                    "ocr_cjk": cjk_len(img_text),
                    "table_anchors_found": [a for a in TABLE_ANCHORS if a in ni],
                    "grade_anchor_found": [a for a in GRADE_ANCHORS if a in ni],
                    "grade_value": grade,
                    "has_advice_section": bool(advice_ocr.strip()),
                    "advice_vs_rtf_similarity": None if sim is None else round(sim, 3),
                    "ocr_history_markers": [w for w in HISTORY_MARKERS if w in img_text],
                    "ocr_disease_words": [w for w in DISEASE_WORDS if w in img_text],
                    "ocr_text": img_text.strip(),
                    "rtf_text": rtf_text.strip(),
                }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  ... {n_done} cases", flush=True)

    with_rtf = [r for r in rows if r["n_rtf"] > 0]
    no_rtf = [r for r in rows if r["n_rtf"] == 0]

    def frac(sub, pred):
        return f"{sum(1 for r in sub if pred(r))}/{len(sub)}" if sub else "0/0"

    sims = [r["advice_vs_rtf_similarity"] for r in with_rtf if r["advice_vs_rtf_similarity"] is not None]
    sims.sort()
    summary = {
        "n_cases_ocred": len(rows),
        "n_with_rtf": len(with_rtf),
        "n_without_rtf": len(no_rtf),
        "cases_with_no_rep_image": [
            f"{r['archive']}/{r['case']}" for r in rows if r["n_rep_images_distinct"] == 0
        ],
        "Q1_diagnosis_beyond_rtf": {
            "images_with_history_marker": frac(rows, lambda r: r["ocr_history_markers"]),
            "images_with_disease_word": frac(rows, lambda r: r["ocr_disease_words"]),
            "advice_vs_rtf_similarity_median": (
                sims[len(sims) // 2] if sims else None
            ),
            "advice_vs_rtf_similarity_p10": (sims[len(sims) // 10] if len(sims) >= 10 else None),
        },
        "Q2_recovery_from_images_when_no_rtf": {
            "has_full_table": frac(no_rtf, lambda r: len(r["table_anchors_found"]) >= 5),
            "has_grade": frac(no_rtf, lambda r: r["grade_value"]),
            "has_advice_text": frac(no_rtf, lambda r: r["has_advice_section"]),
            "cases": [f"{r['archive']}/{r['case']}" for r in no_rtf],
        },
        "table_present_overall": frac(rows, lambda r: len(r["table_anchors_found"]) >= 5),
        "grade_present_overall": frac(rows, lambda r: r["grade_value"]),
    }

    with open(os.path.join(args.out_dir, "ocr_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
