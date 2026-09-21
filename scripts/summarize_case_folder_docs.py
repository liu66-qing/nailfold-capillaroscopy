"""Summarize the per-case document + report-image OCR audit.

Answers the two questions the user's correction raises:
  Q1 Do the per-case documents / report images contain patient-level diagnosis
     (a disease state that is NOT read off this image)?
  Q2 For case folders with no .rtf, does the report image still carry the
     measurement table + grade + advice, i.e. recoverable labels?

Reads artifacts/evidence/case_folder_docs_20260922/ocr_rows.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_case_folder_documents import DISEASE_WORDS, HISTORY_MARKERS  # noqa: E402

TABLE_ANCHORS = ["测量项目", "正常值", "积分", "清晰度", "管袢", "血色", "乳头"]
GRADE_LEVELS = ["大致正常", "轻度异常", "中度异常", "重度异常", "正常"]

# Markers that would prove a patient fact rather than an image reading.
STRONG_HISTORY = ["既往", "病史", "主诉", "现病史", "家族史", "确诊", "已诊断",
                  "诊断为", "入院", "住院", "复诊", "随访", "化验", "检验"]
# Hedges that mark the sentence as inference from this image.
SPECULATION = ["提示", "表明", "可能", "考虑", "建议", "有关", "相关",
               "多见于", "常见于", "注意", "符合", "疑"]


def norm(s: str) -> str:
    return re.sub(r"[\s\.。，、；：\[\]（）()%]+", "", s)


def cjk_sentences(text: str) -> list[str]:
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if len(re.findall(r"[一-鿿]", p)) >= 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/evidence/case_folder_docs_20260922/ocr_rows.jsonl")
    ap.add_argument("--out", default="artifacts/evidence/case_folder_docs_20260922/summary.json")
    ap.add_argument("--canonical", default="artifacts/audits/canonical_labels.csv")
    args = ap.parse_args()

    # The OCR pass was killed and resumed several times, so a case interrupted
    # mid-write can appear twice. Keep the last record per (archive, case).
    by_key: dict[tuple[str, str], dict] = {}
    n_raw = 0
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_raw += 1
            by_key[(r["archive"], r["case"])] = r
    rows = sorted(by_key.values(), key=lambda r: (r["archive"], int(r["case"])))
    n_dupes = n_raw - len(rows)

    canon: set[str] = set()
    if os.path.exists(args.canonical):
        import csv

        with open(args.canonical, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                canon.add(r["exam_case_id"])

    with_rtf = [r for r in rows if r["n_rtf"] > 0]
    no_rtf = [r for r in rows if r["n_rtf"] == 0]
    has_img = [r for r in rows if r["n_rep_images_distinct"] > 0]
    no_rtf_with_img = [r for r in no_rtf if r["n_rep_images_distinct"] > 0]

    # ---- Q1: is there any patient-level diagnosis anywhere? ----
    strong_hits = []
    disease_sentences = []
    for r in rows:
        blob = (r["ocr_text"] or "") + "\n" + (r["rtf_text"] or "")
        s_hits = [w for w in STRONG_HISTORY if w in blob]
        if s_hits:
            strong_hits.append({"case": f"{r['archive']}/{r['case']}", "markers": s_hits})
        for sent in cjk_sentences(blob):
            dz = [w for w in DISEASE_WORDS if w in sent]
            if not dz:
                continue
            disease_sentences.append(
                {
                    "case": f"{r['archive']}/{r['case']}",
                    "sentence": sent,
                    "disease_words": dz,
                    "strong_history": [w for w in STRONG_HISTORY if w in sent],
                    "speculation": [w for w in SPECULATION if w in sent],
                }
            )

    n_dz = len(disease_sentences)
    n_dz_history = sum(1 for d in disease_sentences if d["strong_history"])
    n_dz_spec = sum(1 for d in disease_sentences if d["speculation"] and not d["strong_history"])

    sims = sorted(
        r["advice_vs_rtf_similarity"]
        for r in with_rtf
        if r["advice_vs_rtf_similarity"] is not None
    )

    def pct(v, n):
        return f"{v}/{n}" + (f" ({v / n:.1%})" if n else "")

    # ---- Q2: recoverability from images where rtf is absent ----
    def has_table(r):
        return len(r["table_anchors_found"]) >= 5

    def has_grade(r):
        return bool(r["grade_value"])

    recoverable = [
        f"{r['archive']}/{r['case']}"
        for r in no_rtf_with_img
        if has_table(r) and has_grade(r)
    ]
    recoverable_new = [c for c in recoverable if c not in canon]

    # "正常" is a substring of "大致正常", so take the most specific match only.
    grade_dist = Counter()
    for r in rows:
        gs = [g for g in r["grade_value"] if g in GRADE_LEVELS]
        if gs:
            grade_dist[max(gs, key=len)] += 1
        elif r["grade_value"]:
            grade_dist["正常(bare)"] += 1

    summary = {
        "n_cases": len(rows),
        "n_duplicate_rows_dropped": n_dupes,
        "n_with_rtf": len(with_rtf),
        "n_without_rtf": len(no_rtf),
        "n_with_report_image": len(has_img),
        "n_without_rtf_but_with_report_image": len(no_rtf_with_img),
        "Q1_patient_level_diagnosis": {
            "cases_with_strong_history_marker": pct(len(strong_hits), len(rows)),
            "which": strong_hits[:20],
            "disease_sentences_total": n_dz,
            "disease_sentences_with_strong_history": n_dz_history,
            "disease_sentences_as_image_speculation": n_dz_spec,
            "disease_sentences_neither": n_dz - n_dz_history - n_dz_spec,
        },
        "Q1b_image_vs_rtf_advice": {
            "n_compared": len(sims),
            "similarity_median": sims[len(sims) // 2] if sims else None,
            "similarity_p10": sims[len(sims) // 10] if len(sims) >= 10 else None,
            "n_images_with_extra_history_marker_not_in_rtf": sum(
                1
                for r in rows
                if set(r["ocr_history_markers"]) - set(
                    w for w in HISTORY_MARKERS if w in (r["rtf_text"] or "")
                )
            ),
        },
        "Q2_recoverable_from_image_without_rtf": {
            "table_present": pct(sum(1 for r in no_rtf_with_img if has_table(r)), len(no_rtf_with_img)),
            "grade_present": pct(sum(1 for r in no_rtf_with_img if has_grade(r)), len(no_rtf_with_img)),
            "advice_present": pct(sum(1 for r in no_rtf_with_img if r["has_advice_section"]), len(no_rtf_with_img)),
            "fully_recoverable_cases": recoverable,
            "n_fully_recoverable": len(recoverable),
            "n_not_already_in_canonical_labels": len(recoverable_new),
            "which_new": recoverable_new,
        },
        "table_present_overall": pct(sum(1 for r in rows if has_table(r)), len(rows)),
        "grade_distribution": dict(grade_dist),
        "n_canonical_label_rows": len(canon),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "disease_sentence_examples": disease_sentences[:60]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
