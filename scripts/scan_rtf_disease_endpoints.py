#!/usr/bin/env python
"""Does the report text contain DISEASE endpoints at all? A hard-gate count.

The external review's whole 'predict disease risk instead of report fields'
route rests on one unverified premise: that the RTF reports carry diagnoses.
This script checks that premise before any modelling is proposed. It counts
nothing but keyword presence per exam, and it reports coverage as a fraction
of the exams that actually have a label row.

It deliberately does NOT try to assign a disease label to any case. A keyword
hit is not a diagnosis: '否认高血压' (denies hypertension) contains the same
token as a positive history. Negation handling would be required before any
label exists, and that is a separate, larger job. What this answers is only:
is there enough text to make the route worth attempting, or is it empty?

Reads reports only. No image, no model, no locked case involved.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

DISEASE = {
    "diabetes": ["糖尿病", "血糖", "糖尿", "DM"],
    "hypertension": ["高血压", "血压高"],
    "scleroderma_ctd": ["硬化症", "硬皮", "结缔组织", "系统性硬化", "SSc",
                        "干燥综合", "狼疮", "肌炎", "风湿"],
    "raynaud": ["雷诺", "遇冷变白", "肢端发白"],
    "cardiovascular": ["冠心", "心血管", "心梗", "心肌", "脑梗", "动脉硬化",
                       "冠状动脉"],
    "kidney": ["肾病", "肾功", "蛋白尿", "肾衰", "尿毒"],
    "hyperlipidemia": ["高血脂", "血脂", "胆固醇"],
    "followup_flag": ["随访", "复查", "建议进一步", "门诊随诊"],
}
SECTIONS = ["临床诊断", "诊断", "主诉", "既往史", "现病史", "病史", "检查所见",
            "结论", "建议", "印象", "姓名", "性别", "年龄", "门诊号", "住院号"]
NEGATION = ["否认", "无", "未见", "阴性", "不详"]


def rtf_text(path):
    """Decode a GB-coded RTF: strip control words, then decode \'xx byte runs.

    Verified equivalent to the decoder in evidence_group2_dataset_stats.py.
    Mid-session I suspected that one of producing mojibake and said so; that
    suspicion was WRONG and came from a badly-escaped throwaway probe, not
    from the committed code. Both decoders were run over the same 455 files on
    both machines: identical median CJK length (78) and identical file counts
    for every keyword (糖尿 97, 血脂 150, 风湿 132, 雷诺 29, 高血压 10,
    仪器/放大/倍 all 0). So the group-2 acquisition numbers stand as delivered
    (finger coverage 0.0217, device/magnification zero) and need no revision.
    """
    raw = open(path, "rb").read().decode("latin-1", "ignore")
    raw = re.sub(r"\\\*?\\?[a-zA-Z]+-?[0-9]*[ ]?", "", raw)
    raw = re.sub(r"[{}]", "", raw)
    buf, out, i = bytearray(), [], 0
    while i < len(raw):
        if raw[i] == "\\" and i + 3 < len(raw) and raw[i + 1] == "'":
            try:
                buf.append(int(raw[i + 2:i + 4], 16))
            except ValueError:
                pass
            i += 4
        else:
            if buf:
                out.append(buf.decode("gb18030", "ignore"))
                buf = bytearray()
            out.append(raw[i])
            i += 1
    if buf:
        out.append(buf.decode("gb18030", "ignore"))
    return re.sub(r"\s+", " ", "".join(out))


INFERENCE_MARKERS = ["是", "注意", "符合", "表明", "的表现", "考虑", "提示",
                     "等有关", "改变", "建议", "可疑"]
HISTORY_MARKERS = ["既往", "病史", "确诊", "已诊断", "多年", "服药", "口服",
                   "否认", "家族史", "年前"]
# followup_flag is not a disease; it is tracked separately so it cannot pad
# the count of exams that mention a disease at all
NOT_A_DISEASE = {"followup_flag"}
EXAM_PAT = re.compile(r"[\\/](recovered_archive\d)[\\/](\d+)[\\/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    labelled = set(m.exam_case_id)
    dev = set(m.exam_case_id[m.evaluation_role == "development"])

    # only the three case archives; other .rtf under data/ (training material)
    # belong to no exam and would inflate the file counts
    reports = []
    for arch in ["recovered_archive1", "recovered_archive2",
                 "recovered_archive3"]:
        for root, _dirs, files in os.walk(os.path.join(a.data_root, arch)):
            if "__MACOSX" in root:
                continue  # AppleDouble resource forks, not documents
            for fn in files:
                if fn.lower().endswith(".rtf") and not fn.startswith("._"):
                    reports.append(os.path.join(root, fn))
    print("rtf files found:", len(reports))

    per_exam = defaultdict(lambda: defaultdict(int))
    sect = Counter()
    disease_files = Counter()
    neg_ctx = defaultdict(Counter)
    total_chars = []
    unmatched = 0
    # the question that decides the route: is a disease word a recorded
    # patient fact, or the operator's own guess written out of the image?
    exams_any_disease = set()
    exams_inference_ctx = set()
    exams_history_ctx = set()
    history_word_files = Counter()
    sample_sentences = []
    per_exam_seen = set()
    all_texts = []

    for p in reports:
        try:
            t = rtf_text(p)
        except Exception:
            continue
        total_chars.append(len(t))
        all_texts.append(t)
        for s in SECTIONS:
            if s in t:
                sect[s] += 1
        # exam_case_id is '<archive>/<numeric_directory>', e.g.
        # 'recovered_archive1/28' -> data/recovered_archive1/28/rep_rch1.rtf
        mo = EXAM_PAT.search(p)
        exam = f"{mo.group(1)}/{mo.group(2)}" if mo else None
        if exam is None or exam not in labelled:
            unmatched += 1
        else:
            per_exam_seen.add(exam)
        for hw in HISTORY_MARKERS:
            if hw in t:
                history_word_files[hw] += 1

        for dis, kws in DISEASE.items():
            hit = [k for k in kws if k in t]
            if hit:
                disease_files[dis] += 1
                if exam:
                    per_exam[exam][dis] += 1
                for k in hit:
                    for i in [mm.start() for mm in re.finditer(re.escape(k), t)]:
                        window = t[max(0, i - 8):i]
                        for ng in NEGATION:
                            if ng in window:
                                neg_ctx[dis][ng] += 1
                        # +-30 chars around the mention decides its nature
                        ctx = t[max(0, i - 30):i + 30]
                        if exam and dis not in NOT_A_DISEASE:
                            exams_any_disease.add(exam)
                            if any(mk in ctx for mk in INFERENCE_MARKERS):
                                exams_inference_ctx.add(exam)
                            if any(mk in ctx for mk in HISTORY_MARKERS):
                                exams_history_ctx.add(exam)
                        if len(sample_sentences) < 12:
                            sample_sentences.append(ctx.strip())

    rows = []
    for dis in sorted(DISEASE, key=lambda d: d in NOT_A_DISEASE):
        exams = [e for e in per_exam if per_exam[e].get(dis)]
        rows.append({
            "endpoint": dis,
            "is_a_disease_endpoint": dis not in NOT_A_DISEASE,
            "keywords": ",".join(DISEASE[dis]),
            "n_report_files_with_any_keyword": disease_files[dis],
            "n_exams_with_any_keyword": len(exams),
            "n_exams_in_labelled_manifest": len([e for e in exams
                                                 if e in labelled]),
            "n_exams_in_development": len([e for e in exams if e in dev]),
            "coverage_of_233_labelled": round(len([e for e in exams
                                                   if e in labelled]) / 233, 4),
            "negation_windows": dict(neg_ctx[dis]),
            "meets_30_positive_gate": bool(len([e for e in exams
                                                if e in dev]) >= 30),
        })

    out = {
        "question": "do the reports carry disease endpoints that could replace "
                    "the 21 report fields as a training target?",
        "n_rtf_files_scanned": len(reports),
        "n_files_not_mappable_to_a_labelled_exam": unmatched,
        "n_exams_with_at_least_one_report": len(per_exam_seen),
        "n_development_exams_with_a_report": len(per_exam_seen & dev),
        "n_development_exams_total": len(dev),
        "median_decoded_chars_per_report": (
            int(pd.Series(total_chars).median()) if total_chars else 0),
        "section_headers_found_in_n_files": dict(sect.most_common()),
        "endpoints": rows,
        "nature_of_the_disease_mentions": {
            "n_exams_with_any_disease_word": len(exams_any_disease),
            "n_of_those_inside_an_image_inference_phrase":
                len(exams_inference_ctx),
            "n_of_those_near_a_patient_history_marker": len(exams_history_ctx),
            "history_marker_word_file_counts": dict(history_word_files),
            "inference_markers": INFERENCE_MARKERS,
            "history_markers": HISTORY_MARKERS,
            "sample_contexts_30_chars_each_side": sample_sentences,
            "n_distinct_report_texts": len(set(all_texts)),
            "n_distinct_sentences": len({
                s.strip() for t in all_texts
                for s in re.split(r"[。；]", t) if len(s.strip()) > 4}),
            "n_sentence_occurrences": len([
                s for t in all_texts
                for s in re.split(r"[。；]", t) if len(s.strip()) > 4]),
            "why_this_column_decides_the_route": (
                "if every disease mention sits in an inference phrase "
                "('...是雷诺的表现', '...与糖尿病等有关') and none sits near a "
                "history marker, then the word is the operator's reading of "
                "THIS image, not a fact about the patient. Training "
                "image -> disease on it is circular: the target is a "
                "relabelling of the same image-derived judgement."),
        },
        "how_to_read_this": [
            "a keyword hit is NOT a diagnosis. '否认高血压' contains the same "
            "token as a positive history; the negation_windows column counts "
            "how often a negation word sits within 8 characters before the hit.",
            "the gate the review proposed is >=30 positives AND >=30 negatives "
            "per endpoint, present in all three archives. meets_30_positive_gate "
            "here is an UPPER BOUND on positives: it counts any mention, "
            "including denials and family history.",
            "if an endpoint's upper bound is already below 30, the route is "
            "closed for that endpoint regardless of negation handling.",
        ],
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("\nsections present in N files:", dict(sect.most_common(12)))
    print("median chars/report:", out["median_decoded_chars_per_report"])
    print(f"\n{'endpoint':20s} {'files':>6s} {'exams':>6s} {'dev':>5s} "
          f"{'>=30?':>6s}  negation")
    for r in rows:
        print(f"{r['endpoint']:20s} {r['n_report_files_with_any_keyword']:6d} "
              f"{r['n_exams_with_any_keyword']:6d} "
              f"{r['n_exams_in_development']:5d} "
              f"{str(r['meets_30_positive_gate']):>6s}  {r['negation_windows']}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
