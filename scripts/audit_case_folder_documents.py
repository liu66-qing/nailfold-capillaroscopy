"""Audit the per-case documents in recovered_archive1/2/3.

User's correction to check: "each folder is one case, the diagnosis is in the Word
document in the folder, and the report images carry the same diagnostic advice".

This script answers, per case folder:
  1. which document files exist (.doc/.docx/.rtf/other non-image)
  2. the decoded text of each
  3. whether the text contains patient-level diagnosis markers (history, chief
     complaint, confirmed dx, meds, labs) as opposed to image-reading sentences
  4. whether prn_* and rep_* differ in content
  5. how many distinct texts exist vs how many cases (template reuse)

Read-only. Touches no model, no locked cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter

BS = chr(92)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXT = {".avi", ".mp4", ".mov", ".wmv", ".mkv"}
DOC_EXT = {".doc", ".docx", ".rtf", ".odt", ".wps", ".pdf", ".txt"}

# Patient-level facts: things only a person's record can supply.
HISTORY_MARKERS = [
    "既往", "病史", "主诉", "现病史", "家族史", "过敏史", "个人史",
    "确诊", "已诊断", "诊断为", "入院", "门诊", "住院", "复诊", "随访",
    "服药", "用药", "治疗", "手术", "化验", "检验", "抗体", "血糖",
    "血压值", "身高", "体重", "性别", "年龄", "岁",
]

# Disease names that may appear either as a patient fact or as image speculation.
DISEASE_WORDS = [
    "糖尿", "高血压", "雷诺", "硬化", "结缔", "风湿", "血脂", "冠心",
    "肾", "心血管", "血栓", "贫血", "肿瘤", "甲减", "甲亢", "狼疮",
]

# Hedge/inference verbs that mark a sentence as read-from-image speculation.
SPECULATION_MARKERS = [
    "提示", "表明", "可能", "考虑", "建议", "有关", "相关", "多见于",
    "常见于", "倾向", "疑", "或", "符合",
]

# Image-reading vocabulary (what the 21 report fields already encode).
IMAGE_READING_WORDS = [
    "管袢", "管径", "清晰度", "排列", "分布", "血流", "渗出", "乳头",
    "静脉丛", "红细胞", "聚集", "迂曲", "交叉", "畸形", "顶端", "数量",
]


def decode_rtf(path: str) -> str:
    raw = open(path, "rb").read()
    txt = raw.decode("latin-1")
    out: list[bytes] = []
    i, n = 0, len(txt)
    while i < n:
        c = txt[i]
        if c == BS:
            if i + 1 < n and txt[i + 1] == "'":
                hexs = txt[i + 2 : i + 4]
                try:
                    out.append(bytes([int(hexs, 16)]))
                    i += 4
                    continue
                except ValueError:
                    pass
            m = re.match(r"[a-zA-Z]+(-?[0-9]+)?[ ]?", txt[i + 1 :])
            if m:
                ctrl = m.group(0).strip()
                if ctrl.startswith("par") or ctrl.startswith("line"):
                    out.append(b"\n")
                i += 1 + len(m.group(0))
                continue
            i += 2
            continue
        if c in "{}":
            i += 1
            continue
        out.append(c.encode("latin-1"))
        i += 1
    b = b"".join(out)
    for enc in ("gbk", "gb18030", "utf-8"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", errors="replace")


def decode_docx(path: str) -> str:
    """Extract text from a .docx (zip + word/document.xml) without python-docx."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as exc:  # not a zip => legacy .doc
        return f"<<UNREADABLE docx: {type(exc).__name__}>>"
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def decode_legacy_doc(path: str) -> str:
    """Best-effort text salvage from a legacy binary .doc."""
    raw = open(path, "rb").read()
    # GB18030 runs of CJK
    try:
        txt = raw.decode("gb18030", errors="ignore")
    except Exception:
        txt = raw.decode("latin-1", errors="ignore")
    cjk = re.findall(r"[一-鿿，。；：、（）%0-9A-Za-z]{4,}", txt)
    return "\n".join(cjk)


def decode_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".rtf":
        return decode_rtf(path)
    if ext == ".docx":
        return decode_docx(path)
    if ext in {".doc", ".wps"}:
        return decode_legacy_doc(path)
    if ext == ".txt":
        raw = open(path, "rb").read()
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("latin-1", errors="replace")
    return f"<<NO DECODER for {ext}>>"


def cjk_len(s: str) -> int:
    return len(re.findall(r"[一-鿿]", s))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if cjk_len(p) >= 2]


def classify_disease_sentences(text: str) -> list[dict]:
    """For every sentence containing a disease word, say whether it reads as a
    patient fact (near a history marker) or as image speculation."""
    rows = []
    for sent in split_sentences(text):
        hits = [w for w in DISEASE_WORDS if w in sent]
        if not hits:
            continue
        rows.append(
            {
                "sentence": sent,
                "disease_words": hits,
                "history_markers": [w for w in HISTORY_MARKERS if w in sent],
                "speculation_markers": [w for w in SPECULATION_MARKERS if w in sent],
                "image_reading_words": [w for w in IMAGE_READING_WORDS if w in sent],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument(
        "--archives",
        nargs="+",
        default=["recovered_archive1", "recovered_archive2", "recovered_archive3"],
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-samples", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    per_case: list[dict] = []
    ext_counter: Counter = Counter()
    doc_ext_counter: Counter = Counter()
    text_hashes: Counter = Counter()
    all_disease_rows: list[dict] = []
    samples: list[dict] = []

    for arch in args.archives:
        adir = os.path.join(args.root, arch)
        if not os.path.isdir(adir):
            per_case.append({"archive": arch, "error": "missing"})
            continue
        for case in sorted(os.listdir(adir)):
            cdir = os.path.join(adir, case)
            if not os.path.isdir(cdir):
                continue
            files = sorted(os.listdir(cdir))
            docs, imgs, vids, others = [], [], [], []
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                ext_counter[ext] += 1
                if ext in DOC_EXT:
                    docs.append(f)
                    doc_ext_counter[ext] += 1
                elif ext in IMAGE_EXT:
                    imgs.append(f)
                elif ext in VIDEO_EXT:
                    vids.append(f)
                else:
                    others.append(f)

            doc_records = []
            for d in docs:
                p = os.path.join(cdir, d)
                try:
                    text = decode_document(p)
                except Exception as exc:
                    text = f"<<DECODE ERROR {type(exc).__name__}: {exc}>>"
                text_norm = re.sub(r"\s+", "", text)
                h = hashlib.sha256(text_norm.encode("utf-8", "replace")).hexdigest()[:16]
                text_hashes[h] += 1
                drows = classify_disease_sentences(text)
                for r in drows:
                    r = dict(r)
                    r["archive"] = arch
                    r["case"] = case
                    r["doc"] = d
                    all_disease_rows.append(r)
                rec = {
                    "file": d,
                    "bytes": os.path.getsize(p),
                    "cjk_len": cjk_len(text),
                    "text_sha16": h,
                    "has_history_marker": [w for w in HISTORY_MARKERS if w in text],
                    "has_disease_word": [w for w in DISEASE_WORDS if w in text],
                    "n_disease_sentences": len(drows),
                }
                doc_records.append(rec)
                if len(samples) < args.dump_samples:
                    samples.append(
                        {
                            "archive": arch,
                            "case": case,
                            "file": d,
                            "text": text.strip()[:1200],
                        }
                    )

            # do prn_* and rep_* carry different content?
            prn = [r for r in doc_records if r["file"].lower().startswith("prn")]
            rep = [r for r in doc_records if r["file"].lower().startswith("rep")]
            prn_rep_same = None
            if prn and rep:
                prn_rep_same = prn[0]["text_sha16"] == rep[0]["text_sha16"]

            per_case.append(
                {
                    "archive": arch,
                    "case": case,
                    "n_docs": len(docs),
                    "n_images": len(imgs),
                    "n_videos": len(vids),
                    "docs": doc_records,
                    "other_files": others,
                    "prn_rep_identical": prn_rep_same,
                    "any_history_marker": any(r["has_history_marker"] for r in doc_records),
                }
            )

    real_cases = [c for c in per_case if "case" in c]
    n_cases = len(real_cases)
    cases_with_doc = sum(1 for c in real_cases if c["n_docs"] > 0)
    cases_no_doc = [f"{c['archive']}/{c['case']}" for c in real_cases if c["n_docs"] == 0]
    cases_with_history = [
        f"{c['archive']}/{c['case']}" for c in real_cases if c["any_history_marker"]
    ]
    prn_rep_flags = [c["prn_rep_identical"] for c in real_cases if c["prn_rep_identical"] is not None]

    n_disease_sent = len(all_disease_rows)
    n_as_patient_fact = sum(1 for r in all_disease_rows if r["history_markers"])
    n_as_speculation = sum(
        1 for r in all_disease_rows if r["speculation_markers"] and not r["history_markers"]
    )
    n_neither = n_disease_sent - n_as_patient_fact - n_as_speculation

    distinct_texts = len(text_hashes)
    total_docs = sum(text_hashes.values())

    summary = {
        "n_case_folders": n_cases,
        "n_cases_with_any_document": cases_with_doc,
        "n_cases_without_document": len(cases_no_doc),
        "cases_without_document": cases_no_doc,
        "document_extensions_found": dict(doc_ext_counter),
        "all_extensions_found": dict(ext_counter),
        "n_doc_files": total_docs,
        "n_distinct_document_texts": distinct_texts,
        "template_reuse_ratio": (
            round(1 - distinct_texts / total_docs, 4) if total_docs else None
        ),
        "top_repeated_texts": text_hashes.most_common(5),
        "prn_vs_rep_identical_counts": dict(Counter(prn_rep_flags)),
        "n_cases_with_patient_history_marker": len(cases_with_history),
        "cases_with_patient_history_marker": cases_with_history[:50],
        "disease_sentences": {
            "total": n_disease_sent,
            "attached_to_patient_history": n_as_patient_fact,
            "image_speculation": n_as_speculation,
            "neither": n_neither,
        },
    }

    payload = {
        "summary": summary,
        "samples": samples,
        "disease_sentence_examples": all_disease_rows[:40],
        "per_case": per_case,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
