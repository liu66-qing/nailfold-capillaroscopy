#!/usr/bin/env python
"""How much does the advice text in the reports actually vary?

The proposed product shape is: model outputs field values -> RAG turns them
into health advice. That only has value if the advice DEPENDS on the field
values. If the operator's own advice is a small set of boilerplate strings
that are near-constant across exams, then the RAG layer is not blocked by
model capability -- it was never carrying field-dependent information in the
first place, and a model with 1 usable health field cannot make it so.

This measures the size of the advice space, and whether the advice sentence
moves with the two fields that survived leave-one-archive-out.

Reads reports and the label manifest only. No image, no model. Locked cases
are counted separately and never used for any fit.
"""
import argparse
import collections
import json
import os
import re

import pandas as pd

from scan_rtf_disease_endpoints import EXAM_PAT, rtf_text

# an advice sentence is one that tells the patient/doctor to DO something
ADVICE_CUES = ["建议", "复查", "随诊", "进一步", "治疗", "注意", "戒烟", "休息",
               "锻炼", "运动", "饮食", "遵医嘱", "检查"]
# the two fields that passed all three leave-one-archive-out directions
LOAO_SURVIVORS = ["clarity", "exudation"]
MISSING = {"nan", "", "None", "NaN"}

# Which finding does an advice sentence cite as its reason? The vocabulary is
# the operator's, taken from the sentences actually present in the reports.
# The verdict column is this project's own measured status for that field, so
# the cross-tab answers: is the advice driven by fields we can deliver?
FINDING_VOCAB = {
    "microthrombus": (["白微栓", "微栓"], "FAIL dev CI includes 0 (+0.077); "
                                         "-0.105 on archive1 under LOAO"),
    "subpapillary_venous_plexus": (["静脉丛扩张", "静脉丛"],
                                   "FAIL LOAO archive2 (+0.118 CI incl 0)"),
    "papilla": (["乳头形状", "顶端膨大", "乳头"],
                "FAIL 3-class +0.027 CI[-0.070,+0.130]"),
    "rbc_aggregation": (["红细胞聚集", "聚集"], "FAIL delta -0.011"),
    "crossing_malformation": (["交叉管袢", "畸形"],
                              "FAIL crossing -0.011; malformation CI incl 0; "
                              "segmenter correlation sign INVERTED"),
    "flow_state": (["血流缓慢", "血流速度", "流速"],
                   "FAIL 7-class 0.330 BELOW majority 0.341"),
    "capillary_count": (["管袢减少", "管袢数", "数量"],
                        "FAIL +0.050 CI[-0.011,+0.110]"),
    "afferent_efferent_diameter": (["管径变细", "管径纤细", "管径比例"],
                                   "FAIL MAE > label granularity"),
    "clarity": (["清晰度", "清晰", "模糊"], "PASS LOAO 3/3 (dev +0.330)"),
    "exudation": (["渗出", "血浆成分"], "PASS LOAO 3/3 (dev +0.240)"),
}


def split_sentences(t):
    out = []
    for s in re.split(r"[。；;\n]", t):
        s = re.sub(r"\s+", "", s).strip()
        # drop the font table that leads every decoded report
        s = re.sub(r"^[^，。]*?(宋体|楷体|黑体|Times New Roman|_GB2312|;)+", "", s)
        s = s.replace("\x00", "")
        if len(s) > 3:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    dev = set(m.exam_case_id[m.evaluation_role == "development"])
    locked = set(m.exam_case_id[m.evaluation_role == "locked_test"])
    labels = m.set_index("exam_case_id")

    reports = []
    for arch in ["recovered_archive1", "recovered_archive2",
                 "recovered_archive3"]:
        for root, _d, files in os.walk(os.path.join(a.data_root, arch)):
            if "__MACOSX" in root:
                continue
            for fn in files:
                if fn.lower().endswith(".rtf") and not fn.startswith("._"):
                    reports.append(os.path.join(root, fn))

    advice_by_exam = collections.defaultdict(set)
    advice_counter = collections.Counter()
    n_with_advice = 0
    n_locked_files = 0
    for p in reports:
        mo = EXAM_PAT.search(p)
        exam = f"{mo.group(1)}/{mo.group(2)}" if mo else None
        if exam in locked:
            n_locked_files += 1
        t = rtf_text(p)
        adv = [s for s in split_sentences(t)
               if any(c in s for c in ADVICE_CUES)]
        if adv:
            n_with_advice += 1
        for s in adv:
            advice_counter[s] += 1
            if exam:
                advice_by_exam[exam].add(s)

    # does the advice depend on the two surviving fields?
    dep = {}
    for fld in LOAO_SURVIVORS:
        if fld not in labels.columns:
            dep[fld] = "FIELD NOT IN MANIFEST"
            continue
        tab = collections.defaultdict(collections.Counter)
        for exam, advs in advice_by_exam.items():
            if exam not in dev:
                continue
            v = str(labels.at[exam, fld]) if exam in labels.index else "nan"
            if v in MISSING:
                continue
            for s in advs:
                tab[v][s] += 1
        # the question: is the top advice the SAME string for every label value?
        tops = {v: (c.most_common(1)[0] if c else None) for v, c in tab.items()}
        dep[fld] = {
            "n_dev_exams_with_advice_per_label_value":
                {v: int(sum(c.values())) for v, c in tab.items()},
            "most_common_advice_per_label_value":
                {v: (t[0], int(t[1])) for v, t in tops.items() if t},
            "top_advice_identical_across_label_values":
                len({t[0] for t in tops.values() if t}) <= 1,
        }

    # Which finding does the advice cite as its reason? Counted per
    # (exam, sentence) pair, each pair once. One sentence can cite more than
    # one field, so the column sums to more than the pair count -- it is a
    # ranking of which findings drive advice, not a partition.
    cited = collections.Counter()
    cited_exams = collections.defaultdict(set)
    uncited_pairs = 0
    total_pairs = 0
    for exam, advs in advice_by_exam.items():
        for s in advs:
            total_pairs += 1
            hit = False
            for fld, (vocab, _v) in FINDING_VOCAB.items():
                if any(w in s for w in vocab):
                    cited[fld] += 1
                    cited_exams[fld].add(exam)
                    hit = True
            if not hit:
                uncited_pairs += 1
    reason_rows = []
    for fld, (vocab, verdict) in FINDING_VOCAB.items():
        reason_rows.append({
            "field_cited_as_the_reason": fld,
            "vocabulary": vocab,
            "n_exam_sentence_pairs_citing_it": int(cited[fld]),
            "n_exams_whose_advice_cites_it": len(cited_exams[fld]),
            "share_of_exams_with_advice": round(
                len(cited_exams[fld]) / max(len(advice_by_exam), 1), 4),
            "this_projects_measured_status_for_that_field": verdict,
        })
    reason_rows.sort(key=lambda r: -r["n_exams_whose_advice_cites_it"])

    n_exams = len(advice_by_exam)
    out = {
        "question": "does the report's own advice text depend on the field "
                    "values, or is it boilerplate?",
        "n_reports_scanned": len(reports),
        "n_report_files_belonging_to_locked_cases": n_locked_files,
        "locked_used_for_any_fit": False,
        "n_reports_containing_any_advice_sentence": n_with_advice,
        "n_exams_with_any_advice_sentence": n_exams,
        "n_distinct_advice_sentences": len(advice_counter),
        "n_advice_sentence_occurrences": int(sum(advice_counter.values())),
        "top_25_advice_sentences": [
            {"text": s, "n_occurrences": int(n)}
            for s, n in advice_counter.most_common(25)],
        "share_of_occurrences_in_top_5": round(
            sum(n for _s, n in advice_counter.most_common(5))
            / max(sum(advice_counter.values()), 1), 4),
        "dependence_on_loao_surviving_fields": dep,
        "which_field_the_advice_cites_as_its_reason": reason_rows,
        "n_exam_sentence_pairs_total": total_pairs,
        "n_exam_sentence_pairs_citing_no_field_in_the_vocabulary":
            uncited_pairs,
        "how_to_read_this": [
            "a small n_distinct_advice_sentences with a high "
            "share_of_occurrences_in_top_5 means the advice is a template "
            "menu, not a function of the findings.",
            "top_advice_identical_across_label_values TRUE for a field means "
            "knowing that field does not change what the operator advises, so "
            "predicting it perfectly would not change the advice either.",
            "this says nothing about whether the advice is clinically "
            "correct. It only bounds how much a field-prediction model could "
            "contribute to generating it.",
            "which_field_the_advice_cites_as_its_reason is the decisive table "
            "for the RAG question: if the advice overwhelmingly cites fields "
            "whose measured status is FAIL, then a RAG layer fed only by the "
            "deliverable fields cannot reproduce that advice, no matter how "
            "well the text generation works.",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("reports scanned:", len(reports),
          "| with advice:", n_with_advice,
          "| exams:", n_exams)
    print("distinct advice sentences:", len(advice_counter),
          "over", int(sum(advice_counter.values())), "occurrences")
    print("share of occurrences in top 5:",
          out["share_of_occurrences_in_top_5"])
    print("\ntop 15 advice sentences:")
    for s, n in advice_counter.most_common(15):
        print("  %4d  %s" % (n, s[:80]))
    print("\nwhich field the advice cites as its reason:")
    print("  (%d exam-sentence pairs over %d exams with advice)"
          % (total_pairs, len(advice_by_exam)))
    print("  %-28s %6s %6s %7s  %s"
          % ("field", "pairs", "exams", "share", "measured status"))
    for r in reason_rows:
        print("  %-28s %6d %6d %7.3f  %s" % (
            r["field_cited_as_the_reason"],
            r["n_exam_sentence_pairs_citing_it"],
            r["n_exams_whose_advice_cites_it"],
            r["share_of_exams_with_advice"],
            r["this_projects_measured_status_for_that_field"][:52]))
    print("  citing nothing in the vocabulary:", uncited_pairs, "pairs")

    print("\ndependence on the two LOAO survivors:")
    for fld, v in dep.items():
        print(" ", fld, json.dumps(v, ensure_ascii=False)[:400])
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
