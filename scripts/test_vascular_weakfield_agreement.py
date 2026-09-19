#!/usr/bin/env python
"""Do the 血管数据集 weak-field labels AGREE with our clinical labels?

The dataset supplies 582 image-level labels for five fields that overlap our
manifest: SVP, red-cell aggregation, papilla, sweat duct, hemorrhage. 96 of
those images hash-match images belonging to 50 of our 186 development cases.

That overlap is the only way to ask the question that decides whether this
dataset is usable supervision: on the SAME image, does the external annotator
say what our clinical label says? If they disagree, the dataset is not extra
supervision for our task -- it is a different task wearing the same field
names, and adding it would inject label noise rather than signal.

Also compares the marginal prevalence of each field, because a large
prevalence gap on the same field is itself evidence of a different operational
definition.

Reads the Excel, the audit's hash mapping, and the manifest. No image is
opened, no model runs. Locked-mapped images (32) are reported as a count only
and never compared, since comparing them would read locked truth.
"""
import argparse
import json
import os

import pandas as pd

# Excel column -> our manifest column
PAIRS = [
    ("乳头下静脉丛", "subpapillary_venous_plexus"),
    ("红细胞聚集", "rbc_aggregation"),
    ("乳头", "papilla"),
    ("汗腺导管", "sweat_duct"),
    ("血管是否出血", "hemorrhage"),
]
# map both sides onto a shared 3-value space so they can be compared at all.
# 'not_visible' is kept separate from 'normal' on purpose: collapsing it would
# manufacture agreement.
EXT_MAP = {
    "正常": "normal", "扩张": "abnormal", "不可见": "not_visible",
    "聚集": "abnormal", "异常": "abnormal", "可见": "abnormal",
    "出血": "abnormal",
}
MISSING = {"nan", "", "None", "NaN", "-"}


def our_map(field, v):
    """Map our clinical label text onto normal/abnormal/not_visible.

    Deliberately conservative: anything not recognised returns None and is
    excluded rather than guessed into a class.
    """
    v = str(v).strip()
    if v in MISSING:
        return None
    if v.startswith("[") or "/" in v or "指甲襞" in v:
        return None  # unit strings and bracketed template text
    if field == "subpapillary_venous_plexus":
        # our vocabulary is 不见 / 可见1排 / 可见2排 / >2排,扩张
        if v == "不见":
            return "not_visible"
        if v in ("可见1排", "可见2排"):
            return "normal"
        if "扩张" in v or "排" in v and ">" in v:
            return "abnormal"
        return None
    if field == "rbc_aggregation":
        # 无 / 轻度 / 中度 / 重度
        if v == "无":
            return "normal"
        if v in ("轻度", "中度", "重度"):
            return "abnormal"
        return None
    if field == "papilla":
        # 波纹状 / 浅波纹状 / 平坦 -- 波纹状 is the normal morphology
        if v == "波纹状":
            return "normal"
        if v in ("浅波纹状", "平坦"):
            return "abnormal"
        return None
    if field == "sweat_duct":
        # our label is a COUNT range per nailfold, not a visibility judgement.
        # 0--2 spans 'none visible' and 'two visible', so it cannot be mapped
        # onto the external visible/not_visible axis without inventing a cut.
        return None
    if field == "hemorrhage":
        # 无 / 1--2 / >2 (count of haemorrhagic loops)
        if v == "无":
            return "normal"
        if v[0].isdigit() or v.startswith(">"):
            return "abnormal"
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel",
                    default="data/血管数据集/血管分类标注/血管分类标注.xlsx")
    ap.add_argument("--mapping", default="artifacts/audits/"
                    "vascular_dataset_governance_20260830/source_case_mapping.csv")
    ap.add_argument("--manifest",
                    default="artifacts/manifest/locked_evaluation_v1.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    x = pd.read_excel(a.excel)
    x["图片id"] = x["图片id"].astype(int)
    smap = pd.read_csv(a.mapping)
    m = pd.read_csv(a.manifest)
    m["exam_case_id"] = m.exam_case_id.astype(str)
    lab = m.set_index("exam_case_id")
    dev = set(m.exam_case_id[m.evaluation_role == "development"])

    dev_rows = smap[smap.source_mapping_status == "EXACT_RECOVERED_DEVELOPMENT"]
    n_locked_mapped = int((smap.source_mapping_status
                           == "EXACT_RECOVERED_LOCKED").sum())

    # image -> our case (development only)
    img_case = {}
    for _i, r in dev_rows.iterrows():
        cs = [c.strip() for c in str(r.development_cases).replace(";", ",")
              .split(",") if c.strip()]
        if len(cs) == 1 and cs[0] in dev:
            img_case[int(r.original_id)] = cs[0]

    ext = x.set_index("图片id")
    results = []
    for ecol, ocol in PAIRS:
        if ocol not in lab.columns:
            results.append({"field": ocol, "status": "NOT IN MANIFEST"})
            continue
        rows = []
        for img, case in img_case.items():
            if img not in ext.index:
                continue
            e = EXT_MAP.get(str(ext.at[img, ecol]).strip())
            o = our_map(ocol, lab.at[case, ocol])
            if e is None or o is None:
                continue
            rows.append((case, e, o))
        # case level: an exam is 'abnormal' externally if ANY of its images is
        by_case = {}
        for case, e, o in rows:
            prev = by_case.get(case)
            rank = {"normal": 0, "not_visible": 1, "abnormal": 2}
            if prev is None or rank[e] > rank[prev[0]]:
                by_case[case] = (e, o)
            else:
                by_case[case] = (prev[0], o)
        agree = sum(1 for e, o in by_case.values() if e == o)
        n = len(by_case)
        # chance agreement, so a 'rate' cannot be read as a good number when
        # both sides are dominated by the same class
        chance = 0.0
        if n:
            for cls in ("normal", "abnormal", "not_visible"):
                pe = sum(1 for e, _o in by_case.values() if e == cls) / n
                po = sum(1 for _e, o in by_case.values() if o == cls) / n
                chance += pe * po
        kappa = ((agree / n - chance) / (1 - chance)
                 if n and chance < 1 else None)
        conf = {}
        for e, o in by_case.values():
            conf["ext=%s|ours=%s" % (e, o)] = conf.get(
                "ext=%s|ours=%s" % (e, o), 0) + 1
        # marginal prevalence of 'abnormal' on each side, all rows
        ext_all = [EXT_MAP.get(str(v).strip()) for v in x[ecol]]
        ext_all = [v for v in ext_all if v]
        our_all = [our_map(ocol, v) for v in m[m.evaluation_role
                                               == "development"][ocol]]
        our_all = [v for v in our_all if v]
        results.append({
            "field": ocol,
            "external_column": ecol,
            "n_development_cases_comparable": n,
            "n_agree": agree,
            "case_level_agreement": round(agree / n, 4) if n else None,
            "chance_agreement": round(chance, 4) if n else None,
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "confusion": conf,
            "external_abnormal_share_all_582":
                round(sum(v == "abnormal" for v in ext_all)
                      / max(len(ext_all), 1), 4),
            "our_abnormal_share_development":
                round(sum(v == "abnormal" for v in our_all)
                      / max(len(our_all), 1), 4),
            "external_not_visible_share_all_582":
                round(sum(v == "not_visible" for v in ext_all)
                      / max(len(ext_all), 1), 4),
            "our_not_visible_share_development":
                round(sum(v == "not_visible" for v in our_all)
                      / max(len(our_all), 1), 4),
            "this_projects_measured_status": {
                "subpapillary_venous_plexus": "FAIL LOAO archive2",
                "rbc_aggregation": "FAIL delta -0.011",
                "papilla": "FAIL 3-class CI includes 0",
                "sweat_duct": "FAIL 2 positives, arithmetically impossible",
                "hemorrhage": "FAIL collapsed to one class, AUROC 0.331",
            }[ocol],
            "label_space_note": {
                "subpapillary_venous_plexus":
                    "ours counts rows of vessels (不见/可见1排/可见2排/>2排,扩张); "
                    "external is 不可见/正常/扩张. Mapped 可见1排+可见2排 -> normal.",
                "rbc_aggregation":
                    "ours is severity (无/轻度/中度/重度); external is binary "
                    "正常/聚集. Any severity -> abnormal.",
                "papilla":
                    "ours is morphology (波纹状/浅波纹状/平坦); external is "
                    "正常/异常/不可见. 波纹状 -> normal.",
                "sweat_duct":
                    "NOT COMPARABLE. Ours is a count range per nailfold "
                    "(0--2, 3--4); external is visible/not_visible. 0--2 "
                    "spans both, so no mapping exists without inventing a "
                    "cut. Excluded rather than guessed.",
                "hemorrhage":
                    "ours counts haemorrhagic loops (无/1--2/>2); external is "
                    "binary 出血/正常.",
            }[ocol],
        })

    out = {
        "question": "are the 血管数据集 weak-field labels the same supervision "
                    "as our clinical labels, on the same images?",
        "n_external_rows": len(x),
        "n_images_hash_matched_to_our_development_cases": len(img_case),
        "n_images_hash_matched_to_locked_cases_EXCLUDED": n_locked_mapped,
        "locked_truth_read": False,
        "results": results,
        "how_to_read_this": [
            "case_level_agreement is generous to the dataset: an exam counts "
            "as externally abnormal if ANY of its matched images is, and "
            "unmappable label text is dropped rather than guessed.",
            "a large gap between external_abnormal_share and our share on the "
            "same field is evidence of a different operational definition, "
            "independent of the agreement number.",
            "this is not an inter-rater reliability study. The external "
            "annotator labelled crops; our label is an exam-level clinical "
            "answer. Low agreement means they are not interchangeable, not "
            "that either is wrong.",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print("external rows:", len(x),
          "| images matched to our development cases:", len(img_case),
          "| locked-matched images excluded:", n_locked_mapped)
    print("\n%-28s %4s %6s %7s %7s  %7s %7s"
          % ("field", "n", "rate", "chance", "kappa", "ext_abn", "our_abn"))
    for r in results:
        if r.get("status") == "NOT IN MANIFEST":
            print("%-28s  NOT IN MANIFEST" % r["field"])
            continue
        print("%-28s %4d %6s %7s %7s  %7s %7s" % (
            r["field"], r["n_development_cases_comparable"],
            r["case_level_agreement"], r["chance_agreement"],
            r["cohen_kappa"], r["external_abnormal_share_all_582"],
            r["our_abnormal_share_development"]))
    print("\nconfusions:")
    for r in results:
        if "confusion" in r:
            print(" ", r["field"], r["confusion"])
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
