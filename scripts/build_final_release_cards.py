"""Build the per-field validation cards of the final release from saved results.

No model is fitted and no image is read. Sources:
  15 model rows : artifacts/experiments/test_locked47_all_fields/{all_fields.json,
                  per_case.csv}. A0 fit on dev 186, scored once on locked-47.
                  locked-47 had been read before and selection was done on it,
                  so these numbers are descriptive only.
  4 baseline rows: artifacts/evidence/allfields_20260919/all_fields_status.json
                  (development OOF). The A0 history is kept apart from the card's
                  own numbers, because the printed value is the fixed baseline
                  and not that model.
  1 derived row : no statistic of its own.

Exact McNemar is computed here against the dev-mode answer; it is named apart
from any permutation p value.

Output: release/final_v1/validation_cards.json

    python scripts/build_final_release_cards.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from nailfold_report.final_registry import FIELDS  # noqa: E402

ALL = ROOT / "artifacts/experiments/test_locked47_all_fields/all_fields.json"
PER = ROOT / "artifacts/experiments/test_locked47_all_fields/per_case.csv"
DEVS = ROOT / "artifacts/evidence/allfields_20260919/all_fields_status.json"
OUT = ROOT / "release/final_v1/validation_cards.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def model_card(f, res, per):
    r = res[f]
    x = per[per.field == f]
    mode = r["dev_mode"]
    right_m = x.pred == x.y_true
    right_0 = x.y_true == mode
    b, c = int((right_m & ~right_0).sum()), int((~right_m & right_0).sum())
    recall = {}
    for k in r["classes"]:
        t = x[x.y_true == k]
        recall[str(k)] = None if len(t) == 0 else dict(
            hit=int((t.pred == k).sum()), n=int(len(t)))
    collapsed = len(r["predicted_classes"]) == 1
    missed = [k for k, v in recall.items() if v and v["hit"] == 0]
    note = None
    if collapsed:
        note = "该验证集中模型对所有病例给出同一答案，未识别出少数类"
    elif missed:
        note = "该验证集中以下类别一次也未被正确识别：%s" % ",".join(missed)
    return dict(
        evidence_source="locked47_descriptive", n_evaluable=r["n_test"],
        class_counts=r["test_class_counts"], accuracy=r["accuracy"],
        balanced_accuracy=r["balanced_accuracy"], mode_accuracy=r["mode_accuracy"],
        dev_mode_class=mode, per_class_recall=recall,
        acc_minus_mode_ci95=r["acc_minus_mode_ci95"],
        exact_mcnemar_p=round(mcnemar_exact(b, c), 4),
        exact_mcnemar_discordant=dict(model_only_right=b, baseline_only_right=c),
        predicted_classes=r["predicted_classes"], collapse_note=note,
        beats_baseline_at_005=bool(mcnemar_exact(b, c) < 0.05 and b > c))


def main():
    res = json.loads(ALL.read_text(encoding="utf-8"))["fields"]
    per = pd.read_csv(PER, dtype={"exam_case_id": str})
    dev = json.loads(DEVS.read_text(encoding="utf-8"))["fields"]
    cards = {}
    common_lim = ["locked-47 此前已被多次查看并在其上做过模型选择，数字仅为内部描述性结果，"
                  "偏乐观；不是外部验证或临床验证",
                  "历史一致率描述一组检查中的总体表现，不是本次预测的正确概率"]
    for f, name, src, _, level in FIELDS:
        if src in ("model_v1", "model_supp"):
            c = model_card(f, res, per)
            c.update(source_file=str(ALL.relative_to(ROOT)).replace("\\", "/"),
                     source_sha256=sha(ALL), limitations=list(common_lim))
            if f in ("flow_state", "microthrombus"):
                c["limitations"].append("报告中该项来自动态观察；本系统仅由静态图像给出报告标签关联预测，"
                                        "不是动态事件的检出验证")
        elif src == "fixed_baseline":
            d = dev[f]
            c = dict(evidence_source="dev_oof_fixed_baseline", n_evaluable=d["n"],
                     positives=int(round(d["abnormal_share"] * d["n"])),
                     accuracy=d["baseline_constant"], balanced_accuracy=0.5,
                     mode_accuracy=d["baseline_constant"], per_class_recall=None,
                     acc_minus_mode_ci95=None, exact_mcnemar_p=None,
                     reason_null="固定基线不使用图像，没有个体识别能力可检验",
                     collapse_note="固定基线对所有人给出同一答案，少数类识别为 0",
                     historical_a0_experiment=dict(
                         note="历史 A0 模型实验，不是本行输出的来源",
                         accuracy=d["accuracy"], balanced_accuracy=d["balanced_accuracy"],
                         auroc=d["auroc"], predicted_abnormal_share=d["predicted_abnormal_share"]),
                     source_file=str(DEVS.relative_to(ROOT)).replace("\\", "/"),
                     source_sha256=sha(DEVS),
                     limitations=["开发集 186 例内部估计；阳性例数极少",
                                  "本行不随图像改变，不能用于排除该项异常"])
        else:
            c = dict(evidence_source="derived", n_evaluable=None,
                     reason_null="派生展示，没有独立准确率；不能由两枝准确率推算组合正确率",
                     components=dict(afferent_diameter=res["afferent_diameter"]["accuracy"],
                                     efferent_diameter=res["efferent_diameter"]["accuracy"]),
                     limitations=["报告原始比值是两枝测量值的算术结果，本系统不输出比值"])
        c.update(field_id=f, item=name, source=src, advice_level=level)
        cards[f] = c
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    for f, c in cards.items():
        print(f, c.get("accuracy"), c.get("mode_accuracy"), c.get("exact_mcnemar_p"),
              c.get("collapse_note"))


if __name__ == "__main__":
    main()
