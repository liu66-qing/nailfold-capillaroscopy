"""The 20 printed rows of the final report (expert_router_v1).

Row kinds:
  binary    one calibrated head; answer = q > 0.5
  band3     two one-sided calibrated heads (below / above the report-sheet band
            edge); P(ref) = 1 - P(lo) - P(hi), answer = argmax
  papilla   flat / wavy heads composed the same way (shallow-wavy in between)
  fixed     the answer given by >= 85% of the 233 labelled cases; the image is
            not used and the row carries a footnote saying so
  derived   wording built from the afferent and efferent rows

Confidence of the printed answer = its calibrated probability c:
  c >= 0.70 较高, 0.60 <= c < 0.70 中等, below 较低 (band3/papilla: 0.70 / 0.55).
No row prints a size, a count, a rate, a score or a severity grade.
"""
from __future__ import annotations

BUNDLE_SHA256 = "04ee6b80a88bb2e1ff2370906262dddf46e1e27d2f889f8c745be4bb41b17241"
ENCODER_SHA256 = "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b"
SEG_SHA256 = "08103c92b937c0e94086d735c973da7deab069591af62dc3f677ed6cc66838bb"
DET_SHA256 = "0ca4c3bb952f3a8779f25ef8bcbd852745ea24bce29cd2d66e0f17906823888f"
RELEASE_ID = "expert-router-final"
SCHEMA_VERSION = "nailfold-report/2.0"

# field, printed name, kind, spec, reference wording
#   binary : (target, (answer0, answer1), flagged class or None)
#   band3  : (lo target, hi target, (lo, ref, hi) answers)
FIELDS = (
    ("clarity", "清晰度", "binary", ("clarity", ("清晰", "欠清晰"), 1), "清晰"),
    ("capillary_count", "管袢数", "binary", ("capillary_lo", ("适中", "偏少"), 1),
     "适中"),
    ("afferent_diameter", "输入枝管径", "band3",
     ("afferent_lo", "afferent_hi", ("偏细", "适中", "偏粗")), "适中"),
    ("efferent_diameter", "输出枝管径", "band3",
     ("efferent_lo", "efferent_hi", ("偏细", "适中", "偏粗")), "适中"),
    ("output_input_ratio", "输出/输入枝", "derived", None, "输出枝略粗于输入枝"),
    ("apex_diameter", "袢顶管径", "band3",
     ("apex_lo", "apex_hi", ("偏细", "适中", "偏粗")), "适中"),
    ("loop_length", "管袢长度", "band3",
     ("loop_lo", "loop_hi", ("偏短", "适中", "偏长")), "适中"),
    ("crossing_ratio", "交叉管袢", "binary", ("crossing", ("比例不高", "比例较高"), 1),
     "比例不高"),
    ("malformation_ratio", "不规则管袢", "binary",
     ("malformation", ("比例不高", "比例较高"), 1), "比例不高"),
    ("flow_state", "流态", "binary",
     ("flow_state", ("线流/线粒流", "粒线流/粒流"), None), "—"),
    ("vasomotion", "血管运动性", "fixed", "低频", "低频"),
    ("rbc_aggregation", "红细胞聚集", "binary",
     ("rbc_aggregation", ("无", "有"), None), "—"),
    ("wbc_count", "白细胞", "fixed", "未见增多", "未见增多"),
    ("microthrombus", "白色微粒样片段", "binary", ("microthrombus", ("未见", "可见"), 1),
     "未见"),
    ("blood_color", "血色", "binary", ("blood_color", ("淡红/浅红", "暗红/暗紫"), 1),
     "淡红/浅红"),
    ("exudation", "渗出", "binary", ("exudation", ("无", "有"), 1), "无"),
    ("hemorrhage", "出血", "fixed", "无", "无"),
    ("subpapillary_venous_plexus", "乳头下静脉丛", "binary",
     ("subpapillary_venous_plexus", ("未见", "可见"), None), "—"),
    ("papilla", "乳头", "papilla", ("papilla_wavy", "papilla_flat",
                                   ("波纹状", "浅波纹状", "平坦")), "—"),
    ("sweat_duct", "汗腺导管", "fixed", "少", "少"),
)
FIELD_IDS = tuple(f[0] for f in FIELDS)
BINARY_TARGETS = tuple(sorted({t for f in FIELDS if f[2] in ("binary", "band3", "papilla")
                               for t in ((f[3][0],) if f[2] == "binary" else f[3][:2])}))

CONF_WORDS = ("较低", "中等", "较高")

# phrases no printed value or advice sentence may contain
FORBIDDEN = ("未评估", "μm", "um", "条/mm", "次/min", "个/15s", "个/min", "积分", "总分",
             "重度异常", "中度异常", "轻度异常", "诊断为", "确诊", "疏通", "抗栓", "溶栓",
             "保健品", "服用", "药物",
             "参考范围", "把握", "仅供参考", "模型", "档位")


def confidence(c: float, kind: str) -> int:
    hi, mid = (0.70, 0.60) if kind == "binary" else (0.70, 0.55)
    return 2 if c >= hi else (1 if c >= mid else 0)


def derived_ratio(eff_band: int, aff_band: int) -> str:
    if eff_band > aff_band:
        return "输出枝相对偏粗"
    if eff_band < aff_band:
        return "输入枝相对偏粗"
    return "两侧粗细相当"
