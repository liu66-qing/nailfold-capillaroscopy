"""Interpretation and advice for the final report. Template text, no LLM.

The findings are grouped into themes. Each theme card has three parts: what was
seen, what it is commonly seen with, and what to do. The rules for the wording:
  - titles describe the image, never the reader's body ("管径偏细", not "手凉");
    a body sensation is offered only as something to check ("如果您平时…")
  - no absolute claims ("说明…", "最直接的办法")
  - at most MAX_ACTIONS actions per report, each tagged with the findings it is for
  - a blurry capture is described as a capture condition, and the headline says
    the results come from the clearer part
Themes are ordered by how strong their findings are. That strength is used only
for ordering and is never printed.
Nothing names a disease, a treatment, a drug or a supplement.
"""
from __future__ import annotations

import re
from typing import Any

from .final_registry import FORBIDDEN

THIN = ("afferent_diameter", "efferent_diameter", "apex_diameter")
MAX_ACTIONS = 5

# plain explanation added the first time a technical item name is used
GLOSS = {
    "afferent_diameter": ("输入枝", "输入枝（血液流入的一侧）"),
    "efferent_diameter": ("输出枝", "输出枝（血液流出的一侧）"),
    "apex_diameter": ("袢顶", "袢顶（管袢顶端的弯曲处）"),
}

A = dict(
    photo="重拍前在室温下静坐 10~15 分钟，用温水洗手擦干；拍摄时手指平放、镜头轻贴不按压，"
          "在甲襞处滴一小滴观察油，同一根手指多拍几个位置。",
    warm="手部保暖：天冷出门戴手套，洗手、洗碗尽量用温水，空调房里别让手长时间对着出风口。",
    finger="每天做 3 次手指操：握拳—张开 20 次，再从指根向指尖轻轻按揉每根手指约 1 分钟。",
    bath="睡前用 38~40 ℃ 的温水泡手 5~10 分钟，水温不要过烫。",
    smoke="如果吸烟，尽量减少或戒掉；烟草会让指尖小血管收缩。",
    calm="少喝浓咖啡、浓茶；紧张时做几次慢呼吸（吸气 4 秒、呼气 6 秒）。",
    sit="每坐 1 小时起身走动几分钟，并把双手举过头顶做几次握拳，避免手长时间下垂。",
    wrist="久坐看屏幕时，每 20 分钟做一次手腕绕圈和脚踝勾绷，各 10 次。",
    water="每天喝足水（没有医生限水要求时约 1.5~2 升），少喝含糖饮料。",
    sleep="作息尽量规律，少熬夜。",
    aerobic="每周累计约 150 分钟中等强度运动（快走、骑车、游泳都可以），分几天完成。",
    diet="饮食清淡一些：少油炸、少甜食和加工肉，多吃蔬菜、全谷物和豆类。",
    cuticle="保护甲周皮肤：不撕倒刺、不剪或推甲周死皮，做美甲时避开甲根处理。",
    cream="洗手后涂护手霜，秋冬每天多涂几次，减少甲周干裂。",
    rest_hand="减少频繁接触洗涤剂和冷水的手部劳作，必要时戴手套操作。",
    recheck="1~3 个月后，在相同时段、相同室温下用同一根手指再拍一次，对比这几项的变化。",
    keep="保持现在的运动和作息习惯；天冷时注意手部保暖。",
    yearly="每 6~12 个月在相同条件下复拍一次，留作自己的对照。",
)

HOW_TO_READ = "单次拍摄会受室温、手指和按压影响，在同样条件下多次拍摄，看变化趋势更有意义。"


def _on(fields: dict, f: str, value: str | None = None) -> bool:
    # a row feeds the advice only when its calibrated level is at least the middle one;
    # weak rows still print in the table but never drive a recommendation
    d = fields[f]
    if (d.get("confidence") or 0) < 1:
        return False
    return bool(d.get("deviates")) and (value is None or d.get("value") == value)


def _label(fields: dict, f: str, gloss: bool = False) -> str:
    item = fields[f]["item"]
    if gloss and f in GLOSS:
        item = item.replace(*GLOSS[f])
    return item + fields[f]["value"]


def _theme(key, title, triggers, meaning, actions, hint):
    """actions: list of (action key, findings it is for)."""
    return dict(theme=key, title=title, triggers=triggers, meaning=meaning,
                candidates=actions, hint=hint)


def _theme_photo(fields):
    if not _on(fields, "clarity"):
        return None
    return _theme("photo", "部分图像不够清晰", ["clarity"],
                  "画面偏模糊时，管袢边缘和血色细节不容易看清，多与拍摄时指尖温度、"
                  "按压和对焦有关。按下面的方法重拍一组，可以和这次的结果对比。",
                  [("photo", ["clarity"])], "")


def _theme_cold(fields):
    dark = _on(fields, "blood_color")
    thin = [f for f in THIN if _on(fields, f, "偏细")]
    few = _on(fields, "capillary_count")
    if not (dark or thin or few):
        return None
    ids = (["blood_color"] if dark else []) + thin + (["capillary_count"] if few else [])
    title = "、".join((["管径偏细"] if thin else []) + (["血色偏暗"] if dark else [])
                     + (["可见管袢偏少"] if few else []))
    parts = []
    if thin:
        parts.append("管径偏细常见于指尖小血管处在收缩状态时，受冷、紧张、咖啡因都会让它收紧")
    if dark:
        parts.append("血色偏暗常见于指尖局部血流偏慢时，刚接触冷水或吸烟后也容易出现")
    if few:
        parts.append("手指偏凉时，一部分管袢里的血流减少，拍到的管袢就会偏少")
    acts = [("warm", ids)]
    if dark:
        acts.append(("smoke", ["blood_color"]))
    if thin:
        acts.append(("calm", thin))
    acts += [("finger", ids), ("bath", ids)]
    return _theme("cold", title, ids,
                  "。".join(parts) + "。如果您平时手脚容易凉，这与本次所见一致；"
                  "这类表现和环境、习惯关系密切，保暖和活动后常会有变化。",
                  acts, "常与手部受冷、小血管收缩有关")


def _theme_slow(fields):
    # flow_state / rbc_aggregation are not used: in validation they collapse to the
    # common answer, so they must not drive advice
    thick = [f for f in THIN if _on(fields, f, "偏粗")]
    long_ = _on(fields, "loop_length", "偏长")
    if not (thick or long_):
        return None
    ids = thick + (["loop_length"] if long_ else [])
    title = "、".join((["管径偏粗"] if thick else []) + (["管袢偏长"] if long_ else []))
    parts = []
    if thick:
        parts.append("管径偏粗常见于指尖血液回流偏慢时，久坐、手长时间下垂或刚用热水泡手后更明显")
    if long_:
        parts.append("管袢偏长常见于指尖小血管长时间处在充盈状态时")
    acts = [("sit", ids), ("wrist", ids), ("aerobic", ids), ("water", ids)]
    if thick:
        acts.append(("sleep", thick))
    return _theme("slow", title, ids,
                  "。".join(parts) + "。如果您平时久坐较多，这与本次所见一致；"
                  "多起身活动有助于改善末梢回流。",
                  acts, "常与久坐、手长时间下垂有关")


def _theme_shape(fields):
    cross = _on(fields, "crossing_ratio")
    malf = _on(fields, "malformation_ratio")
    short = _on(fields, "loop_length", "偏短")
    if not (cross or malf or short):
        return None
    ids = (["crossing_ratio"] if cross else []) + (["malformation_ratio"] if malf else []) \
        + (["loop_length"] if short else [])
    title = "、".join((["交叉管袢较多"] if cross else []) + (["形态不规则的管袢较多"] if malf else [])
                     + (["管袢偏短"] if short else []))
    parts = []
    if cross or malf:
        parts.append("交叉、扭曲的管袢在健康人里也常有一部分；比例偏高时，常与甲周皮肤反复受损、"
                     "干燥有关，也可能是个人本来的形态")
    if short:
        parts.append("管袢偏短常与甲周皮肤较厚、干燥或近期修剪甲周有关")
    return _theme("shape", title, ids,
                  "。".join(parts) + "。形态变化比较慢，隔一段时间对比才看得出趋势。",
                  [("cuticle", ids), ("cream", ids), ("rest_hand", ids), ("recheck", ids)],
                  "多与甲周皮肤状态有关")


def _theme_env(fields):
    ex = _on(fields, "exudation")
    mt = _on(fields, "microthrombus")
    if not (ex or mt):
        return None
    ids = (["exudation"] if ex else []) + (["microthrombus"] if mt else [])
    title = "、".join((["管袢周围有渗出样表现"] if ex else [])
                     + (["可见白色微粒样片段"] if mt else []))
    parts = []
    if ex:
        parts.append("渗出样表现会让画面显得发雾，常与局部组织水分偏多、近期手部受冷热刺激有关")
    if mt:
        parts.append("白色微粒样片段是血细胞短暂聚在一起时的样子，常见于那一段血流偏慢时，"
                     "多数会随血流散开")
    return _theme("env", title, ids,
                  "。".join(parts) + "。规律运动、清淡饮食和充足睡眠对这类表现有帮助。",
                  [("aerobic", ids), ("diet", ids), ("water", ids), ("recheck", ids)],
                  "常与近期作息、饮食有关")


THEMES = (_theme_photo, _theme_cold, _theme_slow, _theme_shape, _theme_env)

SEE_DOCTOR = ("如果出现手指遇冷明显发白、发紫，或伴麻木、疼痛、指尖破溃不易愈合，"
              "请带上本报告到医院（风湿免疫科或血管外科）就诊。")
NOT_DIAGNOSIS = "本报告描述的是甲襞图像上的微循环形态特征，用于日常健康管理，不是疾病诊断。"


def _allocate(sections: list[dict], fields: dict) -> None:
    """Round-robin over themes in order, at most MAX_ACTIONS in total, no repeats."""
    used: set[str] = set()
    total = 0
    for s in sections:
        s["actions"] = []
    queues = [list(s["candidates"]) for s in sections]
    while total < MAX_ACTIONS and any(queues):
        for s, q in zip(sections, queues):
            while q and q[0][0] in used:
                q.pop(0)
            if q and total < MAX_ACTIONS:
                key, ids = q.pop(0)
                used.add(key)
                total += 1
                s["actions"].append(dict(
                    text=A[key], **{"for": "、".join(fields[f]["item"] + fields[f]["value"]
                                                    for f in ids)}))
    for s in sections:
        del s["candidates"]


def compose_advice(fields: dict[str, Any]) -> dict[str, Any]:
    sections = [s for s in (t(fields) for t in THEMES) if s]
    photo = [s for s in sections if s["theme"] == "photo"]
    rest = [s for s in sections if s["theme"] != "photo"]
    # strongest findings first; the strength itself is never printed
    rest.sort(key=lambda s: -max((fields[f].get("probability") or 0) for f in s["triggers"]))
    sections = photo + rest
    glossed: set[str] = set()
    for s in sections:
        labels = []
        for f in s["triggers"]:
            labels.append(_label(fields, f, gloss=f not in glossed))
            glossed.add(f)
        s["seen"] = "；".join(labels)
    lead = ("本次部分图像清晰度不足，以下结论来自其中较清晰的部分，建议按文中方法重拍后对比。"
            if photo else "")
    if not rest:
        headline = lead + "整体形态接近多数人的常见表现，保持现在的生活习惯就好。"
        if not photo:
            sections = [dict(theme="keep", title="各项形态接近常见表现", triggers=[],
                             seen="管袢形态、血色和周围环境都接近多数人的常见表现",
                             meaning="这次没有看到需要特别留意的变化。",
                             candidates=[("keep", []), ("aerobic", []), ("yearly", [])])]
    else:
        top = rest[0]
        main = top["title"].split("、")[0]
        whole = "整体形态大致正常" if len(rest) <= 2 else "这次有几处形态变化"
        headline = lead + "%s，最值得留意的是%s，%s。" % (whole, main, top["hint"])
    _allocate(sections, fields)
    for s in sections:
        s.pop("hint", None)
    out = dict(headline=headline, sections=sections, how_to_read=HOW_TO_READ,
               see_doctor=SEE_DOCTOR, disclaimer=NOT_DIAGNOSIS, generated_by="template")
    check_text(out)
    return out


def iter_text(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("field_id", "generated_by", "theme", "triggers"):
                yield from iter_text(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_text(v)


def forbidden_hits(text: str) -> list[str]:
    hits = [w for w in FORBIDDEN if w != "um" and w in text]
    if re.search(r"\d\s*um\b|\bum\b", text):
        hits.append("um")
    return hits


def check_text(obj: Any) -> None:
    for t in iter_text(obj):
        bad = forbidden_hits(t)
        if bad:
            raise ValueError("forbidden wording %s in: %s" % (bad, t[:80]))
