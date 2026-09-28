"""Knowledge cards for the final report's advice text. Looked up by field id; no
vector store, no network at run time.

Term cards paraphrase the definitions in the local training handbook
data/微循环培训资料/微循环知识汇编.doc (converted with Word; chapter 2 "甲襞微循环的
基本知识"). Its disease-association passages ("提示：病人可能…", "…常见") are
deliberately not carried over: the model outputs give no individual evidence
for them. General education cards follow the WHO fact sheets (retrieved
2026-09, URLs kept on the card) and are the same for every person.

Output: release/final_v1/knowledge_cards.json

    python scripts/build_final_knowledge_cards.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release/final_v1/knowledge_cards.json"
HANDBOOK = "《微循环知识手册》（本地培训资料 微循环知识汇编.doc）第二章"

TERMS = {
    "clarity": "清晰度指图像中能否清楚看到管袢的形状。室温、皮肤角质化、皮肤粗糙、手指移动和对焦都会影响清晰度，所以它首先是拍摄质量的提示。",
    "capillary_count": "管袢数是在第一排管袢的计数区内数到的管袢条数。报告单按条数分档；本系统只给出所属档位的模型预测，不给出条数。",
    "afferent_diameter": "输入枝是发夹形管袢中血液流入的一支，管径在管袢中部测量。报告单按测量值分为低于、处于、高于参考范围三档；本系统只给出档位预测，不给出尺寸。",
    "efferent_diameter": "输出枝是发夹形管袢中血液流出的一支，通常比输入枝略粗。本系统只给出三档中的档位预测，不给出尺寸。",
    "output_input_ratio": "报告单中的输出/输入枝比是两枝测量值相除得到的数。本系统没有测量值，因此只并列显示两枝各自的预测档位，不计算比值。",
    "apex_diameter": "袢顶是管袢由输入枝转向输出枝的弯曲顶端。本系统只给出三档中的档位预测，不给出尺寸。",
    "loop_length": "管袢长指管袢从基部到袢顶的长度。本系统只预测它是否超过报告单参考范围的上界档，不给出长度。",
    "crossing_ratio": "交叉管袢指输入枝与输出枝相互交叉的管袢。报告单按交叉管袢所占比例分档；本系统只预测是否超过报告阈值档。",
    "malformation_ratio": "畸形管袢指形态偏离典型发夹形的管袢。报告单按所占比例分档；本系统只预测是否超过报告阈值档。",
    "flow_state": "流态描述管袢内血液流动的样子，从线流、线粒流到粒流、停滞分为多个等级，需要观察动态画面。本系统使用静态图像，只给出与报告标签组的关联预测。",
    "vasomotion": "血管运动性指管袢自发地变宽变细或流速快慢交替的现象，需要按时间观察。静态图像无法提供这一信息，本行显示开发集中最常见的标签。",
    "rbc_aggregation": "红细胞聚集指血流中数个或数十个红细胞集合成团块，报告单分为无、轻、中、重。本系统把有聚集的各级合并为一类，不区分程度。",
    "wbc_count": "白细胞数指一段时间内沿管壁翻滚通过的白细胞个数，需要按时间计数。静态图像无法提供这一信息，本行显示开发集中最常见的标签。",
    "microthrombus": "白微栓指血流中漂过的白色不规则团块，需要在动态画面中辨认。本系统使用静态图像，只给出与报告标签的关联预测，不能检出或排除。",
    "blood_color": "血色指管袢内血液的颜色，报告单区分淡红、浅红、暗红、暗紫。本系统合并为两组给出预测。颜色也受光源、设备和拍摄条件影响。",
    "exudation": "渗出指管袢周围间隙变大、发亮，管袢影像变模糊的表现。本系统预测报告单上是否记为有渗出。",
    "hemorrhage": "出血指红细胞出到管袢外的现象，外伤等局部因素也会造成。本行不使用图像，显示开发集中最常见的标签，不能用来排除出血。",
    "subpapillary_venous_plexus": "乳头下静脉丛是多个管袢汇入的细静脉网，位于管袢下方，部分人群可以看到。本系统预测报告单上记为可见或不见。",
    "papilla": "乳头指甲襞皮肤的真皮乳头轮廓，报告单分为波纹状、浅波纹状、平坦。本系统给出三类中的预测。",
    "sweat_duct": "汗腺导管在管袢之间表现为白色线条或螺旋线条。报告单按个数分档；本行不使用图像，显示开发集中最常见的标签。",
}

EDUCATION = {
    "general_activity": dict(
        title="日常活动",
        text="在身体条件允许的范围内，减少长时间连续坐着，把活动安排进日常生活里。任何活动都比不活动好，所有活动都算数。WHO 对成年人的参考是每周至少 150 分钟中等强度活动；具体安排请结合自己的年龄和身体情况，已有医生制定的计划时以该计划为准。",
        source="WHO Physical activity fact sheet",
        url="https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        retrieved="2026-09"),
    "general_diet": dict(
        title="日常饮食",
        text="多吃蔬菜和水果（WHO 建议每天至少 400 克），主食多选全谷物和豆类；成年人每天食盐少于 5 克，少吃添加糖和含反式脂肪的加工食品。有慢性病饮食要求的，请按医生或营养师的安排执行。",
        source="WHO Healthy diet fact sheet (2026-01-26)",
        url="https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        retrieved="2026-09"),
}

CAPABILITY = (
    "本系统根据同一次检查的静态甲襞图像，输出 15 项报告条目的分类预测，另有 1 项两枝档位组合、"
    "4 项开发集固定基线，共 20 行。它不测量尺寸、条数、次数或流速，不给出分数，不做微循环分级，"
    "不能替代医生的判断，也不给个人治疗方案。模型只在同一来源的 186 例检查上训练，没有外部验证。")


def main():
    cards = dict(
        capability=dict(text=CAPABILITY, source="本系统发布说明"),
        terms={f: dict(text=t, source=HANDBOOK + "；本系统标签说明",
                       note="仅解释标签含义，不含疾病关联")
               for f, t in TERMS.items()},
        education=EDUCATION)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(cards["terms"]), "term cards,", len(EDUCATION), "education cards")


if __name__ == "__main__":
    main()
