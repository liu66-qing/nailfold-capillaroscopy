#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the annotator exam for API models (qwen, deepseek) and score any model.

  run   : python scripts/run_ai_exam.py run qwen
          python scripts/run_ai_exam.py run deepseek
  score : python scripts/run_ai_exam.py score

Answers are appended to answers/<model>.jsonl (resumable). The answer key is
read only by `score`. GPT (codex) and Opus (subagent) write the same jsonl
format into the same folder, and are scored by the same function.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
EXAM = ROOT / "artifacts/experiments/video_kymo_20260925/ai_exam"
KEY = EXAM.parent / "ai_exam_key" / "answer_key_DO_NOT_SHOW.json"
ANS = EXAM.parent / "ai_exam_answers"

PROMPT = """这是一张甲襞毛细血管镜（nailfold capillaroscopy）显微图像的局部裁剪。请只根据图像回答三个问题：

1. vessel：画面中心是否有一条毛细血管袢？(yes/no)
2. crossing：该血管袢是否为交叉型（输入枝与输出枝相互交叉或扭绕成交叉形）？(yes/no)
3. abnormal：该血管袢是否形态异常（迂曲扭曲、多分支、灌木丛状/分叉状），而不是正常的发夹形（hairpin）？交叉型本身不算在这一题内，若是交叉型请回答 no。(yes/no)

只输出一行 JSON，不要其它文字，例如：{"vessel":"yes","crossing":"no","abnormal":"yes"}"""

MODELS = {"qwen": ("qwen", "qwen3.8-max"), "deepseek": ("deepseek", "deepseek-chat")}


def parse(text: str) -> dict | None:
    m = re.search(r"\{[^{}]*\}", text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("vessel", "crossing", "abnormal"):
        v = str(d.get(k, "")).strip().lower()
        out[k] = True if v in ("yes", "true", "是") else False if v in ("no", "false", "否") else None
    return out


def run(name: str) -> None:
    import llm_clients as L
    provider, model = MODELS[name]
    ANS.mkdir(parents=True, exist_ok=True)
    out = ANS / (name + ".jsonl")
    done = set()
    if out.exists():
        # keep only real replies; transport errors are retried on resume
        kept = [l for l in out.read_text(encoding="utf-8").splitlines()
                if l.strip() and not json.loads(l)["raw"].startswith("ERROR")]
        out.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        done = {json.loads(l)["qid"] for l in kept}
    todo = [p for p in sorted((EXAM / "crops").glob("*.jpg")) if p.stem not in done]

    def one(p: Path):
        msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, L.image_part(p, 320)]}]
        try:
            r = L.chat(provider, msg, model=model, max_tokens=200, temperature=0)
            raw = r["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            raw = "ERROR " + type(e).__name__
        return dict(qid=p.stem, model=name, raw=raw[:400], answer=parse(raw))

    with ThreadPoolExecutor(2 if name == "qwen" else 4) as ex, out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ex.map(one, todo)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(name, i, row["answer"], flush=True)


def kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def score() -> None:
    key = {k["qid"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))}
    report = {}
    for f in sorted(ANS.glob("*.jsonl")):
        rows = {}
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["qid"]] = r.get("answer") or {}
        res = dict(answered=len(rows), parse_fail=sum(1 for v in rows.values() if not v))
        for q, truth_field, subset in [
            ("vessel", "is_vessel", lambda k: True),
            ("crossing", "is_crossing", lambda k: k["is_vessel"]),
            ("abnormal", "is_abnormal", lambda k: k["is_abnormal"] is not None),
        ]:
            ids = [i for i, k in key.items() if subset(k) and i in rows]
            t = [key[i][truth_field] for i in ids]
            p = [rows[i].get(q) for i in ids]
            ok = [(x, y) for x, y in zip(t, p) if y is not None]
            res[q] = dict(n=len(ok), acc=round(sum(x == y for x, y in ok) / len(ok), 4) if ok else None,
                          kappa=kappa(t, p),
                          pred_yes_rate=round(sum(y for _, y in ok) / len(ok), 3) if ok else None,
                          true_yes_rate=round(sum(x for x, _ in ok) / len(ok), 3) if ok else None)
        res["passes_kappa_0_4"] = {q: (res[q]["kappa"] or -1) >= 0.4 for q in ("crossing", "abnormal")}
        report[f.stem] = res
    (EXAM.parent / "ai_exam_scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "score": score}[sys.argv[1]]()
