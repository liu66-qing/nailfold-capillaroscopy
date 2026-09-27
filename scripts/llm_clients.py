#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal clients for the external annotation models (Qwen, DeepSeek).

Keys are read from the project .env (git-ignored) into memory only; they are
never printed, logged or written anywhere. GPT and Opus are driven as agents
(codex CLI / Claude subagents), not through this module.

Every call that carries an image goes through `assert_not_locked`, so a
locked-47 case can never leave the machine.

Self-test (no images sent, prints status only):
  PYTHONIOENCODING=utf-8 python scripts/llm_clients.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"

ENDPOINTS = {
    "qwen": dict(env="qwen",
                 url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
                 model="qwen3.8-max"),
    "deepseek": dict(env="deepseek_seek",
                     url="https://api.deepseek.com/chat/completions",
                     model="deepseek-chat"),
}


def load_env() -> dict:
    out = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_LOCKED = None


def assert_not_locked(case_id: str) -> None:
    global _LOCKED
    if _LOCKED is None:
        import pandas as pd
        lab = pd.read_csv(ROOT / "server_code_audit" / "locked_evaluation_v1_reviewed.csv")
        _LOCKED = set(lab[lab.development_fold.isna()].exam_case_id.astype(str))
    if case_id in _LOCKED:
        raise RuntimeError("refusing to send a locked-47 case: %s" % case_id)


def image_part(path, max_side: int = 1024, quality: int = 90) -> dict:
    """Encode an image as an OpenAI-style image_url part, downscaled to max_side."""
    import base64
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def chat(provider: str, messages: list, model: str | None = None,
         timeout: int = 180, retries: int = 3, **extra) -> dict:
    import http.client
    import time
    cfg = ENDPOINTS[provider]
    key = load_env()[cfg["env"]]
    body = dict(model=model or cfg["model"], messages=messages, **extra)
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            cfg["url"], data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (http.client.IncompleteRead, ConnectionError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _selftest() -> None:
    env = load_env()
    for name, cfg in ENDPOINTS.items():
        k = env.get(cfg["env"], "")
        status = dict(present=bool(k), length=len(k),
                      looks_obfuscated=k.startswith("o1_"),
                      looks_sk=k.startswith("sk-"))
        try:
            r = chat(name, [{"role": "user", "content": "reply with the word ok"}],
                     max_tokens=5)
            status["call"] = "ok"
            status["reply"] = r["choices"][0]["message"]["content"][:20]
        except urllib.error.HTTPError as e:
            status["call"] = "http %d" % e.code
            status["body"] = e.read().decode("utf-8", "replace")[:200]
        except Exception as e:  # noqa: BLE001
            status["call"] = type(e).__name__ + ": " + str(e)[:120]
        print(name, json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(_selftest())
