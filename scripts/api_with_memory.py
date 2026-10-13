"""直接调用 LLM API 时自动带记忆（模板，通用）。

原理：LLM API 是无状态的。每次调用必须把记忆放进 messages 传过去。
三步：① 读取记忆 → ② 注入 system prompt → ③ 调 API。

用法：
    python -m scripts.api_with_memory "我最近在复习什么？"

记忆来源（优先级）：
1. 本地缓存 data/memory/shared_memory.md（先运行 python -m scripts.fetch_memory 生成）
2. 直接尝试 GitHub raw 拉取
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE = BASE_DIR / "data" / "memory" / "shared_memory.md"
RAW_URLS = [
    "https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs/llm-context.md",
    "https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs/assistant-memory.md",
]

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")


# ① 读取记忆
def load_memory() -> str:
    # 先本地缓存
    if CACHE.exists():
        return CACHE.read_text(encoding="utf-8")
    # 再试 GitHub raw
    parts = []
    for u in RAW_URLS:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "memory-agent"})
            with urllib.request.urlopen(req, timeout=15) as r:
                parts.append(r.read().decode("utf-8"))
        except Exception:
            pass
    if parts:
        mem = "\n\n".join(parts)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(mem, encoding="utf-8")
        return mem
    return ""


# ③ 调 API（OpenAI 兼容格式，DeepSeek/OpenAI/通义/任何兼容平台通用）
def call_llm(memory: str, question: str, history: list | None = None) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        return "[错误] 未配置 DEEPSEEK_API_KEY"

    # ② 注入记忆到 system prompt
    messages = [{
        "role": "system",
        "content": (f"你是用户的个人 AI 助手。以下是用户的背景记忆，"
                    f"回答时必须基于记忆中的事实：\n\n{memory}")
    }]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    body = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[错误] API 调用失败：{e}"


def main():
    question = " ".join(sys.argv[1:]) or "我最近在复习什么？"
    memory = load_memory()
    if not memory:
        print("[错误] 没有记忆可加载。先运行 python -m scripts.fetch_memory")
        return 1
    print(f"已加载记忆 {len(memory)} 字符，调用 LLM ...\n")
    answer = call_llm(memory, question)
    print(f"问题：{question}\n\n回答：\n{answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
