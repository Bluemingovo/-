"""记忆引导：任何 agent 每次启动时调用，从 GitHub 拉取最新共享记忆。

用法（任何 agent / 任何脚本）：
    python -m scripts.fetch_memory

行为：
1. 从 GitHub raw URL 拉取 docs/llm-context.md 与 docs/assistant-memory.md
2. 合并写入本地缓存 data/memory/shared_memory.md
3. 打印记忆内容（可直接注入 system prompt）

优点：记忆的权威副本在 GitHub（git 版本化），agent 不依赖任何本地服务，
只要能联网就能拿到最新记忆。网络失败时自动回退到本地缓存。
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_BASE = "https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs"
FILES = ["llm-context.md", "assistant-memory.md"]
CACHE = BASE_DIR / "data" / "memory" / "shared_memory.md"


def fetch(url: str, timeout: int = 20) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "memory-agent"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        print(f"[warn] 拉取失败 {url}: {e}", file=sys.stderr)
        return None


def main() -> int:
    parts = []
    for f in FILES:
        content = fetch(f"{RAW_BASE}/{f}")
        if content:
            parts.append(content)
    if parts:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text("\n\n".join(parts), encoding="utf-8")
        print(f"已拉取最新共享记忆 → {CACHE}")
        print("\n" + "=" * 50)
        print("\n\n".join(parts))
        return 0
    # 回退本地缓存
    if CACHE.exists():
        print(f"[warn] 网络不可用，回退本地缓存 {CACHE}")
        print(CACHE.read_text(encoding="utf-8"))
        return 0
    print("[error] 无网络且无本地缓存，无法加载记忆", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
