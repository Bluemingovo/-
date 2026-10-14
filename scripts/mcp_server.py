"""记忆中枢 MCP Server — 让任何支持 MCP 的 Agent 自动共享记忆。

工具：
1. read_memory   — 读取共享记忆（llm-context + assistant-memory）
2. save_memory   — 写入一条长期记忆（共享记忆文件 + 运行时记忆库）
3. list_memories — 列出运行时长期记忆库
4. ask_rag       — 用项目的 RAG 知识库回答

接入（Claude Desktop / Cursor / Cline / Coze 等，配置一次）：
    "memory-hub": {
        "command": "python",
        "args": ["<项目路径>/scripts/mcp_server.py"]
    }

本地自检：
    python -m scripts.mcp_server --test
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from mcp.server.mcpserver import MCPServer

MEMORY_FILE = BASE_DIR / "docs" / "assistant-memory.md"
LLM_CONTEXT = BASE_DIR / "docs" / "llm-context.md"

server = MCPServer("memory-hub", version="1.0.0")


# ---------- 工具实现 ----------
def _read_shared_memory() -> str:
    parts = []
    for p in (LLM_CONTEXT, MEMORY_FILE):
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def _save_shared_memory(key: str, value: str) -> str:
    today = datetime.date.today().isoformat()
    line = f"- {key}：{value}（{today}）"
    text = MEMORY_FILE.read_text(encoding="utf-8") if MEMORY_FILE.exists() else ""
    text = text.rstrip("\n") + "\n" + line + "\n"
    MEMORY_FILE.write_text(text, encoding="utf-8")
    # 同时写运行时记忆库
    try:
        from src.memory import MemoryStore
        from src.config import load_config
        cfg = load_config()
        mem = MemoryStore(cfg.vector_dir.parent / "memory")
        mem.add_memory(key, value, source="mcp")
    except Exception:
        pass
    return f"已记住：{key}：{value}"


def _list_runtime_memories() -> str:
    try:
        from src.memory import MemoryStore
        from src.config import load_config
        cfg = load_config()
        mem = MemoryStore(cfg.vector_dir.parent / "memory")
        items = mem.list_all()
        return json.dumps([{"key": i["key"], "value": i["value"]} for i in items],
                          ensure_ascii=False) if items else "（空）"
    except Exception as e:
        return f"读取失败：{e}"


def _ask_rag(question: str) -> str:
    try:
        from src.config import load_config
        from src.embedding import build_embedder
        from src.vector_store import VectorStore
        from src.agent import RAGAgent
        cfg = load_config()
        store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)
        agent = RAGAgent(cfg, store)
        a = agent.answer(question, k=6)
        return a.text
    except Exception as e:
        return f"RAG 问答失败：{e}"


# ---------- 注册工具 ----------
@server.tool()
def read_memory() -> str:
    """读取用户的共享记忆（项目背景、用户画像、偏好）。对话开始时先调用本工具了解用户。"""
    return _read_shared_memory()


@server.tool()
def save_memory(key: str, value: str) -> str:
    """把值得长期记住的事实写入用户共享记忆。

    Args:
        key: 简短中文短语，如 求职状态
        value: 记忆内容
    """
    return _save_shared_memory(key, value)


@server.tool()
def list_memories() -> str:
    """列出运行时长期记忆库的全部条目。"""
    return _list_runtime_memories()


@server.tool()
def ask_rag(question: str) -> str:
    """用用户的知识库（RAG）回答专业问题，例如项目代码、学习笔记相关内容。

    Args:
        question: 要查询的问题
    """
    return _ask_rag(question)


def main():
    """命令行入口（pip 安装后可用 `llm-rag-mcp` 命令直接调用）。"""
    if "--test" in sys.argv:
        print("read_memory 长度:", len(_read_shared_memory()))
        print(_save_shared_memory("MCP测试", "MCP Server 功能自检"))
        print("list_memories:", _list_runtime_memories())
        print("MCP Server 自检完成")
    else:
        import anyio
        anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    main()
