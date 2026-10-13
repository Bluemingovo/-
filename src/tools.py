"""本地工具（函数调用 / Function Calling）。

这些工具不依赖外部 API，用于演示「工具调用」：当用户的问题不是知识检索、
而是需要查询文档清单 / 搜索文件名时，模型通过函数调用来完成。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.tools import tool

from .documents import SUPPORTED_EXT

_docs_dir = Path(__file__).resolve().parent.parent / "data" / "docs"
_memory_store = None  # 由 app 注入（set_memory_store）


def set_docs_dir(path: Path) -> None:
    global _docs_dir
    _docs_dir = path


def set_memory_store(store) -> None:
    """把 MemoryStore 注入工具层，供 save_memory 工具写入长期记忆。"""
    global _memory_store
    _memory_store = store


@tool
def list_docs() -> str:
    """列出知识库中当前已索引的全部文档（文件名、大小、修改时间）。"""
    if not _docs_dir.exists():
        return "知识库目录不存在。"
    rows = []
    for p in sorted(_docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            size = p.stat().st_size
            rows.append(f"- {p.name}（{size} 字节）")
    return "知识库文档清单：\n" + ("\n".join(rows) if rows else "（空）")


@tool
def search_files(keyword: str) -> str:
    """按关键词搜索知识库中的文件名，返回匹配的文件列表。"""
    if not _docs_dir.exists():
        return "知识库目录不存在。"
    hits = [p.name for p in _docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
            and keyword.lower() in p.name.lower()]
    return "匹配的文件：" + (", ".join(hits) if hits else "无")


@tool
def save_memory(key: str, value: str) -> str:
    """把值得长期记住的事实写入长期记忆（用户偏好、决定、重要信息）。
    当用户透露了值得长期记住的信息时，主动调用本工具。"""
    if _memory_store is None:
        return "记忆库未初始化。"
    _memory_store.add_memory(key, value, source="agent")
    return f"已记住：{key}：{value}"


ALL_TOOLS = [list_docs, search_files, save_memory]
