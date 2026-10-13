"""离线单元测试：不依赖 API key。

运行：python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import HashEmbedder
from src.documents import split_text, load_docs_from_dir


def test_hash_embedding_dim_and_norm():
    emb = HashEmbedder()
    v = emb.embed_query("二分查找的时间复杂度")
    assert len(v) == 256
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedding_deterministic():
    emb = HashEmbedder()
    assert emb.embed_query("hello") == emb.embed_query("hello")


def test_split_text_metadata():
    docs = split_text("第一段。\n\n第二段。", source="a.md", file_mtime=1.0)
    assert docs, "切分结果为空"
    for d in docs:
        assert d.metadata["source"] == "a.md"
        assert "chunk_index" in d.metadata
        assert d.metadata["file_mtime"] == 1.0


def test_load_docs_from_dir(tmp_path):
    (tmp_path / "a.md").write_text("# 标题\n\n内容一二三。", encoding="utf-8")
    (tmp_path / "b.txt").write_text("纯文本内容。", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("fake", encoding="utf-8")  # 假 pdf 应被忽略（无内容）
    docs = load_docs_from_dir(tmp_path)
    sources = {d.metadata["source"] for d in docs}
    assert str(tmp_path / "a.md") in sources
    assert str(tmp_path / "b.txt") in sources


def test_tools(tmp_path):
    from src import tools
    tools.set_docs_dir(tmp_path)
    (tmp_path / "Python笔记.md").write_text("x", encoding="utf-8")
    (tmp_path / "算法笔记.md").write_text("x", encoding="utf-8")

    lst = tools.list_docs.invoke({})
    assert "Python笔记.md" in lst and "算法笔记.md" in lst

    hit = tools.search_files.invoke({"keyword": "python"})
    assert "Python笔记.md" in hit
    miss = tools.search_files.invoke({"keyword": "不存在"})
    assert "无" in miss
