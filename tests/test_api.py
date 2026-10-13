"""增量索引与 FastAPI 接口测试（离线）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.embedding import HashEmbedder
from src.vector_store import VectorStore
from src.documents import SUPPORTED_EXT


def test_incremental_index(tmp_path):
    docs_dir = tmp_path / "docs"
    vec_dir = tmp_path / "vec"
    docs_dir.mkdir()

    (docs_dir / "a.md").write_text("# 文档A\n\n内容：快速排序 O(n log n)。", encoding="utf-8")
    (docs_dir / "b.md").write_text("# 文档B\n\n内容：二分查找 O(log n)。", encoding="utf-8")

    store = VectorStore(HashEmbedder(), vec_dir, docs_dir)

    s1 = store.index()
    assert s1["mode"] == "first"
    assert s1["embedded_chunks"] > 0

    # 无变更：不重嵌入
    s2 = store.index()
    assert s2["embedded_chunks"] == 0

    # 变更 1 个文件：只重嵌入该文件
    time.sleep(1.1)
    (docs_dir / "a.md").write_text(
        "# 文档A\n\n内容：快速排序 O(n log n)。\n\n新增：堆排序也是 O(n log n)。",
        encoding="utf-8")
    s3 = store.index()
    assert s3["mode"] == "incremental"
    assert s3["reindexed_files"] == 1
    assert s3["embedded_chunks"] < s1["embedded_chunks"]


def test_retrieve(tmp_path):
    docs_dir = tmp_path / "docs"
    vec_dir = tmp_path / "vec"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 笔记\n\n二分查找的时间复杂度是 O(log n)。", encoding="utf-8")

    store = VectorStore(HashEmbedder(), vec_dir, docs_dir)
    store.index()
    docs = store.retrieve("二分查找复杂度", k=1)
    assert docs and "二分查找" in docs[0].page_content


def test_fastapi_endpoints_offline(tmp_path, monkeypatch):
    """FastAPI 接口离线测试：/health /docs /index 可用，/ask 无 key 降级 503。"""
    from fastapi.testclient import TestClient

    docs_dir = tmp_path / "docs"
    vec_dir = tmp_path / "vec"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 笔记\n\n内容测试。", encoding="utf-8")

    import src.app as appmod
    # 用测试目录替换模块级 store
    appmod.cfg = appmod.load_config()
    appmod.store = VectorStore(HashEmbedder(), vec_dir, docs_dir)
    appmod.set_docs_dir(docs_dir)
    appmod.agent = None  # 惰性重建

    client = TestClient(appmod.app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    r = client.get("/docs")
    assert r.status_code == 200

    r = client.post("/index", json={"force": False})
    assert r.status_code == 200 and r.json()["embedded_chunks"] > 0

    # 无 key：503 降级
    appmod.cfg.api_key = ""
    r = client.post("/ask", json={"question": "测试"})
    assert r.status_code == 503
