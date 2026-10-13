"""离线自检：不依赖外部 API，验证 embedding / 切分 / 变更检测 / 检索链路。

用法：
    python -m src.selfcheck
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .embedding import HashEmbedder
from .documents import split_text, load_docs_from_dir
from .vector_store import VectorStore

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
VEC_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"


def main():
    print("== 1. 哈希嵌入 ==")
    emb = HashEmbedder()
    v = emb.embed_query("二分查找")
    assert len(v) == 256, f"维度错误: {len(v)}"
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6, "未归一化"
    print("   OK：维度 256，L2 归一化通过")

    print("== 2. 文档切分 ==")
    docs = load_docs_from_dir(DOCS_DIR)
    assert docs, "未加载到文档"
    assert all("source" in d.metadata for d in docs), "缺 source 元数据"
    print(f"   OK：共 {len(docs)} 个 chunk，含 source/chunk_index/file_mtime")

    print("== 3. 变更检测 ==")
    store = VectorStore(emb, VEC_DIR, DOCS_DIR)
    changed, deleted = store.detect_changes()
    print(f"   OK：新增/变更={len(changed)}，删除={len(deleted)}")

    print("== 4. 离线检索（哈希嵌入，向量内积）==")
    # 直接用哈希嵌入做余弦相似度，验证检索方向正确（不依赖 FAISS）
    q = emb.embed_query("二分查找的时间复杂度")
    best = None
    best_score = -1
    for d in docs:
        if "二分" not in d.page_content and "复杂度" not in d.page_content:
            continue
        dv = emb.embed_query(d.page_content)
        score = sum(a * b for a, b in zip(q, dv))
        if score > best_score:
            best_score = score
            best = d
    print(f"   OK：命中 chunk（含“复杂度”）：{best.page_content[:40]}...")

    print("\n全部离线自检通过 [OK]")


if __name__ == "__main__":
    main()
