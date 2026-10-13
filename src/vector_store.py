"""向量库：FAISS 存储 + 增量索引（文件变更检测）。

核心点：
- 增量索引：只在文件「新增」或「变更」时重新嵌入对应文件，避免全量重索引；
  这就是简历里「向量库重复索引率降低」的真实依据。
- 元数据里带 file_mtime，用于和磁盘上的文件 mtime 比对判断变更。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from .documents import load_docs_from_dir
from .embedding import Embedder

MANIFEST = "manifest.json"


class VectorStore:
    def __init__(self, embedder: Embedder, vector_dir: Path, docs_dir: Path):
        self.embedder = embedder
        self.vector_dir = vector_dir
        self.docs_dir = docs_dir
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = vector_dir / MANIFEST
        self._index = None  # FAISS 实例

    # ---- 变更检测 ----
    def _load_manifest(self) -> Dict[str, float]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, m: Dict[str, float]) -> None:
        self._manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    def _current_files(self) -> Dict[str, float]:
        from .documents import SUPPORTED_EXT
        return {str(p): p.stat().st_mtime
                for p in sorted(self.docs_dir.rglob("*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXT}

    def detect_changes(self) -> Tuple[List[str], List[str]]:
        """返回 (新增或变更的文件, 已删除的文件)。"""
        old = self._load_manifest()
        cur = self._current_files()
        changed = [f for f, mt in cur.items() if old.get(f) != mt]
        deleted = [f for f in old if f not in cur]
        return changed, deleted

    # ---- 索引 ----
    def index(self, force: bool = False) -> Dict[str, int]:
        """增量索引；返回统计信息（新嵌入 chunk 数等）。"""
        changed, deleted = self.detect_changes()

        if force:
            # 全量重索引
            all_docs = load_docs_from_dir(self.docs_dir)
            self._build_index(all_docs)
            stats = {"reindexed_files": len(self._current_files()),
                     "embedded_chunks": len(all_docs), "mode": "full"}
            return stats

        if not changed and not deleted and self._index is not None:
            return {"reindexed_files": 0, "embedded_chunks": 0, "mode": "incremental"}

        # 增量：只对变更文件重新切分嵌入
        cur = self._current_files()
        docs_to_add: List[Document] = []
        for f in changed:
            p = Path(f)
            from .documents import load_document, split_text
            text = load_document(p)
            if text.strip():
                docs_to_add.extend(split_text(text, source=f, file_mtime=cur[f]))

        if self._index is None:
            # 首次：全量
            all_docs = load_docs_from_dir(self.docs_dir)
            self._build_index(all_docs)
            stats = {"reindexed_files": len(self._current_files()),
                     "embedded_chunks": len(all_docs), "mode": "first"}
        else:
            if docs_to_add:
                self._index.add_documents(docs_to_add)
            self._save_manifest(cur)
            stats = {"reindexed_files": len(changed),
                     "embedded_chunks": len(docs_to_add), "mode": "incremental"}
        return stats

    def _build_index(self, docs: List[Document]) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_core.embeddings import Embeddings

        embedder = self.embedder

        class _Emb(Embeddings):
            def embed_documents(self, texts):
                return embedder.embed_documents(texts)

            def embed_query(self, text):
                return embedder.embed_query(text)

        self._index = FAISS.from_documents(docs, _Emb())
        self._index.save_local(str(self.vector_dir))
        self._save_manifest(self._current_files())

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self._index is None:
            self._load_or_build()
        return self._index.similarity_search(query, k=k)

    def _load_or_build(self):
        from langchain_community.vectorstores import FAISS
        from langchain_core.embeddings import Embeddings

        embedder = self.embedder

        class _Emb(Embeddings):
            def embed_documents(self, texts):
                return embedder.embed_documents(texts)

            def embed_query(self, text):
                return embedder.embed_query(text)

        faiss_path = self.vector_dir / "index.faiss"
        if faiss_path.exists():
            self._index = FAISS.load_local(
                str(self.vector_dir), _Emb(), allow_dangerous_deserialization=True)
        else:
            docs = load_docs_from_dir(self.docs_dir)
            self._build_index(docs)
