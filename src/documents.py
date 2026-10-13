"""文档加载与切分。

支持 .txt / .md / .pdf。切分使用中文友好的分隔符，
每个 chunk 附带元数据（source、chunk_index、file_mtime），
供「文件变更检测」判断是否需要重新索引。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXT = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)
    raise ValueError(f"不支持的文件类型: {ext}")


def split_text(text: str, source: str, file_mtime: float,
               chunk_size: int = 400, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, c in enumerate(chunks):
        docs.append(Document(
            page_content=c,
            metadata={
                "source": source,
                "chunk_index": i,
                "file_mtime": file_mtime,
            },
        ))
    return docs


def load_docs_from_dir(docs_dir: Path) -> List[Document]:
    """加载目录下所有支持的文档并切分。"""
    docs: List[Document] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            text = load_document(p)
            if text.strip():
                docs.extend(split_text(text, source=str(p), file_mtime=p.stat().st_mtime))
    return docs
