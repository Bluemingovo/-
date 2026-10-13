"""把任意 GitHub 仓库索引进知识库。

用法：
    python -m scripts.ingest_github --repo https://github.com/Bluemingovo/Blueming-llm-rag-agent
    python -m scripts.ingest_github --repo https://github.com/langchain-ai/langchain --branch main --depth 1

原理：git clone 到临时目录 → 提取 .md/.txt/.py/.yml 文本 → 复用切分器 → 加入向量库。
之后 /ask 就能"参考"这个仓库的内容回答问题。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.embedding import build_embedder
from src.vector_store import VectorStore
from src.documents import split_text

FILE_EXT = {".md", ".txt", ".py", ".yml", ".yaml", ".toml", ".json", ".rst"}


def clone_repo(url: str, branch: str | None, depth: int) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ingest_"))
    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(tmp / "repo")]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    return tmp / "repo"


def repo_to_chunks(repo: Path) -> list:
    """遍历仓库文件生成 chunk（复用项目切分器，source 标记为 repo 路径）。"""
    docs = []
    for p in sorted(repo.rglob("*")):
        if not p.is_file():
            continue
        if ".git" in p.parts or p.suffix.lower() not in FILE_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            docs.extend(split_text(text, source=f"repo:{p.relative_to(repo)}",
                                   file_mtime=p.stat().st_mtime))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="GitHub 仓库 URL")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=200, help="最多索引的文件数(防止超大型仓库)")
    args = ap.parse_args()

    print(f"克隆 {args.repo} ...")
    repo = clone_repo(args.repo, args.branch, args.depth)
    try:
        docs = repo_to_chunks(repo)
        if args.max_files:
            # 按文件去重限制
            seen, kept = set(), []
            for d in docs:
                if d.metadata["source"] not in seen:
                    seen.add(d.metadata["source"])
                if len(seen) > args.max_files:
                    break
                kept.append(d)
            docs = kept
        print(f"提取 {len(docs)} 个 chunk")

        cfg = load_config()
        store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)
        total = store.add_repo_documents(docs)
        print(f"已持久化并入知识库（repo 文档共 {total} 个 chunk），重启服务后仍可检索")
    finally:
        shutil.rmtree(repo.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
