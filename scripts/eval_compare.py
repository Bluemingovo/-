"""三模式检索对比评测：vector / hybrid / hybrid_rerank。

用法：
    python -m scripts.eval_compare --k 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.embedding import build_embedder
from src.vector_store import VectorStore


def hit_at_k(docs, source: str, k: int) -> bool:
    got = {d.metadata.get("source", "").replace("\\", "/").split("/")[-1] for d in docs[:k]}
    return source in got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)
    store.index()

    qfile = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
    questions = json.loads(qfile.read_text(encoding="utf-8"))
    print(f"评测集：{len(questions)} 题，Hit@{args.k}\n")

    results = {"vector": 0, "hybrid": 0, "hybrid_rerank": 0}
    for q in questions:
        row = []
        for mode in results:
            docs = store.retrieve(q["question"], k=args.k, mode=mode)
            ok = hit_at_k(docs, q["source"], args.k)
            results[mode] += int(ok)
            row.append("✓" if ok else "✗")
        print(f"  {' | '.join(row)}  {q['question'][:40]}")

    n = len(questions)
    print("\n=== 对比结果 ===")
    for mode, hits in results.items():
        print(f"  {mode:16s} Hit@{args.k} = {hits}/{n} = {hits/n:.0%}")


if __name__ == "__main__":
    main()
