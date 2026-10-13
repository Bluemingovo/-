"""评测脚本。

指标：
1. 检索命中率（Retrieval Hit@k）：正确答案来源文档是否出现在 top-k 检索结果里。
2. 答案正确性：LLM 回答是否包含黄金答案要点（可选，需真实 LLM）。
3. 增量索引效率：对比「全量重索引」与「增量索引」嵌入的 chunk 数，证明变更检测收益。

用法：
    python -m src.eval --k 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from .config import load_config
from .embedding import build_embedder
from .vector_store import VectorStore
from .documents import load_docs_from_dir


def hit_at_k(store: VectorStore, questions: List[dict], k: int) -> float:
    hits = 0
    for q in questions:
        docs = store.retrieve(q["question"], k=k)
        got = {d.metadata.get("source", "").replace("\\", "/").split("/")[-1] for d in docs}
        want = q["source"]
        if want in got:
            hits += 1
    return hits / len(questions) if questions else 0.0


def index_efficiency(cfg) -> dict:
    """模拟：全量 vs 增量 的嵌入 chunk 数对比。"""
    all_docs = load_docs_from_dir(cfg.docs_dir)
    n_all = len(all_docs)
    # 增量：假设只有 1 个文件变更
    store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)
    changed, _ = store.detect_changes()
    n_changed = len(changed) if changed else 1
    return {
        "total_chunks": n_all,
        "changed_files": n_changed,
        "full_reindex_chunks": n_all,
        "incremental_chunks_estimate": max(1, int(n_all * n_changed / max(1, len(store._current_files())))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--with-llm", action="store_true", help="同时评估答案正确性（需 API key）")
    args = ap.parse_args()

    cfg = load_config()
    embedder = build_embedder(cfg.embedding_backend, model=cfg.embedding_model)
    store = VectorStore(embedder, cfg.vector_dir, cfg.docs_dir)
    store.index()

    qfile = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
    questions = json.loads(qfile.read_text(encoding="utf-8"))

    hk = hit_at_k(store, questions, args.k)
    print(f"检索命中率 Hit@{args.k}: {hk:.1%}")

    eff = index_efficiency(cfg)
    print(f"索引效率: 全量={eff['full_reindex_chunks']} chunks, "
          f"增量≈{eff['incremental_chunks_estimate']} chunks, "
          f"节省≈{(1 - eff['incremental_chunks_estimate']/max(1,eff['full_reindex_chunks'])):.0%}")

    if args.with_llm:
        from .agent import RAGAgent
        agent = RAGAgent(cfg, store)
        correct = 0
        for q in questions:
            a = agent.answer(q["question"], k=args.k)
            ok = any(pt in a.text for pt in q.get("gold_points", []))
            correct += int(ok)
            print(f"  Q: {q['question']}\n  A: {a.text[:80]}...  {'✓' if ok else '✗'}")
        print(f"答案正确率: {correct/len(questions):.1%}")


if __name__ == "__main__":
    main()
