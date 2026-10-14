"""RAG 质量评测（LLM-as-judge，指标口径对齐 RAGAS）。

三个指标（RAGAS 标准定义）：
1. faithfulness（忠实度）：答案中的陈述是否全部有上下文依据（防幻觉）
2. answer_relevancy（答案相关性）：答案是否直接、完整地回答了问题
3. context_precision（上下文精度）：检索到的上下文中有多少真正帮助回答

用法：
    python -m scripts.eval_ragas
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.embedding import build_embedder
from src.vector_store import VectorStore
from src.llm import build_llm


def judge(llm, prompt: str) -> float:
    """让 LLM 输出 0~1 分数。"""
    resp = llm.invoke(prompt)
    m = re.search(r"(0?\.\d+|1\.0|[01])\s*$|分数[:：]\s*(0?\.\d+|1\.0|[01])", resp.content)
    if m:
        v = float(m.group(1) or m.group(2))
        return max(0.0, min(1.0, v))
    # 兜底：搜所有小数
    nums = re.findall(r"0?\.\d+|1\.0|[01](?=\D|$)", resp.content)
    if nums:
        return max(0.0, min(1.0, float(nums[0])))
    return 0.0


def faithfulness(llm, question: str, context: str, answer: str) -> float:
    prompt = (f"你是一名严格的评测裁判。判断下面的【答案】中的陈述是否全部有【上下文】依据，"
              f"没有编造。输出一个 0 到 1 的分数（1=完全忠于上下文，0=完全编造），只输出数字。\n\n"
              f"【问题】{question}\n【上下文】\n{context[:2000]}\n【答案】\n{answer[:1000]}")
    return judge(llm, prompt)


def answer_relevancy(llm, question: str, answer: str) -> float:
    prompt = (f"你是一名严格的评测裁判。判断下面的【答案】是否直接、完整地回答了【问题】。"
              f"输出一个 0 到 1 的分数（1=完全切题完整，0=答非所问），只输出数字。\n\n"
              f"【问题】{question}\n【答案】\n{answer[:1000]}")
    return judge(llm, prompt)


def context_precision(llm, question: str, chunks: list) -> float:
    listed = "\n\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(chunks))
    prompt = (f"你是一名严格的评测裁判。下面是为回答【问题】检索到的 {len(chunks)} 条上下文。"
              f"判断其中多少条真正有助于回答该问题。输出一个 0 到 1 的比例分数"
              f"（1=全部有用，0=全部无用），只输出数字。\n\n"
              f"【问题】{question}\n【检索上下文】\n{listed}")
    return judge(llm, prompt)


def main():
    cfg = load_config()
    llm = build_llm(cfg)
    store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)
    store.index()

    qfile = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
    questions = json.loads(qfile.read_text(encoding="utf-8"))[:5]  # 评测 5 题控制费用

    from src.agent import RAGAgent
    agent = RAGAgent(cfg, store)

    f_scores, r_scores, p_scores = [], [], []
    for q in questions:
        question = q["question"]
        docs = store.retrieve(question, k=6, mode="hybrid")
        context = "\n\n".join(d.page_content for d in docs)
        a = agent.answer(question, k=6, mode="hybrid")
        answer = a.text
        f = faithfulness(llm, question, context, answer)
        r = answer_relevancy(llm, question, answer)
        p = context_precision(llm, question, [d.page_content for d in docs])
        f_scores.append(f); r_scores.append(r); p_scores.append(p)
        print(f"  {question[:30]}...  faithfulness={f:.2f} relevancy={r:.2f} precision={p:.2f}")

    def avg(x):
        return sum(x) / len(x) if x else 0

    print("\n=== RAGAS 口径评测结果（5 题，deepseek-chat 裁判） ===")
    print(f"  faithfulness（忠实度）      : {avg(f_scores):.2f}")
    print(f"  answer_relevancy（答案相关性）: {avg(r_scores):.2f}")
    print(f"  context_precision（上下文精度）: {avg(p_scores):.2f}")


if __name__ == "__main__":
    main()
