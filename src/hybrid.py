"""混合检索：BM25（关键词）+ 向量（语义）→ RRF 倒数排名融合。

为什么需要混合检索：
- 纯向量检索对专有名词/精确关键词不敏感（"函数调用"可能召回到"调用示例"而非实现）
- BM25 关键词检索精确但无语义泛化
- 两者互补，RRF（Reciprocal Rank Fusion）合并排名，是 2026 年 RAG 的标准做法

BM25 公式（自实现，面试可讲）：
    score(d, q) = Σ IDF(qi) * [ f(qi,d) * (k1+1) / (f(qi,d) + k1*(1-b+b*|d|/avgdl)) ]
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """中文按字 + 英文/数字按词切分（BM25 用）。"""
    tokens = []
    for part in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower()):
        if re.fullmatch(r"[a-zA-Z0-9_]+", part):
            tokens.append(part)
        else:
            tokens.extend(part)  # 中文逐字
    return tokens


class BM25Index:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.doc_tokens = [tokenize(d) for d in corpus]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
        self.n = max(1, len(corpus))

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        t = self.doc_tokens[doc_idx]
        dl = self.doc_len[doc_idx]
        # 词频
        tf = {}
        for term in t:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        for qt in set(q_tokens):
            f = tf.get(qt, 0)
            if f == 0:
                continue
            denom = f + K1 * (1 - B + B * dl / max(1.0, self.avgdl))
            s += self._idf(qt) * (f * (K1 + 1)) / denom
        return s

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def rrf_fusion(*rankings: List[int], k: int = 60) -> List[int]:
    """RRF 融合多个排名列表，返回文档索引按融合分排序。

    RRF 分数 = Σ 1/(k + rank_i)，rank 从 1 开始。
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def hybrid_search(bm25: BM25Index, vector_doc_ids: List[int],
                  query: str, top_k: int = 10) -> List[int]:
    """混合检索：BM25 与向量各自召回 top_k，RRF 融合取前 top_k。"""
    bm25_rank = [i for i, _ in bm25.search(query, top_k)]
    return rrf_fusion(bm25_rank, vector_doc_ids)[:top_k]
