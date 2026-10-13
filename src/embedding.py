"""Embedding 抽象。

默认使用「哈希嵌入」：完全不依赖外部 API，可离线复现、便于单元测试。
生产环境可替换为真实 embedding 模型（见 README「接入真实 Embedding」）。
"""
from __future__ import annotations

import hashlib
from typing import List

DIM = 256  # 哈希嵌入维度


class Embedder:
    """统一接口：把文本变成向量。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """基于字符 n-gram 哈希的确定性嵌入。

    特点：可离线复现、零依赖；适合在没有 embedding API 时验证整条链路。
    局限：语义区分度弱于真实 embedding 模型，正式评测请换真实模型。
    """

    def __init__(self, dim: int = DIM, n: int = 3):
        self.dim = dim
        self.n = n

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = (text or "").strip().lower()
        if not text:
            return vec
        # 字符 n-gram（对中文按字、对英文按字符即可）
        for i in range(max(1, len(text) - self.n + 1)):
            gram = text[i : i + self.n]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def build_embedder(backend: str = "hash", model: str = "", **kwargs) -> Embedder:
    if backend == "hash":
        return HashEmbedder(**kwargs)
    if backend in ("openai", "deepseek"):
        # 真实 embedding 走 OpenAI 兼容接口（按需开启）
        from .embedding_openai import OpenAIEmbedder
        return OpenAIEmbedder(model=model, **kwargs)
    raise ValueError(f"未知 embedding backend: {backend}")
