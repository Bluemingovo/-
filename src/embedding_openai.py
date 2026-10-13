"""OpenAI 兼容 embedding（可选，替换哈希嵌入用）。

示例：接入阿里云 DashScope text-embedding-v3（OpenAI 兼容），或任意
OpenAI-compatible embedding 服务。需在 .env 中设置 EMBEDDING_BACKEND=openai
以及对应的 EMBEDDING_MODEL、EMBEDDING_BASE_URL、EMBEDDING_API_KEY。
"""
from __future__ import annotations

import os
from typing import List, Optional


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL", "")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url or None)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        client = self._client()
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
