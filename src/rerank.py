"""重排（Rerank）：对召回结果用交叉编码器精排，进一步提升精确度。

实现为可插拔：配置硅基流动（SiliconFlow）免费 API 即可启用；
未配置时跳过重排层（退化为混合检索）。

使用（.env）：
    RERANK_BACKEND=siliconflow   # 留空/缺失 = 跳过重排
    RERANK_MODEL=BAAI/bge-reranker-v2-m3
    SILICONFLOW_API_KEY=sk-xxx
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Tuple


def rerank_available() -> bool:
    return bool(os.getenv("RERANK_BACKEND")) and bool(os.getenv("SILICONFLOW_API_KEY"))


def rerank(query: str, candidates: List[Tuple[int, str]], top_k: int) -> List[int]:
    """重排候选 (doc_id, text) 列表，返回按相关度排序的 doc_id 前 top_k。"""
    if not candidates:
        return []
    if not rerank_available():
        # 无 rerank：保持原序（退化为混合检索结果）
        return [i for i, _ in candidates[:top_k]]

    model = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    body = json.dumps({
        "model": model,
        "query": query,
        "documents": [t for _, t in candidates],
        "top_n": min(top_k, len(candidates)),
        "return_documents": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.siliconflow.cn/v1/rerank",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[warn] rerank 调用失败，退回原始排序: {e}")
        return [i for i, _ in candidates[:top_k]]

    ordered = [item["index"] for item in data.get("results", [])]
    ids = [candidates[i][0] for i in ordered if i < len(candidates)]
    # 补上未返回的候选
    rest = [candidates[i][0] for i in range(len(candidates)) if i not in ordered]
    return (ids + rest)[:top_k]
