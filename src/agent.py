"""RAG 问答主流程：检索 + 工具调用。

流程：
1. 先用向量库检索 top-k 相关 chunk（RAG）；
2. 把检索结果 + 本地工具（函数调用）一起交给 LLM；
3. LLM 判断：问题能从检索内容回答 → 直接回答；否则调用工具查询文档清单/搜索文件；
4. 返回答案 + 引用来源 + 是否触发工具调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document

from .config import Config
from .llm import build_llm
from .vector_store import VectorStore
from .tools import ALL_TOOLS

SYSTEM_PROMPT = """你是个人知识库问答助手。规则：
1. 优先依据【检索到的资料】回答；回答要具体、给出依据。
2. 当用户询问"有哪些文档/文件清单"或"找某个文件"时，【必须】调用工具 list_docs 或 search_files 获取实时清单，不得仅凭检索内容回答。
3. 如果检索资料不足以回答，明确说"资料中未找到"，不要编造。
4. 回答末尾用【来源：文件名】标注依据。"""


@dataclass
class Answer:
    text: str = ""
    sources: List[str] = field(default_factory=list)
    used_tool: bool = False


class RAGAgent:
    def __init__(self, cfg: Config, store: VectorStore):
        self.cfg = cfg
        self.store = store
        self.llm = build_llm(cfg)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

    def answer(self, question: str, k: int = 4, structured: bool = False) -> Answer:
        # 1. 检索
        docs = self.store.retrieve(question, k=k)
        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        sources = sorted({d.metadata.get("source", "?").split("\\")[-1].split("/")[-1] for d in docs})

        if structured:
            # 结构化输出：JSON 模式（response_format=json_object）
            llm = self.llm.bind(response_format={"type": "json_object"})
            prompt = (f"【检索到的资料】\n{context}\n\n【用户问题】\n{question}\n\n"
                      "只输出一个 JSON 对象，不要输出其他内容，格式："
                      '{"answer": "回答内容", "sources": ["文件名"], "confidence": 0.0~1.0}')
            resp = llm.invoke(prompt)
            import json
            try:
                data = json.loads(resp.content)
                return Answer(text=data.get("answer", ""),
                              sources=data.get("sources", sources),
                              used_tool=False)
            except Exception:
                return Answer(text=resp.content, sources=sources, used_tool=False)

        # 2. 首轮：带检索上下文 + 工具
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"【检索到的资料】\n{context}\n\n【用户问题】\n{question}"),
        ]
        resp = self.llm_with_tools.invoke(messages)

        # 3. 处理工具调用
        used_tool = False
        if getattr(resp, "tool_calls", None):
            used_tool = True
            for tc in resp.tool_calls:
                tool_map = {t.name: t for t in ALL_TOOLS}
                t = tool_map.get(tc["name"])
                if t is None:
                    continue
                result = t.invoke(tc["args"])
                messages.append(resp)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            final = self.llm.invoke(messages)
            return Answer(text=final.content, sources=sources, used_tool=used_tool)

        return Answer(text=resp.content, sources=sources, used_tool=False)
