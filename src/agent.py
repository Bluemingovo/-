"""RAG 问答主流程：检索 + 工具调用 + 记忆。

流程：
0. 记忆：召回相关长期记忆 + 会话内多轮历史（跨会话记忆，可复用）
1. 先用向量库检索 top-k 相关 chunk（RAG）
2. 把检索结果 + 历史 + 本地工具（函数调用）一起交给 LLM
3. LLM 判断：问题能从检索内容回答 → 直接回答；否则调用工具查询文档清单/搜索文件
4. 本轮结束后：会话历史写入短期记忆；remember=True 时用 LLM 提炼长期记忆
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

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
    memories_saved: int = 0


class RAGAgent:
    def __init__(self, cfg: Config, store: VectorStore, memory=None):
        self.cfg = cfg
        self.store = store
        self.llm = build_llm(cfg)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.memory = memory

    def answer(self, question: str, k: int = 4, structured: bool = False,
               session_id: str = "default", remember: bool = False) -> Answer:
        # 0. 记忆：召回相关长期记忆 + 会话内历史
        memories, history = [], []
        if self.memory:
            memories = self.memory.recall(question)
            history = self.memory.history(session_id)
        sys_prompt = SYSTEM_PROMPT
        if memories:
            sys_prompt += "\n\n【长期记忆（此前对话中记住的事实）】\n" + "\n".join(memories)

        # 1. 检索
        docs = self.store.retrieve(question, k=k)
        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        sources = sorted({d.metadata.get("source", "?").split("\\")[-1].split("/")[-1] for d in docs})

        # 2. 结构化输出模式（JSON schema）
        if structured:
            llm = self.llm.bind(response_format={"type": "json_object"})
            prompt = (f"【检索到的资料】\n{context}\n\n【用户问题】\n{question}\n\n"
                      "只输出一个 JSON 对象，不要输出其他内容，格式："
                      '{"answer": "回答内容", "sources": ["文件名"], "confidence": 0.0~1.0}')
            resp = llm.invoke(prompt)
            try:
                data = json.loads(resp.content)
                text = data.get("answer", "")
                sources = data.get("sources", sources)
            except Exception:
                text = resp.content
            saved = self._post_turn(session_id, question, text, remember)
            return Answer(text=text, sources=sources, used_tool=False, memories_saved=saved)

        # 3. 组装消息：系统提示 + 历史 + 检索上下文 + 当前问题
        messages = [SystemMessage(content=sys_prompt)]
        if self.memory:
            messages.extend(history)
        messages.append(HumanMessage(content=f"【检索到的资料】\n{context}\n\n【用户问题】\n{question}"))
        resp = self.llm_with_tools.invoke(messages)

        # 4. 处理工具调用
        if getattr(resp, "tool_calls", None):
            for tc in resp.tool_calls:
                tool_map = {t.name: t for t in ALL_TOOLS}
                t = tool_map.get(tc["name"])
                if t is None:
                    continue
                result = t.invoke(tc["args"])
                messages.append(resp)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            final = self.llm.invoke(messages)
            saved = self._post_turn(session_id, question, final.content, remember)
            return Answer(text=final.content, sources=sources, used_tool=True, memories_saved=saved)

        saved = self._post_turn(session_id, question, resp.content, remember)
        return Answer(text=resp.content, sources=sources, used_tool=False, memories_saved=saved)

    def _post_turn(self, session_id: str, question: str, answer: str, remember: bool) -> int:
        """本轮结束：写短期历史；remember=True 时提炼长期记忆。"""
        if not self.memory:
            return 0
        self.memory.add_pair(session_id, question, answer)
        if not remember:
            return 0
        items = self.memory.extract_memories(self.llm, question, answer)
        for it in items:
            self.memory.add_memory(it["key"], it["value"], source="agent")
        return len(items)
