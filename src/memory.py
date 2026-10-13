"""记忆模块（通用，任何 Agent 可复用）。

两层记忆：
1. 短期记忆：会话内多轮对话历史（内存 dict，按 session_id 隔离）
2. 长期记忆：跨会话持久化（JSON 文件），每次对话结束自动提炼写入，
   下次对话开始时按相似度检索相关记忆注入上下文。

用法（任意 Agent）：
    from .memory import MemoryStore
    mem = MemoryStore(memory_dir)
    # 对话前
    recall = mem.recall(question)          # 检索相关长期记忆
    # 对话后
    mem.extract_and_save(question, answer) # 提炼并写入新记忆
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

LONG_TERM_FILE = "long_term.json"


class MemoryStore:
    def __init__(self, memory_dir: Path, max_history: int = 10):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        # 短期记忆：session_id -> 消息列表
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._long_term: List[dict] = self._load()

    # ---------- 长期记忆 ----------
    def _load(self) -> List[dict]:
        p = self.memory_dir / LONG_TERM_FILE
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        p = self.memory_dir / LONG_TERM_FILE
        p.write_text(json.dumps(self._long_term, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def add_memory(self, key: str, value: str, source: str = "assistant") -> None:
        """写入一条长期记忆（相同 key 覆盖旧值并刷新时间）。"""
        now = time.time()
        for item in self._long_term:
            if item["key"] == key:
                item.update({"value": value, "ts": now, "source": source})
                self._save()
                return
        self._long_term.append({"key": key, "value": value, "ts": now, "source": source})
        self._save()

    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """按关键词相似度检索相关记忆（零依赖，中文按字切分）。"""
        if not self._long_term:
            return []
        terms = {c for c in query if c.strip()}
        if not terms:
            return []
        scored = []
        for item in self._long_term:
            text = f"{item['key']} {item['value']}"
            score = sum(1 for c in text if c in terms) / max(1, len(terms))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: -x[0])
        return [f"- {it['key']}: {it['value']}" for _, it in scored[:top_k]]

    def list_all(self) -> List[dict]:
        return sorted(self._long_term, key=lambda x: -x.get("ts", 0))

    def clear(self) -> None:
        self._long_term = []
        self._save()

    # ---------- 短期记忆（多轮对话） ----------
    def history(self, session_id: str) -> List[BaseMessage]:
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        self._sessions.setdefault(session_id, [])
        self._sessions[session_id].append(message)
        # 只保留最近 N 轮
        if len(self._sessions[session_id]) > self.max_history * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history * 2:]

    def add_pair(self, session_id: str, question: str, answer: str) -> None:
        self.add_message(session_id, HumanMessage(content=question))
        self.add_message(session_id, AIMessage(content=answer))

    # ---------- 记忆提炼（需 LLM） ----------
    def extract_memories(self, llm, question: str, answer: str) -> List[Dict[str, str]]:
        """用 LLM 从一轮问答中提炼值得长期记住的事实（键值对）。"""
        prompt = (
            "从下面的问答中提炼「值得长期记住的事实」（用户偏好、决定、新知识），"
            "输出 JSON 数组，每项 {key, value}，key 用简短中文短语。"
            "没有值得记的就输出 []，只输出 JSON：\n\n"
            f"问：{question}\n答：{answer[:2000]}"
        )
        resp = llm.invoke(prompt)
        import re
        try:
            m = re.search(r"\[.*\]", resp.content, re.S)
            data = json.loads(m.group(0))
            return [{"key": d["key"], "value": d["value"]} for d in data if "key" in d and "value" in d]
        except Exception:
            return []
