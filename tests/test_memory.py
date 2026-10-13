"""记忆模块离线测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from src.memory import MemoryStore


def test_long_term_add_and_recall(tmp_path):
    mem = MemoryStore(tmp_path)
    mem.add_memory("用户偏好", "喜欢简洁的中文回答", source="test")
    mem.add_memory("项目背景", "正在投递 AI 应用开发实习", source="test")

    hits = mem.recall("用户喜欢什么风格的答案")
    assert hits, "应召回相关记忆"
    assert "简洁" in hits[0]

    # 与查询无关的记忆不应出现在首位
    hits2 = mem.recall("项目背景是什么")
    assert any("投递" in h for h in hits2)


def test_long_term_overwrite_same_key(tmp_path):
    mem = MemoryStore(tmp_path)
    mem.add_memory("城市", "上海", source="test")
    mem.add_memory("城市", "杭州", source="test")
    assert len(mem.list_all()) == 1
    assert mem.list_all()[0]["value"] == "杭州"


def test_short_term_history_cap(tmp_path):
    mem = MemoryStore(tmp_path, max_history=2)
    for i in range(5):
        mem.add_pair("s1", f"问题{i}", f"答案{i}")
    # max_history=2 → 只保留最近 2 轮 = 4 条消息
    assert len(mem.history("s1")) == 4
    # 会话隔离
    assert mem.history("s2") == []


def test_clear(tmp_path):
    mem = MemoryStore(tmp_path)
    mem.add_memory("k", "v", source="test")
    mem.clear()
    assert mem.list_all() == []


def test_extract_memories_offline_format(tmp_path):
    """不调 LLM：验证提炼函数对非 JSON 输出容错（返回空列表不崩溃）。"""
    mem = MemoryStore(tmp_path)

    class FakeLLM:
        def invoke(self, prompt):
            class R:
                content = "不是 JSON"
            return R()

    assert mem.extract_memories(FakeLLM(), "q", "a") == []

    class JSONLLM:
        def invoke(self, prompt):
            class R:
                content = '[{"key": "用户喜欢", "value": "Python"}]'
            return R()

    out = mem.extract_memories(JSONLLM(), "q", "a")
    assert out == [{"key": "用户喜欢", "value": "Python"}]
