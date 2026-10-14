"""Streamlit 前端 — 知识库问答助手（一个链接交付）。

本地运行：
    python -m streamlit run streamlit_app.py

云端部署（Streamlit Cloud）：
    1. 推送到 GitHub
    2. share.streamlit.io → New app → 选仓库 → 主文件 streamlit_app.py
    3. Secrets 里配置 DEEPSEEK_API_KEY
    4. 获得公网链接，发给任何人

配置读取顺序：环境变量 → Streamlit Secrets（云端）→ 本地 .env
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# ---------- 配置（云端用 st.secrets，本地用 .env） ----------
import os

def _load_config() -> dict:
    cfg = {}
    try:
        import streamlit as st
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
        cfg.update({k: v for k, v in secrets.items() if isinstance(v, str)})
    except Exception:
        pass
    if not cfg.get("DEEPSEEK_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        cfg = {k: v for k, v in os.environ.items() if v and k in (
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE",
            "EMBEDDING_BACKEND", "VECTOR_DIR", "DOCS_DIR")}
    return cfg


@st.cache_resource
def build_agent():
    from src.config import load_config
    from src.embedding import build_embedder
    from src.vector_store import VectorStore
    from src.agent import RAGAgent
    from src.tools import set_docs_dir
    from src.memory import MemoryStore
    from src.tools import set_memory_store

    env = _load_config()
    if env.get("DEEPSEEK_API_KEY"):
        os.environ.update(env)
    cfg = load_config()
    embedder = build_embedder(cfg.embedding_backend, model=cfg.embedding_model)
    store = VectorStore(embedder, cfg.vector_dir, cfg.docs_dir)
    set_docs_dir(cfg.docs_dir)
    memory = MemoryStore(cfg.vector_dir.parent / "memory")
    set_memory_store(memory)
    return RAGAgent(cfg, store, memory)


# ---------- 页面 ----------
st.set_page_config(page_title="个人知识库问答助手", page_icon="📚")
st.title("📚 个人知识库问答助手")
st.caption("RAG + 工具调用 + 记忆 · 基于 LangChain / FastAPI / DeepSeek")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("来源：" + "、".join(msg["sources"]))

# 输入框
question = st.chat_input("问我知识库里的问题，例如：二分查找的时间复杂度是多少？")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中…")
        try:
            agent = build_agent()
            answer = agent.answer(question, k=6, session_id="streamlit")
            placeholder.markdown(answer.text)
            if answer.sources:
                st.caption("来源：" + "、".join(answer.sources))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "sources": answer.sources})
        except Exception as e:
            placeholder.error(f"出错了：{e}\n\n请检查 DEEPSEEK_API_KEY 是否已配置。")
