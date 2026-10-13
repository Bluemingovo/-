"""FastAPI 服务：把 RAG 问答系统包装成 REST 接口。

接口：
- POST /ask      {question, k?}  -> {answer, sources, used_tool}
- POST /index    {force?}        -> 索引统计（触发增量索引）
- GET  /docs                     -> 当前文档清单
- GET  /health                   -> 健康检查
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .config import load_config
from .embedding import build_embedder
from .vector_store import VectorStore
from .agent import RAGAgent
from .tools import set_docs_dir
from .memory import MemoryStore

cfg = load_config()
embedder = build_embedder(cfg.embedding_backend, model=cfg.embedding_model)
store = VectorStore(embedder, cfg.vector_dir, cfg.docs_dir)
set_docs_dir(cfg.docs_dir)
memory = MemoryStore(cfg.vector_dir.parent / "memory")
from .tools import set_memory_store
set_memory_store(memory)

agent = None


def get_agent():
    global agent
    if agent is None:
        agent = RAGAgent(cfg, store, memory)
    return agent

app = FastAPI(title="Personal Knowledge Base RAG", version="1.0.0")

# 静态前端（聊天页面）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    from fastapi.responses import FileResponse
    index = STATIC_DIR / "index.html"
    return FileResponse(str(index)) if index.exists() else {"hint": "GET /docs 查看接口"}


class AskReq(BaseModel):
    question: str
    k: int = 6
    structured: bool = False  # 结构化输出（JSON schema）
    session_id: str = "default"  # 会话 id：多轮对话记忆隔离
    remember: bool = False  # 是否提炼长期记忆（额外一次 LLM 调用）


class IndexReq(BaseModel):
    force: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskReq):
    if not cfg.api_key:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503,
                            content={"error": "未配置 DEEPSEEK_API_KEY，请在 .env 中填写后重启服务"})
    a = get_agent().answer(req.question, k=req.k, structured=req.structured,
                           session_id=req.session_id, remember=req.remember)
    return {"answer": a.text, "sources": a.sources, "used_tool": a.used_tool,
            "memories_saved": a.memories_saved}


@app.get("/memory")
def memory_list():
    """查看运行时长期记忆。"""
    return {"memories": memory.list_all()}


@app.delete("/memory")
def memory_clear():
    """清空长期记忆。"""
    memory.clear()
    return {"status": "cleared"}


# ---- 助手记忆（跨 agent 共享记忆：任何 agent 通过 HTTP 拉取） ----
ASSISTANT_MEMORY_FILE = Path(__file__).resolve().parent.parent / "docs" / "assistant-memory.md"
LLM_CONTEXT_FILE = Path(__file__).resolve().parent.parent / "docs" / "llm-context.md"


@app.get("/assistant-memory")
def assistant_memory_get():
    """返回共享记忆（文本）：其他 agent 调这个接口即可拿到最新记忆。"""
    parts = []
    if LLM_CONTEXT_FILE.exists():
        parts.append(LLM_CONTEXT_FILE.read_text(encoding="utf-8"))
    if ASSISTANT_MEMORY_FILE.exists():
        parts.append(ASSISTANT_MEMORY_FILE.read_text(encoding="utf-8"))
    return {"memory": "\n\n".join(parts)}


@app.post("/assistant-memory")
def assistant_memory_update(req: dict):
    """其他 agent 写入记忆条目：{key, value}，追加到共享记忆文件。"""
    key = req.get("key", "").strip()
    value = req.get("value", "").strip()
    if not key or not value:
        return {"status": "ignored"}
    ASSISTANT_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ASSISTANT_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n- {key}: {value}\n")
    return {"status": "saved", "key": key}


@app.post("/index")
def index(req: IndexReq):
    stats = store.index(force=req.force)
    return stats


@app.get("/docs")
def docs():
    from .tools import list_docs
    return {"docs": list_docs.invoke({})}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
