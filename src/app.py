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
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .config import load_config
from .embedding import build_embedder
from .vector_store import VectorStore
from .agent import RAGAgent
from .tools import set_docs_dir

cfg = load_config()
embedder = build_embedder(cfg.embedding_backend, model=cfg.embedding_model)
store = VectorStore(embedder, cfg.vector_dir, cfg.docs_dir)
set_docs_dir(cfg.docs_dir)

agent = None


def get_agent():
    global agent
    if agent is None:
        agent = RAGAgent(cfg, store)
    return agent

app = FastAPI(title="Personal Knowledge Base RAG", version="1.0.0")


class AskReq(BaseModel):
    question: str
    k: int = 4


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
    a = get_agent().answer(req.question, k=req.k)
    return {"answer": a.text, "sources": a.sources, "used_tool": a.used_tool}


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
