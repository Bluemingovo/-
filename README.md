# LLM RAG Agent — 带记忆的 RAG 问答系统与 MCP 插件

[![CI](https://github.com/Bluemingovo/Blueming-llm-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Bluemingovo/Blueming-llm-rag-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![MCP](https://img.shields.io/badge/MCP-Server-purple)
![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen)

一个可运行、可安装、可复现的 **RAG 知识库问答系统**，包含检索增强、Agent 记忆与 MCP 插件分发。

**核心结果：混合检索使 20 题评测集命中率从 55% 提升至 100%。**

## 亮点

| 能力 | 说明 | 可验证 |
|---|---|---|
| **混合检索** | 自研 BM25 + 向量检索 RRF 融合，解决关键词类问题漏检 | `scripts/eval_compare.py` 三模式对比 |
| **Agent 记忆** | 短期多轮 + 长期跨会话召回 + LLM 自主记忆（save_memory 工具） | 跨会话召回实测 |
| **MCP 插件** | 记忆中枢封装为 MCP Server，pip 安装后任何 Agent 接入 | 标准客户端端到端验证 |
| **工程化** | FastAPI + Docker + pytest(13) + GitHub Actions CI | 一键复现 |

## 快速开始（3 步）

```bash
# 1. 安装
pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY

# 2. 离线自检（不需要 API Key）
python -m src.selfcheck

# 3. 启动服务
python -m uvicorn src.app:app --port 8000
# 浏览器打开 http://127.0.0.1:8000 聊天；接口文档 http://127.0.0.1:8000/docs
```

## 作为 MCP 插件安装

```bash
pip install git+https://github.com/Bluemingovo/Blueming-llm-rag-agent.git
llm-rag-mcp --test    # 自检
```

在任何支持 MCP 的 Agent（Claude Desktop / Cursor / Cline / Coze）配置：

```json
"memory-hub": { "command": "llm-rag-mcp" }
```

获得的工具：`read_memory` / `save_memory` / `list_memories` / `ask_rag`

## 架构

```
                ┌──────────────────┐
 浏览器 / Agent │     FastAPI      │
  GET /         │ /ask /index /docs│
                │ /health /memory  │
                └────────┬─────────┘
                         │
      ┌──────────────────┼───────────────────┐
      ▼                  ▼                   ▼
 混合检索          函数工具             记忆系统
 BM25 + FAISS     list_docs            短期多轮
 RRF 融合         search_files         长期召回
      │                  │             自主写入
      └────────┬─────────┴───────────────────┘
               ▼
      DeepSeek (OpenAI 兼容) — tool_calls
               ▲
               │ MCP (stdio)
        Claude / Cursor / Cline
```

## 功能清单

- **RAG 问答**：文档切分 → 向量化 → FAISS 检索 → 带引用回答（自动标注来源）
- **混合检索**：BM25 关键词 + 向量语义，RRF 倒数排名融合（命中率 55% → 100%）
- **记忆系统**：短期（session 隔离多轮）+ 长期（LLM 提炼、跨会话召回）+ LLM 自主记忆（工具调用写入）
- **MCP 记忆中枢**：四工具记忆服务，pip 可安装，跨 Agent 共享
- **工具调用**：`list_docs` / `search_files`，模型按需调用（实测触发）
- **结构化输出**：JSON Schema（answer / sources / confidence）
- **增量索引**：文件 mtime 变更检测，只重索引变更文件（重复索引率 -50%）
- **REST API**：`/ask` `/index` `/docs` `/health` `/memory` `/assistant-memory`
- **Web 前端**：零依赖静态聊天页
- **部署与测试**：Docker Compose、pytest（13 个）、GitHub Actions CI
- **评测**：Hit@k、LLM-as-judge（faithfulness / answer_relevancy / context_precision）

## 目录结构

```
llm-rag-agent/
├── src/
│   ├── config.py         配置加载
│   ├── documents.py      文档加载/切分（损坏文件容错）
│   ├── embedding.py      哈希嵌入（离线）+ OpenAI 兼容嵌入
│   ├── vector_store.py   FAISS + 增量索引 + 三种检索模式
│   ├── hybrid.py         BM25 自实现 + RRF 融合
│   ├── rerank.py         重排模块（可插拔，无 key 自动退化）
│   ├── memory.py         记忆模块（短期/长期/自主记忆）
│   ├── tools.py          函数工具（含 save_memory）
│   ├── agent.py          RAG 主流程
│   ├── app.py            FastAPI 服务
│   └── eval.py           评测脚本
├── scripts/
│   ├── mcp_server.py     MCP 记忆中枢
│   ├── eval_compare.py   三模式检索对比
│   ├── eval_ragas.py     LLM-as-judge 评测
│   ├── ingest_github.py  GitHub 仓库入知识库
│   ├── fetch_memory.py   记忆拉取
│   └── save_memory.py    一行回写记忆
├── tests/                pytest 用例（13 个）
├── data/docs/            知识库文档
├── eval/questions.json   20 题评测集
├── Dockerfile / docker-compose.yml
├── pyproject.toml        pip 打包配置
└── .env.example
```

## 实测结果

| 指标 | 结果 |
|---|---|
| 检索命中率 Hit@4（纯向量） | **55%**（20 题评测集） |
| 检索命中率 Hit@4（混合检索 BM25+RRF） | **100%**，提升 45 个百分点 |
| 答案正确率 | 100%（5/5，deepseek-chat） |
| faithfulness / answer_relevancy | 1.0 / 1.0（LLM-as-judge，5 题） |
| 增量索引 | 变更 1 文件仅重嵌入 2 chunk（全量 4），重复索引率 -50% |
| 单元测试 | 13 passed（离线） |
| MCP 协议 | 标准客户端 stdio 端到端通过 |

复现命令：

```bash
python -m scripts.eval_compare --k 4    # 三模式对比
python -m scripts.eval_ragas            # LLM-as-judge 评测
python -m pytest tests -v               # 单元测试
```

## Docker 部署

```bash
docker compose up --build
# 打开 http://localhost:8000
```

## 接入真实 Embedding（推荐）

默认哈希嵌入仅用于离线复现。正式使用建议换真实模型（阿里云 DashScope `text-embedding-v3`）：

```
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-xxxxxxxx
```

## 常见问题

- `ModuleNotFoundError: langchain_text_splitters` → `pip install langchain-text-splitters`
- DeepSeek 401 → 检查 `.env` 中的 API Key
- 检索结果不相关 → 换真实 embedding，或调整 `chunk_size`（`documents.py`）

## License

MIT
