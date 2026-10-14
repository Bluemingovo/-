# 基于 LangChain 的个人知识库问答系统

一个**真实可运行**的 RAG（检索增强生成）+ 工具调用（Function Calling）+ REST API 项目，
覆盖大模型应用开发的核心链路：**接口调用 → 检索增强 → 函数调用 → 结构化输出 → 服务化 → 部署 → 测试**。
同时提供 **MCP 记忆中枢**：pip 可安装的跨 Agent 共享记忆插件。

## 作为插件安装（任何 Agent 一行接入）

```bash
pip install git+https://github.com/Bluemingovo/Blueming-llm-rag-agent.git

# 自检
llm-rag-mcp --test

# Agent 的 MCP 配置（Claude Desktop / Cursor / Cline / Coze 等）
"memory-hub": { "command": "llm-rag-mcp" }
```

安装后任何支持 MCP 的 Agent 自动获得四个工具：
`read_memory`（读共享记忆）/ `save_memory`（写记忆）/ `list_memories`（列记忆）/ `ask_rag`（RAG 问答）

## 架构

```
                 ┌──────────────┐
 浏览器(静态前端) │   FastAPI    │
    GET /        │  /ask        │
                 │  /index      │
                 │  /docs       │
                 └──────┬───────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   RAG 检索链       函数工具          结构化输出
  文档切分→向量化   list_docs       response_format
  →FAISS 检索      search_files     =json_object
        │               ▲
        └───────┬───────┘
                ▼
         DeepSeek (OpenAI 兼容 API)
         deepseek-chat + tool_calls
```

## 功能

- **RAG 问答**：文档切分 → 向量化 → FAISS 检索 → 带引用回答
- **记忆系统**：短期（多轮对话历史，按 session 隔离）+ 长期（LLM 提炼、跨会话召回，通用可复用模块）
- **工具调用**：本地工具（列出文档 / 搜索文件），模型按需调用
- **结构化输出**：JSON schema 模式（answer / sources / confidence）
- **增量索引**：通过文件 mtime 变更检测，只重索引变更文件（降低重复索引率）
- **REST 接口**：FastAPI 提供 `/ask` `/index` `/docs` `/health` `/memory`
- **Web 前端**：零依赖静态聊天页面（浏览器打开 `http://127.0.0.1:8000/` 即可使用）
- **部署**：Dockerfile + docker-compose 一键启动
- **测试**：pytest 单元测试（离线可跑，不依赖 API key）+ GitHub Actions CI
- **评测**：检索命中率 Hit@k、答案正确率、索引效率对比

## 目录结构

```
llm-rag-agent/
├── src/
│   ├── config.py         # 配置加载
│   ├── embedding.py      # 哈希嵌入（可离线）+ 真实嵌入接口
│   ├── embedding_openai.py
│   ├── documents.py      # 文档加载/切分
│   ├── vector_store.py   # FAISS + 增量索引/变更检测
│   ├── tools.py          # 函数工具（list_docs / search_files）
│   ├── llm.py            # DeepSeek(OpenAI兼容) LLM
│   ├── agent.py          # RAG 主流程（检索+工具调用）
│   ├── app.py            # FastAPI 服务
│   ├── eval.py           # 评测脚本
│   └── selfcheck.py      # 离线自检（无需 API）
├── data/docs/            # 知识库文档
├── eval/questions.json   # 评测集
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key（申请地址 https://platform.deepseek.com）：

```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 3. 离线自检（不需要 API Key）

```bash
python -m src.selfcheck
```

### 4. 建索引

```bash
python -m src.eval --k 4
```

### 5. 启动服务

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

测试接口：

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"二分查找的时间复杂度是多少？\"}"
```

## 接入真实 Embedding（推荐）

默认哈希嵌入仅用于离线验证。正式评测建议换真实模型，例如阿里云 DashScope
`text-embedding-v3`（OpenAI 兼容）。在 `.env` 中设置：

```
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-xxxxxxxx
```

## 实测结果（DeepSeek API）

| 指标 | 结果 |
|---|---|
| 检索命中率 Hit@4（纯向量） | 55%（20 题评测集） |
| 检索命中率 Hit@4（**混合检索 BM25+向量 RRF**） | **100%（20 题），提升 45 个百分点** |
| 答案正确率 | 100%（5/5，deepseek-chat 实测） |
| 增量索引 | 变更 1 个文件仅重嵌入 2 个 chunk（全量 4 个），重复索引率降低约 50% |
| 工具调用 | 文档清单类问题实测触发 list_docs 工具（used_tool=true） |
| 记忆系统 | 短期多轮指代正确；长期记忆跨会话召回实测通过（新会话记得用户偏好） |

三模式对比复现：`python -m scripts.eval_compare --k 4`
| 结构化输出 | JSON schema 模式实测可用（answer/sources/confidence） |
| REST 接口 | /ask /index /docs /health 实测可用 |
| 单元测试 | 13 passed（pytest，离线） |

复现：`python -m src.eval --k 4 --with-llm`

## 测试与 CI

```bash
# 离线单元测试（无需 API key）
python -m pytest tests -v

# 离线自检
python -m src.selfcheck
```

GitHub Actions 在每次 push 时自动运行 pytest（见 `.github/workflows/ci.yml`）。

## Docker 部署

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000 使用聊天页面
```

> 说明：默认哈希嵌入用于离线可复现验证；正式使用建议接入真实 embedding
> （如 DashScope text-embedding-v3），检索质量会更好。

## 评测口径

- **检索命中率 Hit@k**：黄金来源文档是否出现在 top-k 检索结果中。
- **答案正确率**：LLM 回答是否包含黄金答案要点（`--with-llm`）。
- **索引效率**：全量重索引 vs 增量索引的嵌入 chunk 数对比。

## 常见问题

- **`ModuleNotFoundError: langchain_text_splitters`** → 单独 `pip install langchain-text-splitters`
- **DeepSeek 报 401** → API Key 未填或失效，检查 `.env`
- **检索结果不相关** → 换真实 embedding，或调整 `chunk_size`（`documents.py`）
