# 基于 LangChain 的个人知识库问答系统

一个**真实可运行**的 RAG（检索增强生成）+ 工具调用（Function Calling）+ REST API 项目。
用于把「用 AI」升级成「做 AI 应用」：证明你能独立完成大模型接口调用、向量检索、
函数调用与工程化交付。

## 功能

- **RAG 问答**：文档切分 → 向量化 → FAISS 检索 → 带引用回答
- **工具调用**：本地工具（列出文档 / 搜索文件），模型按需调用
- **增量索引**：通过文件 mtime 变更检测，只重索引变更文件（降低重复索引率）
- **REST 接口**：FastAPI 提供 `/ask` `/index` `/docs` `/health`
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
| 检索命中率 Hit@4 | 100%（5 题评测集） |
| 答案正确率 | 100%（5/5，deepseek-chat 实测） |
| 增量索引 | 变更 1 个文件仅重嵌入 2 个 chunk（全量 4 个），重复索引率降低约 50% |
| 工具调用 | 文档清单类问题实测触发 list_docs 工具（used_tool=true） |
| REST 接口 | /ask /index /docs /health 实测可用 |

复现：`python -m src.eval --k 4 --with-llm`

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
