# LLM RAG Agent — 带记忆的 RAG 问答系统与 MCP 插件

[![CI](https://github.com/Bluemingovo/Blueming-llm-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Bluemingovo/Blueming-llm-rag-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![MCP](https://img.shields.io/badge/MCP-Server-purple)
![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen)

> 一个可运行、可安装、可复现的 **RAG 知识库问答系统**，包含检索增强、Agent 记忆与 MCP 插件分发。
>
> **核心结果：混合检索使 20 题评测集命中率从 55% 提升至 100%。**

🔗 **在线 Demo**：[blueming-llm-rag-agent-3o6iwfctgh6tpqnqeykxrp.streamlit.app](https://blueming-llm-rag-agent-3o6iwfctgh6tpqnqeykxrp.streamlit.app/)

---

## 一、为什么做这个

纯 RAG 有几个真实痛点，这个项目是围绕它们设计的：

1. **纯向量检索对“关键词类问题”漏检严重** —— 用户问一个术语/编号，向量相似度反而不如关键词匹配
2. **Agent 会“失忆”** —— 多轮对话记不住，跨会话更记不住
3. **记忆能力不该绑死在一个 Agent 里** —— 想把它标准化，让 Claude/Cursor/Cline 都能接入

所以做了三件事：**混合检索**、**分层记忆**、**MCP 封装**。

---

## 二、核心结果（可复现）

| 指标 | 结果 | 复现命令 |
|---|---|---|
| 检索命中率 Hit@4（纯向量） | **55%**（20 题评测集） | `python -m scripts.eval_compare --k 4` |
| 检索命中率 Hit@4（BM25+RRF 混合） | **100%**（提升 45 个百分点） | 同上 |
| 答案正确率 | 100%（5/5，deepseek-chat） | — |
| faithfulness / answer_relevancy | 1.0 / 1.0（LLM-as-judge，5 题） | `python -m scripts.eval_ragas` |
| 增量索引 | 变更 1 文件仅重嵌入 2 chunk（重复索引率 -50%） | — |
| 单元测试 | **13 passed**（离线可跑） | `python -m pytest tests -v` |
| MCP 协议 | 标准客户端 stdio 端到端通过 | `llm-rag-mcp --test` |

---

## 三、设计决策（面试重点）

### 1. 为什么用「BM25 + 向量 + RRF」，而不是只用向量？
纯向量擅长语义、不擅长精确关键词；BM25 反过来。**两者互补**，实测把关键词类问题的命中率从 55% 拉到 100%。

### 2. 为什么用 RRF（倒数排名融合），而不是加权求和？
RRF 只依赖**排名**、不依赖**分数尺度**——BM25 和余弦相似度的分数量纲完全不同，加权求和需要调权重且不稳定；RRF 无需归一化，**更鲁棒、更少参数**。

### 3. 记忆系统为什么分「短期 / 长期 / 自主写入」？
- **短期**：session 隔离，保证多轮对话连贯
- **长期**：LLM 提炼要点、跨会话召回
- **自主写入**：模型通过 `save_memory` 工具主动记忆，而不是被动全量存

### 4. 为什么封装成 MCP？
MCP 是 Agent 工具的标准协议。封装后 **pip 一装，Claude/Cursor/Cline 都能用**，从“一个项目”变成“一个可复用的能力”。

### 5. 增量索引怎么做的？
按文件 **mtime** 检测变更，只重索引变更文件，避免全量重嵌入（实测重复索引率 -50%）。

---

## 四、评测方法（诚实版）

- **评测集**：20 个问题，人工构造，覆盖关键词类 / 语义类 / 多跳类
- **指标**：Hit@4（前 4 条检索结果是否包含答案所在 chunk）
- **对比**：纯向量 / 纯 BM25 / 混合（RRF）三模式

**局限（必须说清楚）**：
- **20 题样本偏小**，55%→100% 是“方向性结论”，不是统计显著结论
- 后续计划扩到 **50~100 题**，并加入置信区间

> 我宁愿把局限写出来，也不愿用一个小样本得出“100%”的夸张结论。

---

## 五、快速开始（3 步）

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

## 六、作为 MCP 插件安装

```bash
pip install git+https://github.com/Bluemingovo/Blueming-llm-rag-agent.git
llm-rag-mcp --test    # 自检
```

在任何支持 MCP 的 Agent（Claude Desktop / Cursor / Cline / Coze）配置：

```json
"memory-hub": { "command": "llm-rag-mcp" }
```

获得的工具：`read_memory` / `save_memory` / `list_memories` / `ask_rag`

---

## 七、架构

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

---

## 八、功能清单

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

---

## 九、目录结构

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

---

## 十、局限与改进

| 局限 | 改进计划 |
|---|---|
| 评测集只有 20 题 | 扩到 50~100 题，加置信区间 |
| 默认哈希嵌入（离线用） | 生产切真实 embedding（DashScope `text-embedding-v3`） |
| 没有线上 demo | **已部署 Streamlit Cloud**（见顶部链接） |
| 没有 baseline（纯 LLM 无 RAG） | 加一个“无检索”对照组 |
| 单机内存存储 | 可换 Redis/Postgres |

---

## 十一、Docker 部署

```bash
docker compose up --build
# 打开 http://localhost:8000
```

## 十二、接入真实 Embedding（推荐）

默认哈希嵌入仅用于离线复现。正式使用建议换真实模型（阿里云 DashScope `text-embedding-v3`）：

```
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-xxxxxxxx
```

## License

MIT
