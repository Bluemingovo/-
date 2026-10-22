# LLM 上下文（喂给任何大模型的浓缩版项目说明）

> 用途：在任何 agent 调用 LLM 的 system prompt 里附上本文件内容，
> 让模型"了解"这个仓库。复制粘贴即可，无需联网。

## 项目：基于 LangChain 的个人知识库问答系统

仓库：https://github.com/Bluemingovo/Blueming-llm-rag-agent
作者：祖名省（上海师范大学天华学院 通信工程 本科在读，2027 毕业，求职 AI 应用开发实习）

### 一句话
个人知识库 RAG 问答系统：文档 → 切分 → 向量化(FAISS) → 检索 → DeepSeek 大模型带引用回答，
支持函数调用工具、双层记忆（含 LLM 自主记忆）、MCP Server、结构化输出、FastAPI 服务、Docker 部署与 pytest 测试。

### 技术栈
Python 3.12 / LangChain / FAISS / FastAPI / DeepSeek(OpenAI 兼容) / MCP / Docker / pytest / GitHub Actions

### 架构（数据流）
用户问题 → 长期记忆召回(memory.py) → 向量检索 top-k(vector_store.py)
→ 拼接上下文 + 会话历史 → LLM(带函数工具 bind_tools) → 答案 + 引用来源
→ 对话后：写入短期历史；LLM 自主调用 save_memory 工具记录值得记的事实

### 核心模块
- src/documents.py    文档加载(txt/md/pdf)与中文切分(400/50)，损坏文件容错
- src/embedding.py    哈希嵌入(离线可复现占位) + OpenAI 兼容嵌入接口
- src/vector_store.py FAISS + mtime 变更检测的增量索引(重复索引率-50%)
- src/memory.py       通用记忆模块：短期(会话隔离多轮) + 长期(JSON+LLM提炼+跨会话召回)
- src/tools.py        函数工具：list_docs / search_files / save_memory(自主记忆)
- src/agent.py        RAG 主流程 + 结构化输出(JSON Schema)
- src/app.py          FastAPI：/ask /index /docs /health /memory /assistant-memory + 静态聊天前端
- scripts/mcp_server.py   MCP 记忆中枢：read_memory/save_memory/list_memories/ask_rag 四工具
- scripts/ingest_github.py 任意 GitHub 仓库索引进知识库
- scripts/fetch_memory.py / save_memory.py 记忆拉取与一行回写
- src/eval.py         评测：Hit@k / 答案正确率 / 索引效率

### 实测数字（小样本，口径可复现）
- 检索命中率 Hit@4 = 100%（混合检索，20 题评测集；纯向量 55% → 混合 100%，+45pp）
- 答案正确率 100%（DeepSeek deepseek-chat 实测 5/5）
- 增量索引：变更 1 文件仅重嵌 2 chunk（全量 4）→ 重复索引率 -50%
- 单元测试 13 个全过（pytest 离线）；长期记忆跨会话召回实测通过
- MCP 协议端到端验证通过（标准客户端 stdio 连接，4 工具注册与调用）

### 已知取舍
- 哈希嵌入是离线占位实现，README 有 DashScope text-embedding-v3 接入方案
- 评测集 20 题（已从 5 题扩容完成，仍属小样本）
