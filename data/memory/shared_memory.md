# LLM 上下文（喂给任何大模型的浓缩版项目说明）

> 用途：在任何 agent 调用 LLM 的 system prompt 里附上本文件内容，
> 让模型"了解"这个仓库。复制粘贴即可，无需联网。

## 项目：基于 LangChain 的个人知识库问答系统

仓库：https://github.com/Bluemingovo/Blueming-llm-rag-agent
作者：祖名省（上海师范大学天华学院 通信工程 本科在读，2027 毕业，求职 AI 应用开发实习）

### 一句话
个人知识库 RAG 问答系统：文档 → 切分 → 向量化(FAISS) → 检索 → DeepSeek 大模型带引用回答，
支持函数调用工具、记忆系统、结构化输出、FastAPI 服务、Docker 部署与 pytest 测试。

### 技术栈
Python 3.12 / LangChain / FAISS / FastAPI / DeepSeek(OpenAI 兼容) / Docker / pytest / GitHub Actions

### 架构（数据流）
用户问题 → 长期记忆召回(memory.py) → 向量检索 top-k(vector_store.py)
→ 拼接上下文 + 会话历史 → LLM(带函数工具 bind_tools) → 答案 + 引用来源
→ 对话后：写入短期历史；remember=True 时 LLM 提炼长期记忆

### 核心模块
- src/documents.py    文档加载(txt/md/pdf)与中文切分(400/50)，损坏文件容错
- src/embedding.py    哈希嵌入(离线可复现占位) + OpenAI 兼容嵌入接口
- src/vector_store.py FAISS + mtime 变更检测的增量索引(重复索引率-50%)
- src/memory.py       通用记忆模块：短期(会话隔离多轮) + 长期(JSON+LLM提炼+跨会话召回)
- src/tools.py        函数工具：list_docs / search_files
- src/agent.py        RAG 主流程 + 结构化输出(JSON Schema)
- src/app.py          FastAPI：/ask /index /docs /health /memory + 静态聊天前端
- src/eval.py         评测：Hit@k / 答案正确率 / 索引效率

### 实测数字（小样本，口径可复现）
- 检索命中率 Hit@4 = 100%（5 题评测集）
- 答案正确率 100%（DeepSeek deepseek-chat 实测 5/5）
- 增量索引：变更 1 文件仅重嵌 2 chunk（全量 4）→ 重复索引率 -50%
- 单元测试 13 个全过（pytest 离线）；长期记忆跨会话召回实测通过

### 已知取舍
- 哈希嵌入是离线占位实现，README 有 DashScope text-embedding-v3 接入方案
- 评测集仅 5 题（小样本），正在扩容


# 助手记忆（每次会话结束自动更新）

> 本文件是 AI 助手对用户的长期记忆，存储在项目仓库中（git 版本化）。
> 每次会话结束时由助手更新；新会话开始时助手读取此文件恢复上下文。

## 用户画像

- 姓名：祖名省；求职意向：AI 应用开发/大模型应用开发（实习）
- 学校：上海师范大学天华学院 · 通信工程 · 本科 · 2023.09–2027.06（2027 年毕业）
- 所在地：上海；手机 18075020923；邮箱 zumingsheng@outlook.com
- 背景：国家励志奖学金；微专业集成电路、职场胜任力；核心课含人工智能技术、C 语言、概率论
- 经历：Coze 多轮需求澄清助手（60%→86%）；Transformer 情感分类课程设计（约 90%）；校园活动运营（AI 提效 40→25 分钟）
- 近期状态：正在复习数据结构与算法（已存入项目长期记忆）

## 项目状态

- 仓库：https://github.com/Bluemingovo/Blueming-llm-rag-agent（本地：default-workspace\llm-rag-agent）
- 已实现：RAG 检索链、工具调用、增量索引、结构化输出、记忆系统（短期+长期）、GitHub 仓库摄取（ingest_github.py）、LLM 上下文文件（docs/llm-context.md）、FastAPI、静态前端、Docker、pytest（13 个）、GitHub Actions CI
- 真实数字：Hit@4=100%（5 题）、答案正确率 100%（5/5）、增量索引 -50%、13 tests passed、跨会话记忆召回通过、问仓库内容可正确回答（k=6）
- 待迭代：真实 Embedding（DashScope）、流式输出、多轮记忆 UI、评测集扩容

## 用户偏好与约定

- 文件管理：简历只维护一份（桌面\简历\祖名省-AI应用开发实习生.html + .pdf），禁止新建副本；投递版 PDF 可放公司子文件夹
- 进度面板：default-workspace\求职作战面板.md（唯一），每次会话有进展都要更新
- 助手记忆：本文件，每次会话结束更新（新增事实、纠正旧记忆）
- 沟通：中文；行动导向（用户偏好"直接帮他做"，而非只给建议）
- 诚实原则：简历与项目内容必须真实可验证，不编造经历与数字
- 简历中个人信息与日期（2026.04/2026.09 等）待用户核对

## 待办（用户侧）

- 通读 src/ 代码；模拟面试 3 轮
- 往 data/docs 丢真实学习资料并日常使用
- 核对简历信息后投递：中证指数、立邦、亚信、贝加莱
