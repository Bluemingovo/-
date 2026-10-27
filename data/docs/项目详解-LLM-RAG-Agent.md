# 项目详解：LLM RAG Agent

## 项目简介
一个可运行、可安装、可复现的 RAG 知识库问答系统，包含检索增强、Agent 记忆与 MCP 插件分发。
GitHub：github.com/Bluemingovo/Blueming-llm-rag-agent

## 核心功能
1. **RAG 问答**：文档切分 → 向量化 → FAISS 检索 → 带引用回答
2. **混合检索**：BM25 关键词 + 向量语义，RRF 倒数排名融合
3. **记忆系统**：短期多轮 + 长期跨会话 + LLM 自主写入
4. **MCP 插件**：记忆中枢封装为 MCP Server，pip 可安装
5. **工程化**：FastAPI + Docker + pytest(13) + GitHub Actions CI

## 核心结果
- 检索命中率 Hit@4：纯向量 55% → 混合检索 100%（20 题评测集）
- 答案正确率：100%（5/5，deepseek-chat）
- faithfulness / answer_relevancy：1.0 / 1.0
- 单元测试：13 个全过
- 增量索引：重复索引率 -50%

## 设计决策
1. **为什么用混合检索**：纯向量不擅长精确关键词，BM25 反过来，两者互补
2. **为什么用 RRF 而不是加权求和**：RRF 只依赖排名、不依赖分数尺度，更鲁棒、更少参数
3. **为什么记忆分短期/长期/自主写入**：短期保证多轮连贯，长期跨会话召回，自主写入让模型主动记忆
4. **为什么封装成 MCP**：MCP 是 Agent 工具标准协议，pip 一装，Claude/Cursor/Cline 都能用
5. **增量索引**：按文件 mtime 检测变更，只重索引变更文件

## 技术栈
- Python、FastAPI、FAISS、BM25、RRF、LangChain
- DeepSeek API（OpenAI 兼容）
- Docker、pytest、GitHub Actions、MCP

## 评测方法
- 20 题人工评测集，覆盖关键词类 / 语义类 / 多跳类
- 指标：Hit@4（前 4 条检索结果是否包含答案 chunk）
- 局限：20 题样本偏小，结论是方向性的

## 局限与改进
- 评测集扩到 50~100 题
- 默认哈希嵌入换成真实 embedding（DashScope text-embedding-v3）
- 加“纯 LLM 无 RAG”baseline 对照
