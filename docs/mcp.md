# MCP 记忆中枢：高级接入方式

> 把共享记忆做成 MCP Server（Model Context Protocol，行业标准协议）。
> 任何支持 MCP 的 Agent（Claude Desktop / Cursor / Cline / Coze 等）**配置一次**，
> 之后每次对话自动拥有「读记忆 / 写记忆 / 列记忆 / 问知识库」四个工具，
> 不需要贴链接、不需要手动操作。

## 工具列表

| 工具 | 作用 | Agent 何时调用 |
|---|---|---|
| `read_memory` | 读取全部共享记忆 | 对话开始时（了解用户背景） |
| `save_memory` | 写入一条长期记忆 | 用户透露值得记的事时（自主判断） |
| `list_memories` | 列出运行时记忆库 | 需要浏览记忆时 |
| `ask_rag` | 用你的 RAG 知识库问答 | 专业问题（代码、笔记） |

## 接入步骤（以 Claude Desktop 为例，2 分钟）

1. 安装依赖：`pip install mcp`
2. 打开 Claude Desktop → Settings → Developer → Edit Config
3. 在 `claude_desktop_config.json` 的 `mcpServers` 里加：

```json
{
  "mcpServers": {
    "memory-hub": {
      "command": "python",
      "args": ["C:\\Users\\zms\\Documents\\deepseek-harness\\default-workspace\\llm-rag-agent\\scripts\\mcp_server.py"],
      "env": { "PYTHONIOENCODING": "utf-8" }
    }
  }
}
```

4. 重启 Claude Desktop → 设置里能看到 memory-hub 及其 4 个工具 → 完成

之后对 Claude 说"读一下我的记忆"或直接开始对话，它会自动用工具。

## 其他平台

- **Cursor**：Settings → MCP → Add new MCP server，command 同上
- **Cline**：`cline_mcp_settings.json` 加同结构配置
- **Coze 国际版**：Plugin 市场搜 MCP，或自定义 MCP 连接，command 同上
- **任何支持 MCP 的 Agent**：配置结构都一样（command + args）

## 本地自检

```bash
python -m scripts.mcp_server --test
```

输出 read_memory 长度、写入测试记忆、列出记忆库，即为正常。

## 架构

```
 Claude / Cursor / Cline / Coze / ...
        │  MCP 协议（stdio，行业标准）
        ▼
  scripts/mcp_server.py（记忆中枢）
        ├─ read_memory  → docs/llm-context.md + assistant-memory.md（GitHub 版本化）
        ├─ save_memory  → 写共享记忆文件 + 运行时记忆库
        ├─ list_memories→ data/memory/long_term.json
        └─ ask_rag      → 项目 RAG 链路（FAISS + DeepSeek）
```
