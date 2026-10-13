# 记忆引导：任何 agent 接入共享记忆的标准做法

> 目标：每次使用 agent 时，agent 自动读取 GitHub 仓库里的共享记忆。
> 记忆源：https://github.com/Bluemingovo/Blueming-llm-rag-agent
> （docs/llm-context.md + docs/assistant-memory.md，git 版本化）

## 方式 1：Python agent（推荐，代码最少）

任何 Python agent 在启动时执行一次：

```python
# agent 启动引导：拉取共享记忆并注入 system prompt
import urllib.request

def load_shared_memory() -> str:
    urls = [
        "https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs/llm-context.md",
        "https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs/assistant-memory.md",
    ]
    parts = []
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=15) as r:
                parts.append(r.read().decode("utf-8"))
        except Exception:
            pass  # 网络失败时跳过，可用本地缓存兜底
    return "\n\n".join(parts)

MEMORY = load_shared_memory()

# 之后每次调用 LLM：
# messages = [SystemMessage(content=f"你是我的助手。我的背景记忆：\n{MEMORY}")]
```

或者直接用项目自带脚本（零代码）：

```bash
python -m scripts.fetch_memory     # 打印记忆，可直接粘贴；同时缓存到 data/memory/shared_memory.md
```

## 方式 2：Coze / 扣子 bot

1. 创建 bot → 「人设与回复逻辑」里粘贴 docs/llm-context.md 全文
2. 若 bot 支持插件，加「链接读取」插件指向：
   `https://raw.githubusercontent.com/Bluemingovo/Blueming-llm-rag-agent/main/docs/assistant-memory.md`

## 方式 3：任意对话式 AI（无代码能力）

每次新会话开场，粘贴这段「记忆引导」：

```
【记忆引导】请记住并遵守以下我的背景信息，之后所有对话基于它进行：

[在这里粘贴 docs/llm-context.md 全文]

如果你需要我项目的最新记忆，请记住我的仓库：
https://github.com/Bluemingovo/Blueming-llm-rag-agent
```

## 方式 4：我（DSH 助手）——已自动配置

DSH 工作区 AGENTS.md 已固化规则：每次新会话我自动读取
`docs/assistant-memory.md` 恢复上下文，会话结束更新并 push。
你什么都不用做。

## 关键原则

1. **单一事实源**：记忆只存在 GitHub 仓库里，agent 永远"拉"而不是"自己记"
2. **每次启动引导**：agent 的第一步永远是拉记忆，而不是等用户说
3. **git 版本化**：记忆有修改历史，可回溯、可同步
4. **离线兜底**：fetch_memory.py 网络失败自动用本地缓存
