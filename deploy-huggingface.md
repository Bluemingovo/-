# 部署到 HuggingFace Spaces（Docker）

> 目标：把这个项目变成**别人点开就能用**的在线 Demo。

## 一、改 3 个地方

### 1. Dockerfile 用 `Dockerfile.hf`（已提供）

HF Spaces 固定监听 **7860** 端口，`Dockerfile.hf` 已经改好：
- `EXPOSE 7860`
- `CMD ... --port 7860`

### 2. Space 仓库 README 顶部加 YAML 头

```yaml
---
title: LLM RAG Agent
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

### 3. 在 Space 设置里加 Secret

- Settings → Variables and secrets → New secret
- `DEEPSEEK_API_KEY` = 你的 key
- 可选：`EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`

## 二、创建 Space

1. 打开 https://huggingface.co/new-space
2. Space name：`llm-rag-agent`
3. SDK：**Docker**
4. Hardware：**CPU basic（免费）**
5. Visibility：Public
6. Create

## 三、推代码

```bash
git clone https://huggingface.co/spaces/你的用户名/llm-rag-agent
cd llm-rag-agent
# 把你的项目文件拷进来
# 用 Dockerfile.hf 覆盖 Dockerfile
cp Dockerfile.hf Dockerfile
# README 加上 YAML 头
git add .
git commit -m "deploy: HF Space"
git push
```

## 四、验证清单

- [ ] Space 状态 Running
- [ ] 首页能加载聊天页
- [ ] `/docs` Swagger 正常
- [ ] 能问答、能返回引用
- [ ] 连续两轮能记住上一轮

## 五、常见坑

| 问题 | 解决 |
|---|---|
| Build 失败 | 检查端口是不是 7860 |
| 打开 404 | 根路径要有页面，或重定向到 /docs |
| API 401 | Secret 名字核对 `DEEPSEEK_API_KEY` |
| Space 休眠 | 免费版 48h 无访问会睡，点开自动唤醒 |
| 数据丢失 | Space 文件系统临时，重启重置（demo 可接受） |
| FAISS 装不上 | Dockerfile 里装 `libgomp1` |
