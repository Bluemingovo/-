"""增量索引效果验证：改一个文件，看只重嵌入多少 chunk。"""
from __future__ import annotations
import sys, time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config
from src.embedding import build_embedder
from src.vector_store import VectorStore

cfg = load_config()
store = VectorStore(build_embedder(cfg.embedding_backend), cfg.vector_dir, cfg.docs_dir)

s1 = store.index()
print("第一次索引:", s1)
s2 = store.index()
print("无变更再索引:", s2)

# 修改一个文件，追加一行
p = cfg.docs_dir / "Python基础速查.md"
p.write_text(p.read_text(encoding="utf-8") + "\n## 追加内容\n列表推导式也可以嵌套使用。\n", encoding="utf-8")
time.sleep(1.1)  # 确保 mtime 变化
s3 = store.index()
print("修改1个文件后再索引:", s3)
