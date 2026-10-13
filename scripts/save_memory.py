"""一行命令回写记忆：任何 agent/任何人用一句话把新事实存进共享记忆。

用法：
    python -m scripts.save_memory "求职状态" "已投字节，等面试通知"
    python -m scripts.save_memory "新技能" "学会了Docker" --no-push

行为：
1. 追加到 docs/assistant-memory.md 的「最近新事实」区块（带日期）
2. git commit + push 到 GitHub（默认开启；网络失败自动重试几次）
3. 其他 agent 下次 fetch_memory / 读取 raw URL 即看到新记忆
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_FILE = BASE_DIR / "docs" / "assistant-memory.md"
SECTION = "## 最近新事实（agent 回写）"


def append_memory(key: str, value: str) -> None:
    text = MEMORY_FILE.read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    line = f"- {key}：{value}（{today}）"
    if SECTION in text:
        # 插入到区块最后一条之后（区块下一行开始）
        idx = text.index(SECTION)
        tail = text[idx:]
        # 找到区块后第一个空行结束位置
        lines = tail.split("\n")
        # 在区块标题后追加（保持简单：插到文件末尾前）
        text = text.rstrip("\n") + "\n" + line + "\n"
    else:
        text = text.rstrip("\n") + f"\n\n{SECTION}\n\n{line}\n"
    MEMORY_FILE.write_text(text, encoding="utf-8")


def commit_and_push(retries: int = 5) -> bool:
    try:
        subprocess.run(["git", "-C", str(BASE_DIR), "add", "docs/assistant-memory.md"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(BASE_DIR), "-c", "user.name=ZumingSheng",
                        "-c", "user.email=zumingsheng@outlook.com",
                        "commit", "-m", "memory: agent 回写新事实"],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[warn] 本地提交失败：{e.stderr.decode('utf-8', 'ignore')[:200]}")
        return False
    import time
    for _ in range(retries):
        r = subprocess.run(["git", "-C", str(BASE_DIR), "push", "origin", "main"],
                           capture_output=True)
        if r.returncode == 0:
            return True
        time.sleep(15)
    print("[warn] push 未成功（网络），记忆已存本地，稍后可用 git push 补推")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", help="记忆条目名（简短中文短语）")
    ap.add_argument("value", help="记忆内容")
    ap.add_argument("--no-push", action="store_true", help="只写本地不推送")
    args = ap.parse_args()

    append_memory(args.key, args.value)
    print(f"✅ 记忆已写入：{args.key}：{args.value}")
    if not args.no_push:
        ok = commit_and_push()
        print("✅ 已提交并推送 GitHub" if ok else "⚠️ 推送待补")
        print("其他 agent 运行 fetch_memory 或读 raw URL 即可看到新记忆")


if __name__ == "__main__":
    main()
