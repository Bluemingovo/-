"""端到端验证：多轮对话记忆 + 长期记忆提炼。"""
import sys, json, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8126"

def ask(question, session_id="s1", remember=False):
    body = json.dumps({"question": question, "session_id": session_id,
                       "remember": remember}).encode("utf-8")
    req = urllib.request.Request(BASE + "/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))

# 多轮 1：正常问答
r1 = ask("二分查找的时间复杂度是多少？")
print("轮1:", r1["answer"][:60].replace("\n", " "))

# 多轮 2：用"它"指代，验证短期记忆
r2 = ask("那它和顺序查找谁更快？")
print("轮2:", r2["answer"][:80].replace("\n", " "))

# 多轮 3：remember=True 提炼长期记忆
r3 = ask("我最近在复习数据结构，帮我记住这个偏好", remember=True)
print("轮3:", r3["answer"][:60].replace("\n", " "), "| 提炼记忆数:", r3.get("memories_saved"))

# 新会话（无历史）问偏好——验证长期记忆召回
body = json.dumps({"question": "我之前说过我在复习什么？", "session_id": "s2"}).encode("utf-8")
req = urllib.request.Request(BASE + "/ask", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    r4 = json.loads(r.read().decode("utf-8"))
print("新会话轮4:", r4["answer"][:120].replace("\n", " "))

# 查看长期记忆库
with urllib.request.urlopen(BASE + "/memory", timeout=10) as r:
    mem = json.loads(r.read().decode("utf-8"))
print("长期记忆条数:", len(mem["memories"]))
for m in mem["memories"][:5]:
    print(" -", m["key"], "=>", m["value"][:50])
