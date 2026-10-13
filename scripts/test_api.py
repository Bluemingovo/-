"""用 Python 客户端测试 /ask 接口（避免终端中文编码问题）。"""
import sys, json, urllib.request

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8124"

def ask(question):
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(BASE + "/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))

for q in ["快速排序的平均时间复杂度是多少？", "我的知识库里现在有哪些文档？"]:
    res = ask(q)
    print("Q:", q)
    print("A:", res["answer"][:300])
    print("sources:", res["sources"], "| used_tool:", res["used_tool"])
    print("-" * 60)
