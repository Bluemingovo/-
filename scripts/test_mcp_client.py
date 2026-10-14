"""用标准 MCP 客户端验证 mcp_server（真实协议测试）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "scripts" / "mcp_server.py")],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("=== 服务器注册的工具 ===")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description[:50]}")

            print("\n=== 调用 read_memory ===")
            r = await session.call_tool("read_memory", {})
            text = r.content[0].text if r.content else ""
            print(f"  返回 {len(text)} 字符，开头: {text[:60]}")

            print("\n=== 调用 list_memories ===")
            r = await session.call_tool("list_memories", {})
            print(f"  记忆库: {r.content[0].text}")

            print("\n=== 调用 save_memory ===")
            r = await session.call_tool("save_memory",
                                        {"key": "MCP协议验证", "value": "标准客户端工具调用成功"})
            print(f"  {r.content[0].text}")

            print("\nMCP 协议端到端验证通过 ✅")


if __name__ == "__main__":
    anyio.run(main)
