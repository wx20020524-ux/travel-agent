import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    client = MultiServerMCPClient({
        "amap": {
            "transport": "http",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"
            }
        }
    })

    tools = await client.get_tools()

    print("工具数量：", len(tools))

    for t in tools:
        print(t.name)

asyncio.run(main())