#!/usr/bin/env python3
"""调试 MCP 连接问题"""
import asyncio
from config import CONFIG
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_mcp_connection():
    """测试 MCP 连接"""
    print("=" * 60)
    print("MCP 连接调试")
    print("=" * 60)
    
    print(f"\nAPI Key: {CONFIG.api_key[:10]}...")
    print(f"Transport: {CONFIG.mcp_transport}")
    print(f"URL: {CONFIG.mcp_url}")
    
    try:
        print("\n正在创建 MCP 客户端...")
        client = MultiServerMCPClient({
            "amap-server": {
                "transport": CONFIG.mcp_transport,
                "url": CONFIG.mcp_url,
                "headers": {
                    "Authorization": f"Bearer {CONFIG.api_key}"
                }
            }
        })
        
        print("正在获取工具列表...")
        tools = await client.get_tools()
        print(f"\n✓ 成功获取 {len(tools)} 个工具:")
        for i, tool in enumerate(tools[:10], 1):
            print(f"  {i}. {tool.name}")
        if len(tools) > 10:
            print(f"  ... 还有 {len(tools) - 10} 个工具")
        
    except Exception as e:
        print(f"\n✗ 错误: {type(e).__name__}: {e}")
        import traceback
        print("\n详细堆栈:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
