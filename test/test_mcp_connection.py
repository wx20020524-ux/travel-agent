#!/usr/bin/env python3
"""测试 MCP 连接"""
import asyncio
import sys

async def test_mcp_connection():
    """测试 MCP 连接"""
    print("=" * 60)
    print("高德地图 MCP 连接测试")
    print("=" * 60)
    
    try:
        from config import CONFIG
        from mcp_client import McpClientManager
        
        print("\n[1] 加载配置...")
        print(f"    API Key 前缀: {CONFIG.api_key[:15]}...")
        print(f"    MCP URL: {CONFIG.mcp_url}")
        
        print("\n[2] 创建 MCP 客户端...")
        manager = McpClientManager()
        
        print("\n[3] 尝试连接 MCP 服务...")
        print("    这可能需要几秒钟，请稍候...")
        
        tools = await manager.get_all_tools()
        
        print(f"\n[4] 连接成功！")
        print(f"    共获取到 {len(tools)} 个工具")
        
        print("\n可用的工具列表:")
        print("-" * 60)
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool.name}")
        
        print("\n" + "=" * 60)
        print("MCP 服务连接测试完成 - 成功！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[错误] 连接失败:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        
        print("\n" + "=" * 60)
        print("可能的原因:")
        print("  1. 未在阿里百炼控制台开通高德地图 MCP 服务")
        print("  2. API Key 权限不足")
        print("  3. 网络连接问题")
        print("  4. 阿里百炼服务暂时不可用")
        print("\n解决方法:")
        print("  请查看 MCP_SETUP_GUIDE.md 文件获取详细配置指南")
        print("=" * 60)
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_mcp_connection())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        sys.exit(1)
