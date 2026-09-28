"""
MCP 客户端管理器 —— 单例模式，全局共享高德地图 MCP 连接。
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from config import CONFIG


class McpClientManager:
    """
    高德地图 MCP 客户端单例。
    这是单例模式
    保证整个程序永远只有一个 McpClientManager
    保证缓存只创建一次，不重复请求
    保证客户端不重复初始化

    职责：
      1. 管理与阿里百炼 MCP 服务器的连接
      2. 按领域（poi/weather/route）分发工具子集
      3. 缓存已加载工具，避免重复请求

    用法：
      manager = McpClientManager()
      poi_tools = await manager.get_tools_for("poi")
      route_tools = await manager.get_tools_for("route")
    """

    _instance: "McpClientManager | None" = None

    def __new__(cls) -> "McpClientManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._client: MultiServerMCPClient | None = None
        self._tools_cache: dict[str, list[BaseTool]] = {}
        self._initialized = True

    # ==================== 连接管理 ====================

    async def _get_client(self) -> MultiServerMCPClient:
        """懒加载 MCP 客户端"""
        if self._client is None:
            self._client = MultiServerMCPClient({
                "amap-server": {
                    "transport": CONFIG.mcp_transport,
                    "url": CONFIG.mcp_url,
                    "headers": {
                        "Authorization": f"Bearer {CONFIG.api_key}",
                        "Content-Type": "application/json"
                    }
                }
            })
        return self._client

    # ==================== 工具获取 ====================

    async def get_all_tools(self) -> list[BaseTool]:
        """获取 MCP 服务器暴露的全部工具，带重试机制"""
        if "all" not in self._tools_cache:
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    client = await self._get_client()
                    print(f"正在连接 MCP 服务器 (尝试 {attempt + 1}/{max_retries})...")
                    self._tools_cache["all"] = await client.get_tools()
                    print(f"成功获取 {len(self._tools_cache['all'])} 个工具")
                    return self._tools_cache["all"]
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"连接失败: {e}，{retry_delay}秒后重试...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise RuntimeError(
                            f"无法连接到 MCP 服务器: {e}\n"
                            f"请检查：\n"
                            f"1. API Key 是否正确\n"
                            f"2. 网络连接是否正常\n"
                            f"3. 阿里百炼服务是否可用"
                        ) from e
        
        return self._tools_cache["all"]

    async def get_tools_for(self, domain: str) -> list[BaseTool]:
        """按领域获取工具子集"""
        all_tools = await self.get_all_tools()
        target_names = set(CONFIG.tool_domains.get(domain, []))
        return [t for t in all_tools if t.name in target_names]

    # ==================== 生命周期 ====================

    async def close(self):
        """关闭 MCP 连接（如有需要）"""
        self._client = None
        self._tools_cache.clear()

    @classmethod
    def reset(cls):
        """重置单例（测试用）"""
        cls._instance = None
