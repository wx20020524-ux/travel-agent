"""
行程规划总控 Agent —— 持有子 Agent 作为工具，统一编排并流式输出最终行程。
"""
import re
from typing import AsyncIterator
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel

from mcp_client import McpClientManager
from rag_engine import get_rag_engine
from agents.specialist import SpecialistAgent
from prompts import (
    HOTEL_AGENT_PROMPT,
    ATTRACTION_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
)

# 工具名 → 用户友好的中文标签
TOOL_LABELS = {
    "query_knowledge": ("📚", "检索本地知识库"),
    "query_weather":     ("🌤️", "查询天气"),
    "search_hotel":      ("🏨", "搜索酒店"),
    "search_attraction": ("🏛️", "搜索景点"),
    "maps_direction_walking_by_address":             ("🚶", "规划步行路线"),
    "maps_direction_driving_by_address":             ("🚗", "规划驾车路线"),
    "maps_direction_transit_integrated_by_address":  ("🚌", "规划公交路线"),
}

# 匹配子 Agent 内部泄漏的 [TOOL_CALL:...] 模式
_TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:[^\]]*\]")


class TripPlanner:
    """
    旅行规划总控智能体。

    架构:
      Planner (总控)
        ├── query_knowledge  → 本地 RAG 知识库（攻略/美食/住宿/贴士）
        ├── search_hotel      → HotelAgent      → MCP: maps_text_search
        ├── search_attraction → AttractionAgent → MCP: maps_text_search
        ├── query_weather     → WeatherAgent    → MCP: maps_weather
        └── maps_direction_*  → 直接 MCP 路线工具

    用法:
        planner = TripPlanner(llm)
        await planner.build()

        # 非流式
        result = await planner.invoke("杭州3日游...")

        # 流式（逐 token 输出到控制台）
        async for token in planner.stream("杭州3日游..."):
            print(token, end="", flush=True)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.mcp = McpClientManager()
        self.rag = get_rag_engine()

        # 子 Agent（build 时初始化）
        self._hotel_agent: SpecialistAgent | None = None
        self._attraction_agent: SpecialistAgent | None = None
        self._weather_agent: SpecialistAgent | None = None

        # 顶层 Planner Agent
        self._agent = None

    # ==================== 构建 ====================

    async def build(self):
        """初始化所有子 Agent + 组装 Planner"""
        if self._agent is not None:
            return

        # 0. 初始化 RAG 知识库
        try:
            rag_ok = await self.rag.build(force_rebuild=False)
            if rag_ok:
                stats = self.rag.get_stats()
                print(f"[Planner] RAG 就绪: docs={stats['docs_count']}, chunks={stats['chunks_count']}")
            else:
                print("[Planner] RAG 初始化失败，将不启用知识库检索")
        except Exception as e:
            print(f"[Planner] RAG 初始化异常（不影响其他功能）: {e}")

        # 1. 按领域加载 MCP 工具
        poi_tools = await self.mcp.get_tools_for("poi")
        weather_tools = await self.mcp.get_tools_for("weather")
        route_tools = await self.mcp.get_tools_for("route")

        # 2. 创建子 Agent
        self._hotel_agent = SpecialistAgent(
            self.llm, "HotelAgent", HOTEL_AGENT_PROMPT, poi_tools
        )
        self._attraction_agent = SpecialistAgent(
            self.llm, "AttractionAgent", ATTRACTION_AGENT_PROMPT, poi_tools
        )
        self._weather_agent = SpecialistAgent(
            self.llm, "WeatherAgent", WEATHER_AGENT_PROMPT, weather_tools
        )
        await self._hotel_agent.build()
        await self._attraction_agent.build()
        await self._weather_agent.build()

        # 3. 将子 Agent 包装为 Tool
        @tool
        def query_knowledge(query: str) -> str:
            """
            检索本地旅行知识库。
            输入自然语言查询（如"杭州西湖怎么玩"、"成都必吃美食"），
            返回知识库中相关的攻略、美食推荐、住宿建议、门票价格、开放时间、贴士等内容。
            若知识库无内容会返回提示，再调用外部工具。
            """
            return self.rag.search_knowledge_formatted(query)

        @tool
        async def search_hotel(query: str) -> str:
            """搜索酒店。输入城市+偏好，返回酒店列表。"""
            return await self._hotel_agent.invoke(query)

        @tool
        async def search_attraction(query: str) -> str:
            """搜索景点。输入城市+类型偏好，返回景点列表。"""
            return await self._attraction_agent.invoke(query)

        @tool
        async def query_weather(query: str) -> str:
            """查询天气。输入城市+日期，返回天气概况。"""
            return await self._weather_agent.invoke(query)

        # 4. 组装 Planner：知识库工具 + 子 Agent 工具 + 路线 MCP 工具
        all_tools = [
            query_knowledge,
            search_hotel, search_attraction, query_weather,
            *route_tools,
        ]

        self._agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=PLANNER_AGENT_PROMPT,
        )

    # ==================== 非流式调用 ====================

    async def invoke(self, user_input: str) -> str:
        """输入自然语言需求，返回完整旅行计划"""
        await self.build()
        result = await self._agent.ainvoke({
            "messages": [{"role": "user", "content": user_input}]
        })
        return result["messages"][-1].content

    # ==================== 流式调用 ====================

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """
        流式输出旅行计划（带 SSL/网络错误重试）。

        逐 token yield 文本 + 工具调用状态标记:
          - 普通 token: 直接 yield（过滤掉子 Agent 内部 TOOL_CALL 泄漏）
          - 工具开始:   yield "\\n[调用: tool_name]\\n"
          - 工具结束:   yield "\\n[完成: tool_name]\\n"
        """
        import asyncio
        import time
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                await self.build()

                async for event in self._agent.astream_events(
                    {"messages": [{"role": "user", "content": user_input}]},
                    version="v2",
                ):
                    kind = event.get("event", "")

                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            # 过滤子 Agent 内部 TOOL_CALL 格式泄漏
                            content = _TOOL_CALL_PATTERN.sub("", content)
                            if content.strip():
                                yield content

                    elif kind == "on_tool_start":
                        name = event.get("name", "unknown")
                        emoji, label = TOOL_LABELS.get(name, ("🔧", name))
                        yield f"\n{emoji} {label}...\n"

                    elif kind == "on_tool_end":
                        name = event.get("name", "unknown")
                        if name in TOOL_LABELS:
                            pass  # 静默处理，避免刷屏
                
                return
                
            except Exception as e:
                error_msg = str(e)
                is_network_error = any(
                    keyword in error_msg.lower()
                    for keyword in ["ssl", "eof", "connection", "timeout", "network", "reset"]
                )
                
                if attempt < max_retries and is_network_error:
                    yield f"\n⚠️ 网络连接问题（尝试 {attempt}/{max_retries}）：{type(e).__name__}\n"
                    yield f"   {retry_delay}秒后自动重试...\n"
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    # 重置 agent 状态以便重新连接
                    self._agent = None
                    self.mcp.close()
                else:
                    if is_network_error:
                        yield f"\n❌ 网络连接失败（已重试 {max_retries} 次）\n"
                        yield f"   建议：检查网络连接，或稍后重试\n"
                    raise
