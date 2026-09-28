"""
专业子 Agent —— POI 搜索 / 天气查询 / 酒店推荐。
每个实例封装一个独立的 LangGraph Agent，只持有自己领域的 MCP 工具。
"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from monitor.trace import current_trace, Span


class SpecialistAgent:
    """
    领域专家智能体。

    用法:
        agent = SpecialistAgent(llm, "POI搜索专家", system_prompt, tools)
        await agent.build()
        result = await agent.invoke("搜索北京故宫")
    """

    def __init__(
        self,
        llm: BaseChatModel,
        name: str,
        system_prompt: str,
        tools: list[BaseTool],
    ):
        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self._agent = None

    async def build(self):
        """构建底层 LangGraph Agent"""
        if self._agent is None:
            self._agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )

    async def invoke(self, user_input: str) -> str:
        """非流式调用（带链路追踪 span）"""
        await self.build()
        trace = current_trace()
        span: Span | None = None
        try:
            if trace:
                span = trace.start_span(
                    f"Specialist.{self.name}",
                    metadata={"agent_type": self.name, "query": (user_input or "")[:100]},
                )
            result = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            if span:
                span.end(success=True)
            return result["messages"][-1].content
        except Exception as e:
            if span and not span._ended:
                span.end(success=False, error=str(e)[:200])
            raise

    async def stream(self, user_input: str):
        """
        流式调用，逐 token yield。

        用于内部被 Planner 调用时不需要流式，但保留能力。
        """
        await self.build()
        async for event in self._agent.astream_events(
            {"messages": [{"role": "user", "content": user_input}]},
            version="v2",
        ):
            if event.get("event") == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
