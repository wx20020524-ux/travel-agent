"""
行程规划总控 Agent —— 持有子 Agent 作为工具，统一编排并流式输出最终行程。

v2: 集成长期记忆管理
  - LangGraph Memory (MemorySaver checkpointer)
  - 用户偏好持久化 + 动态注入 system prompt
  - 上下文窗口裁剪
  - 历史对话自动总结

v3: 集成 LangSmith 追踪
  - 自动采集 LLM 调用 / Tool 执行 / Agent 节点
  - Trace 级元数据注入 (user_id, session_id, model)
  - LangSmith Dashboard 可视化: https://smith.langchain.com/
"""
import uuid
from typing import AsyncIterator, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from mcptools.mcp_client import McpClientManager
from rag.rag_engine import get_rag_engine
from agents.specialist import SpecialistAgent
from monitor.trace import new_trace, current_trace, end_trace, Span
from monitor.langsmith_ import (
    is_langsmith_available,
    get_run_tree,
    trace_metadata,
)
from agents.prompts import (
    HOTEL_AGENT_PROMPT,
    ATTRACTION_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
    build_planner_prompt,
)

from config import CONFIG
from memory.store import UserProfileStore
from memory.summarizer import ConversationSummarizer
from memory.context import ContextManager

# 工具名 -> 用户友好的中文标签（纯 ASCII/中文，兼容 Windows GBK）
TOOL_LABELS = {
    "query_knowledge": ("[RAG]", "检索本地知识库"),
    "query_weather":     ("[Weather]", "查询天气"),
    "search_hotel":      ("[Hotel]", "搜索酒店"),
    "search_attraction": ("[Attraction]", "搜索景点"),
    "maps_direction_walking":            ("[Walk]", "规划步行路线"),
    "maps_direction_driving":            ("[Drive]", "规划驾车路线"),
    "maps_direction_transit_integrated": ("[Transit]", "规划公交路线"),
}

class TripPlanner:
    """
    旅行规划总控智能体（v2: 长期记忆集成版）。

    架构:
      Planner (总控)
        ├── query_knowledge  -> 本地 RAG 知识库（攻略/美食/住宿/贴士）
        ├── search_hotel      -> HotelAgent      -> MCP: maps_text_search
        ├── search_attraction -> AttractionAgent -> MCP: maps_text_search
        ├── query_weather     -> WeatherAgent    -> MCP: maps_weather
        └── maps_direction_*  -> 直接 MCP 路线工具

    Memory 架构:
      ┌─ UserProfileStore (SQLite) ─┐
      │  用户偏好 JSON              │
      │  对话历史                    │
      │  会话摘要                    │
      └─────────────────────────────┘
              ↓ load / save
      ┌─ MemorySaver (LangGraph) ───┐
      │  Agent 状态持久化            │
      └─────────────────────────────┘
              ↓
      ┌─ ContextManager ────────────┐
      │  Token 估算 + 消息裁剪       │
      └─────────────────────────────┘

    用法:
        # 无记忆（向后兼容）
        planner = TripPlanner(llm)
        await planner.build()
        async for token in planner.stream("杭州3日游..."):
            print(token, end="")

        # 带记忆
        planner = TripPlanner(llm)
        async for token in planner.stream("杭州3日游...", user_id="user_001"):
            print(token, end="")
    """

    def __init__(
        self,
        llm: BaseChatModel,
        store: Optional[UserProfileStore] = None,
        summarizer: Optional[ConversationSummarizer] = None,
        context_mgr: Optional[ContextManager] = None,
    ):
        self.llm = llm
        self.mcp = McpClientManager()
        self.rag = get_rag_engine()

        # ---- 长期记忆组件 ----
        self._store = store
        self._summarizer = summarizer
        self._context_mgr = context_mgr or ContextManager(
            max_tokens=CONFIG.memory_max_input_tokens
        )

        # ---- 子 Agent（build 时初始化）----
        self._hotel_agent: SpecialistAgent | None = None
        self._attraction_agent: SpecialistAgent | None = None
        self._weather_agent: SpecialistAgent | None = None

        # ---- 顶层 Planner Agent ----
        self._agent = None
        self._checkpointer: Optional[MemorySaver] = None
        self._built_with_prefs: str = ""  # 用于判断是否需要重建

    # ==================== 懒加载 memory 组件 ====================

    @property
    def store(self) -> UserProfileStore:
        if self._store is None:
            self._store = UserProfileStore(CONFIG.memory_db_path)
        return self._store

    @property
    def summarizer(self) -> ConversationSummarizer:
        if self._summarizer is None:
            self._summarizer = ConversationSummarizer(self.llm)
        return self._summarizer

    # ==================== 构建 ====================

    async def build(
        self,
        user_preferences_text: str = "",
        *,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化所有子 Agent + 组装 Planner。

        Args:
            user_preferences_text: 用户长期偏好文本（来自 UserPreferences.to_prompt_fragment()），
                                   为空时使用默认 Prompt。
        """
        # 如果 preferences 没变且已构建，直接复用
        effective_system_prompt = system_prompt or build_planner_prompt(user_preferences_text)
        if self._agent is not None and self._built_with_prefs == effective_system_prompt:
            return

        # preferences 变了，强制重建
        self._agent = None

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

        # 4. 组装 Planner
        all_tools = [
            query_knowledge,
            search_hotel, search_attraction, query_weather,
            *route_tools,
        ]

        # 动态构建 system prompt（注入用户偏好）
        # 创建 MemorySaver checkpointer（LangGraph 状态持久化）
        if self._checkpointer is None:
            self._checkpointer = MemorySaver()

        self._agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=effective_system_prompt,
            checkpointer=self._checkpointer,
        )

        self._built_with_prefs = effective_system_prompt
        if user_preferences_text or system_prompt:
            print(f"[Planner] 已注入用户长期偏好，Agent 重建完成")

    # ==================== 用户上下文加载 ====================

    def _load_user_context(self, user_id: str) -> tuple[str, list[str]]:
        """
        加载用户长期记忆上下文。

        Returns:
            (preferences_text, history_summaries): 偏好 Prompt 文本 + 历史摘要列表
        """
        try:
            prefs = self.store.get_preferences(user_id)
            prefs_text = prefs.to_prompt_fragment()

            summaries = self.store.get_summaries(user_id, limit=3)
            summary_texts = [s["summary"] for s in summaries if s.get("summary")]

            return prefs_text, summary_texts
        except Exception as e:
            print(f"[Planner] 加载用户记忆失败: {e}")
            return "", []

    def _prepare_agent_messages(
        self,
        user_input: str,
        user_preferences_text: str,
        history_summaries: list[str],
    ) -> list[dict[str, str]]:
        base_system_prompt = build_planner_prompt(user_preferences_text)
        prepared_messages, _ = self._context_mgr.prepare(
            base_system_prompt,
            [{"role": "user", "content": user_input}],
            history_summaries,
        )
        return prepared_messages

    # ==================== 非流式调用 ====================

    async def invoke(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """输入自然语言需求，返回完整旅行计划（支持记忆）"""
        prefs_text = ""
        history_summaries = []
        if user_id:
            prefs_text, history_summaries = self._load_user_context(user_id)
        prepared_messages = self._prepare_agent_messages(
            user_input, prefs_text, history_summaries
        )
        await self.build(system_prompt=prepared_messages[0]["content"])

        thread_id = session_id or str(uuid.uuid4())
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": thread_id},
        }

        result = await self._agent.ainvoke(
            {"messages": prepared_messages[1:]},
            config=config if config else None,
        )
        output = result["messages"][-1].content

        # 保存对话历史
        if user_id:
            self._save_conversation(user_id, thread_id, user_input, output)

        return output

    # ==================== 流式调用 ====================

    async def stream(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        流式输出旅行计划（带 SSL/网络错误重试 + 链路追踪 + 长期记忆）。

        Args:
            user_input: 用户输入
            user_id:     用户标识（传 None 则禁用记忆功能）
            session_id:  会话标识（传 None 则自动生成）

        Yields:
            逐 token 文本 + 工具调用状态标记
        """
        import asyncio

        max_retries = 3
        retry_delay = 2
        trace = new_trace(user_input)
        _active_spans: dict[str, Span] = {}

        # ---- 加载用户记忆 ----
        prefs_text = ""
        history_summaries = []
        sid = session_id or str(uuid.uuid4())
        if user_id:
            prefs_text, history_summaries = self._load_user_context(user_id)

        # ---- LangSmith 追踪元数据 ----
        _ls_metadata = trace_metadata(user_id, sid)

        # 收集完整输出（用于后续保存）
        output_buffer: list[str] = []

        # ---- LangSmith Root Run ----
        ls_run = None
        if is_langsmith_available():
            ls_run = get_run_tree(
                "Planner.trip_plan",
                metadata=_ls_metadata,
            )

        try:
            for attempt in range(1, max_retries + 1):
                try:
                    prepared_messages = self._prepare_agent_messages(
                        user_input, prefs_text, history_summaries
                    )
                    await self.build(system_prompt=prepared_messages[0]["content"])

                    # LangGraph 配置（checkpointer 必须传 thread_id）
                    agent_config: dict = {
                        "configurable": {"thread_id": sid},
                        "recursion_limit": 50,  # 复杂旅行规划需要更多工具调用步数
                    }
                    # 注入 LangSmith metadata（LangChain 自动上报到 trace）
                    if is_langsmith_available():
                        agent_config["metadata"] = _ls_metadata

                    async for event in self._agent.astream_events(
                        {"messages": prepared_messages[1:]},
                        version="v2",
                        config=agent_config,
                    ):
                        kind = event.get("event", "")

                        if kind == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                output_buffer.append(content)
                                yield content

                        elif kind == "on_tool_start":
                            name = event.get("name", "unknown")
                            run_id = event.get("run_id", "")
                            emoji, label = TOOL_LABELS.get(name, ("[*]", name))
                            yield f"\n{emoji} {label}...\n"

                            span = trace.start_span(
                                name, metadata={"tool_label": label}
                            )
                            _active_spans[run_id] = span

                        elif kind == "on_tool_end":
                            name = event.get("name", "unknown")
                            run_id = event.get("run_id", "")
                            span = _active_spans.pop(run_id, None)
                            if span:
                                span.end(success=True)

                    # ---- 成功：保存对话 + 触发总结 ----
                    end_trace(trace, success=True)

                    if user_id:
                        full_output = "".join(output_buffer)
                        self._save_conversation(user_id, sid, user_input, full_output)
                        await self._maybe_summarize(user_id, sid)

                    # ---- LangSmith Run 完成 ----
                    if ls_run:
                        try:
                            ls_run.end(outputs={
                                "status": "success",
                                "output_chars": len("".join(output_buffer)),
                            })
                            ls_run.post()
                        except Exception:
                            pass

                    return

                except Exception as e:
                    # 结束所有未关闭的 span（标记为 error）
                    for sp in _active_spans.values():
                        if not sp._ended:
                            sp.end(success=False, error=str(e)[:200])
                    _active_spans.clear()

                    error_msg = str(e)
                    is_retryable = any(
                        keyword in error_msg.lower()
                        for keyword in [
                            "ssl", "eof", "connection", "timeout", "network", "reset",
                            "invalidparameter", "json format", "internalerror",
                        ]
                    )
                    # 也检查是否为 DashScope InvalidParameter 异常
                    if not is_retryable:
                        try:
                            from dashscope.common.error import InvalidParameter
                            if isinstance(e, InvalidParameter):
                                is_retryable = True
                        except ImportError:
                            pass

                    if attempt < max_retries and is_retryable:
                        yield f"\n[WARN] 网络连接问题（尝试 {attempt}/{max_retries}）：{type(e).__name__}\n"
                        yield f"   {retry_delay}秒后自动重试...\n"
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        self._agent = None
                        await self.mcp.close()
                    else:
                        if is_retryable:
                            yield f"\n[FAIL] API 调用失败（已重试 {max_retries} 次）\n"
                            yield f"   建议：稍后重试\n"
                        # ---- LangSmith Run 错误 ----
                        if ls_run:
                            try:
                                ls_run.end(error=error_msg[:500])
                                ls_run.post()
                            except Exception:
                                pass
                        end_trace(trace, success=False, error=error_msg)
                        raise
        finally:
            # 确保 LangSmith Run 一定被关闭
            if ls_run and hasattr(ls_run, 'end_time') and ls_run.end_time is None:
                try:
                    ls_run.end(error="unexpected termination")
                    ls_run.post()
                except Exception:
                    pass

    # ==================== 对话持久化 ====================

    def _save_conversation(
        self, user_id: str, session_id: str, user_input: str, assistant_output: str
    ):
        """保存一轮对话到 SQLite"""
        try:
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output[:5000]},  # 截断过长输出
            ]
            self.store.add_messages_batch(user_id, session_id, messages)
        except Exception as e:
            print(f"[Planner] 保存对话失败: {e}")

    async def _maybe_summarize(self, user_id: str, session_id: str):
        """
        检查是否需要触发对话总结。
        当对话的 token 估算超过阈值时，调用 LLM 总结并提取偏好。
        """
        try:
            messages = self.store.get_messages(user_id, session_id, limit=100)
            if not messages:
                return

            # 估算 token 数
            total_tokens = self._context_mgr.estimate_messages_tokens(messages)
            if total_tokens < CONFIG.memory_summarize_threshold:
                return  # 未达阈值，不总结

            print(f"[Planner] 触发对话总结: user={user_id}, session={session_id}, "
                  f"messages={len(messages)}, tokens~{total_tokens}")

            result = await self.summarizer.summarize(messages)

            if result.summary_text:
                # 保存总结
                self.store.save_summary(
                    user_id, session_id,
                    result.summary_text,
                    result.preferences.to_dict(),
                    len(messages),
                )
                # 更新用户偏好
                self.store.update_preferences(user_id, result.preferences)
                # 标记消息已总结
                message_ids = [m["id"] for m in messages if "id" in m]
                if message_ids:
                    self.store.mark_summarized(message_ids)

                print(f"[Planner] 对话总结完成: 偏好={result.preferences.to_dict()}")

        except Exception as e:
            print(f"[Planner] 对话总结失败: {e}")
