"""
LangSmith 集成 —— Agent 执行链路追踪 & 可视化。

核心能力:
  1. 启用 LangSmith Tracing: 自动采集 LLM 调用 / Tool 执行 / Agent 节点
  2. 自定义 Metadata: 注入 user_id, session_id 到每条 Trace
  3. Run Tree 管理: 对关键节点（Planner / RAG / Specialist）创建子 Run
  4. 优雅降级: 未配置 API Key 时静默跳过，不影响主流程

启用方式:
  方式 A（推荐）: 在 .env 中设置
    LANGSMITH_ENABLED=true
    LANGCHAIN_API_KEY=lsv2_pt_xxxx
    LANGCHAIN_PROJECT=travel-agent

  方式 B: 代码中调用
    from monitor.langsmith_ import init_langsmith
    init_langsmith(api_key="lsv2_pt_xxxx", project="travel-agent")

LangSmith Dashboard: https://smith.langchain.com/
"""

import os
import functools
from typing import Any, Callable, Dict, Optional

from config import CONFIG


# 全局状态
_langsmith_available: Optional[bool] = None


def is_langsmith_available() -> bool:
    """检查 LangSmith 是否可用（SDK 已安装 + API Key 已配置）"""
    global _langsmith_available
    if _langsmith_available is not None:
        return _langsmith_available

    api_key = CONFIG.langsmith_api_key or os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        _langsmith_available = False
        return False

    try:
        import langsmith
        _langsmith_available = True
    except ImportError:
        print("[LangSmith] langsmith SDK 未安装，跳过。安装: pip install langsmith")
        _langsmith_available = False

    return _langsmith_available


def init_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """
    初始化 LangSmith 追踪。

    设置环境变量以启用 LangChain 自动 Tracing。
    所有 LangChain / LangGraph 调用会自动上报到 LangSmith。

    Args:
        api_key:  LangSmith API Key（不传则从环境变量读取）
        project:  项目名称（不传则用 config 中的默认值）
        endpoint: API 端点（SaaS 用户无需设置）

    Returns:
        True 表示已启用，False 表示跳过
    """
    key = api_key or CONFIG.langsmith_api_key or os.getenv("LANGCHAIN_API_KEY", "")
    if not key:
        print("[LangSmith] 未配置 LANGCHAIN_API_KEY，跳过 Tracing")
        return False

    # 检查 SDK
    try:
        import langsmith  # noqa: F401
    except ImportError:
        print("[LangSmith] langsmith SDK 未安装，跳过。安装: pip install langsmith")
        return False

    # 设置环境变量（LangChain 自动读取）
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = project or CONFIG.langsmith_project
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    elif CONFIG.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = CONFIG.langsmith_endpoint

    print(f"[LangSmith] Tracing 已启用 | project={os.environ['LANGCHAIN_PROJECT']}")
    return True


def get_run_tree(name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    创建 LangSmith Run Tree（用于自定义子 Run 层级）。

    在 Planner / RAG / Specialist 等关键节点创建子 Run，
    可在 LangSmith Dashboard 中看到完整的调用层级树。

    Args:
        name:     Run 名称，如 "Planner.root", "RAG.search", "Specialist.HotelAgent"
        metadata: 自定义元数据，如 {"user_id": "user_001", "session_id": "abc123"}

    Returns:
        LangSmith RunTree 对象，或 None（LangSmith 不可用时）

    用法:
        run = get_run_tree("Planner.root", {"user_id": user_id})
        if run:
            with run:
                # ... 执行 Agent 逻辑 ...
                run.end(outputs={"plan_length": 500})
    """
    if not is_langsmith_available():
        return None

    try:
        from langsmith import run_helpers
        # 获取当前活跃的 RunTree（由 LangChain callback 自动创建）
        parent = run_helpers.get_current_run_tree()
        return run_helpers.RunTree(
            name=name,
            run_type="chain",
            inputs={},
            extra=metadata or {},
            parent_run=parent,
        )
    except Exception as e:
        print(f"[LangSmith] 创建 RunTree 失败: {e}")
        return None


def trace_metadata(user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    构建 Trace 级别的 Metadata，注入到每次调用。

    Args:
        user_id:    用户标识
        session_id: 会话标识
        extra:      额外自定义字段

    Returns:
        metadata 字典
    """
    metadata = {
        "framework": "LangGraph",
        "model": CONFIG.model_name,
        "temperature": CONFIG.temperature,
    }
    if user_id:
        metadata["user_id"] = user_id
    if session_id:
        metadata["session_id"] = session_id
    if extra:
        metadata.update(extra)
    return metadata


def traceable(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    装饰器: 将函数标记为 LangSmith 可追踪。

    用法:
        @traceable(name="RAG.search", run_type="retriever")
        def search(query: str) -> str:
            ...
    """
    if not is_langsmith_available():
        return lambda f: f  # no-op 装饰器

    try:
        from langsmith import traceable as _traceable
        kwargs = {"run_type": run_type}
        if name:
            kwargs["name"] = name
        if metadata:
            kwargs["metadata"] = metadata
        return _traceable(**kwargs)
    except ImportError:
        return lambda f: f
