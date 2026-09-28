import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")

import agents.planner as planner_module
import config as config_module
from agents.planner import TOOL_LABELS, TripPlanner
from agents.prompts import PLANNER_AGENT_PROMPT_TEMPLATE
import rag.rag_engine as rag_engine_module


def _response(content, tool_calls=None):
    return {
        "output": {
            "choices": [
                {"message": {"content": content, "tool_calls": tool_calls or []}}
            ]
        }
    }


def test_subtract_handles_none_and_mismatched_tool_calls():
    current = _response(
        None,
        [{"function": {"name": "maps_weather", "arguments": '{"city": "北京"}'}}],
    )
    previous = _response(None, [])
    result = config_module._patched_subtract(None, current, previous)

    assert result["output"]["choices"][0]["message"]["content"] is None
    assert result["output"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "maps_weather"

    current = _response(
        "Hello world",
        [{"function": {}}],
    )
    previous = _response("Hello", [])
    result = config_module._patched_subtract(None, current, previous)

    assert result["output"]["choices"][0]["message"]["content"] == " world"


def test_create_llm_uses_dashscope_compatible_endpoint():
    captured = {}

    class FakeChatTongyi:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    original = config_module.ChatOpenAI
    config_module.ChatOpenAI = FakeChatTongyi
    try:
        config = config_module.Config(api_key="test-key")
        config.create_llm()
    finally:
        config_module.ChatOpenAI = original

    assert captured == {
        "model": config.model_name,
        "api_key": "test-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": config.temperature,
        "streaming": True,
        "timeout": config.request_timeout,
        "max_retries": config.max_retries,
    }
    assert "http_client" not in captured
    assert not hasattr(config_module, "_create_robust_http_client")


def test_route_tool_names_match_amap_mcp():
    expected = {
        "maps_direction_walking",
        "maps_direction_driving",
        "maps_direction_transit_integrated",
    }
    config = config_module.Config(api_key="test-key")

    assert set(config.tool_domains["route"]) == expected
    assert expected.issubset(TOOL_LABELS)
    assert all(name in PLANNER_AGENT_PROMPT_TEMPLATE for name in expected)
    assert "_by_address" not in PLANNER_AGENT_PROMPT_TEMPLATE


def test_stream_preserves_whitespace_tokens():
    class FakeAgent:
        async def astream_events(self, *_args, **_kwargs):
            for content in ("Hello", " ", "world", "\n"):
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": type("Chunk", (), {"content": content})()},
                }

    planner = object.__new__(TripPlanner)
    planner._prepare_agent_messages = (
        lambda *_args, **_kwargs: [{"content": "system"}, {"content": "user"}]
    )

    async def fake_build(*_args, **_kwargs):
        return None

    planner.build = fake_build
    planner._agent = FakeAgent()

    originals = {
        "new_trace": planner_module.new_trace,
        "end_trace": planner_module.end_trace,
        "trace_metadata": planner_module.trace_metadata,
        "is_langsmith_available": planner_module.is_langsmith_available,
    }
    planner_module.new_trace = lambda *_args, **_kwargs: object()
    planner_module.end_trace = lambda *_args, **_kwargs: None
    planner_module.trace_metadata = lambda *_args, **_kwargs: {}
    planner_module.is_langsmith_available = lambda: False

    async def collect_tokens():
        return [token async for token in planner.stream("test")]

    try:
        tokens = asyncio.run(collect_tokens())
    finally:
        planner_module.new_trace = originals["new_trace"]
        planner_module.end_trace = originals["end_trace"]
        planner_module.trace_metadata = originals["trace_metadata"]
        planner_module.is_langsmith_available = originals["is_langsmith_available"]

    assert tokens == ["Hello", " ", "world", "\n"]


def test_planner_is_session_scoped_in_streamlit_app():
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
    )
    with open(app_path, encoding="utf-8") as app_file:
        app_source = app_file.read()

    assert "@st.cache_resource" not in app_source
    assert "st.session_state.planner" in app_source


def test_qwen_uses_environment_api_key():
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", "test_qwen.py"
    )
    with open(test_path, encoding="utf-8") as test_file:
        test_source = test_file.read()

    assert 'os.getenv("DASHSCOPE_API_KEY")' in test_source
    assert "os.getenv(\"sk-" not in test_source


def test_test_scripts_use_package_imports():
    test_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(test_dir, "test_mcp_connection.py"), encoding="utf-8") as f:
        assert "from mcptools.mcp_client import McpClientManager" in f.read()
    with open(os.path.join(test_dir, "test_rag.py"), encoding="utf-8") as f:
        assert "from rag.rag_engine import get_rag_engine" in f.read()


def test_rag_force_rebuild_deletes_existing_collection():
    import shutil
    import tempfile
    import langchain_community.vectorstores as vectorstores_module
    from rag.rag_engine import RagEngine

    calls = []

    class FakeChroma:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def delete_collection(self):
            calls.append(("delete", None))

        @classmethod
        def from_documents(cls, *_args, **_kwargs):
            return cls()

    temp_dir = tempfile.mkdtemp(
        prefix="rag-rebuild-test-", dir=os.path.dirname(os.path.abspath(__file__))
    )
    original_chroma = vectorstores_module.Chroma
    original_vectorstore_dir = config_module.CONFIG.rag_vectorstore_dir
    original_create_embeddings = config_module.CONFIG.create_embeddings

    vectorstores_module.Chroma = FakeChroma
    config_module.CONFIG.rag_vectorstore_dir = temp_dir
    config_module.CONFIG.create_embeddings = lambda: object()
    try:
        engine = RagEngine()
        assert asyncio.run(engine.build(force_rebuild=True)) is True
    finally:
        vectorstores_module.Chroma = original_chroma
        config_module.CONFIG.rag_vectorstore_dir = original_vectorstore_dir
        config_module.CONFIG.create_embeddings = original_create_embeddings
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert ("delete", None) in calls
    assert any(name == "init" for name, _ in calls)


async def verify_route_tools_live():
    from mcptools.mcp_client import McpClientManager

    manager = McpClientManager()
    try:
        tools = await manager.get_tools_for("route")
    finally:
        await manager.close()
    return {tool.name for tool in tools}


if __name__ == "__main__":
    test_subtract_handles_none_and_mismatched_tool_calls()
    test_create_llm_uses_dashscope_compatible_endpoint()
    test_route_tool_names_match_amap_mcp()
    test_stream_preserves_whitespace_tokens()
    test_planner_is_session_scoped_in_streamlit_app()
    test_qwen_uses_environment_api_key()
    test_test_scripts_use_package_imports()
    test_rag_force_rebuild_deletes_existing_collection()
    print("[PASS] bugfix regression tests")
    if "--live-route" in sys.argv:
        route_tools = asyncio.run(verify_route_tools_live())
        assert route_tools == {
            "maps_direction_walking",
            "maps_direction_driving",
            "maps_direction_transit_integrated",
        }
        print(f"[PASS] live route tools: {sorted(route_tools)}")
