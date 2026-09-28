import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import TripPlanner
from config import CONFIG
from memory.context import ContextManager


def test_prepare_agent_messages_injects_history_summaries():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)

    messages = planner._prepare_agent_messages(
        '继续按照上次的偏好规划',
        '预算偏好: 豪华',
        ['用户喜欢成都美食和自然风光'],
    )

    assert messages[0]['role'] == 'system'
    assert '用户喜欢成都美食和自然风光' in messages[0]['content']
    assert '预算偏好: 豪华' in messages[0]['content']
    assert messages[-1] == {
        'role': 'user',
        'content': '继续按照上次的偏好规划',
    }


def test_stream_reuses_session_id_and_passes_user_message_only():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)

    captured = []
    built_prompts = []

    class FakeAgent:
        async def astream_events(self, inputs, version, config):
            captured.append((inputs, config))
            yield {
                'event': 'on_chat_model_stream',
                'data': {'chunk': SimpleNamespace(content='OK')},
            }

    async def fake_build(**kwargs):
        built_prompts.append(kwargs['system_prompt'])

    planner._agent = FakeAgent()
    planner.build = fake_build

    async def run_stream():
        outputs = []
        async for token in planner.stream(
            '帮我规划成都一日游',
            user_id=None,
            session_id='fixed-session',
        ):
            outputs.append(token)
        return outputs

    outputs = asyncio.run(run_stream())
    outputs = asyncio.run(run_stream())

    assert outputs == ['OK']
    assert len(captured) == 2
    assert all(config['configurable']['thread_id'] == 'fixed-session' for _, config in captured)
    assert all(inputs['messages'] == [{
        'role': 'user',
        'content': '帮我规划成都一日游',
    }] for inputs, _ in captured)
    assert built_prompts


def test_invoke_without_memory_passes_thread_id():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)
    captured = []

    class FakeAgent:
        async def ainvoke(self, inputs, config=None):
            captured.append((inputs, config))
            return {"messages": [SimpleNamespace(content="OK")]}

    async def fake_build(**kwargs):
        return None

    planner._agent = FakeAgent()
    planner.build = fake_build

    output = asyncio.run(planner.invoke(
        "帮我规划成都一日游",
        user_id=None,
        session_id="fixed-session",
    ))

    assert output == "OK"
    assert len(captured) == 1
    inputs, config = captured[0]
    assert inputs["messages"] == [{
        "role": "user",
        "content": "帮我规划成都一日游",
    }]
    assert config["configurable"]["thread_id"] == "fixed-session"


def test_invoke_without_memory_generates_thread_id():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)
    captured = []

    class FakeAgent:
        async def ainvoke(self, inputs, config=None):
            captured.append((inputs, config))
            return {"messages": [SimpleNamespace(content="OK")]}

    async def fake_build(**kwargs):
        return None

    planner._agent = FakeAgent()
    planner.build = fake_build

    output = asyncio.run(planner.invoke(
        "Plan a one-day trip",
        user_id=None,
        session_id=None,
    ))

    assert output == "OK"
    _, config = captured[0]
    assert config["configurable"]["thread_id"]


def test_stream_posts_langsmith_run_after_success():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)
    lifecycle_calls = []

    class FakeAgent:
        async def astream_events(self, inputs, version, config):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="OK")},
            }

    class FakeRun:
        end_time = None

        def end(self, **outputs):
            lifecycle_calls.append(("end", outputs))
            self.end_time = object()

        def post(self):
            lifecycle_calls.append(("post",))

    async def fake_build(**kwargs):
        return None

    planner._agent = FakeAgent()
    planner.build = fake_build

    import agents.planner as planner_module
    original_available = planner_module.is_langsmith_available
    original_get_run_tree = planner_module.get_run_tree
    planner_module.is_langsmith_available = lambda: True
    planner_module.get_run_tree = lambda *args, **kwargs: FakeRun()
    try:
        async def run_stream():
            outputs = []
            async for token in planner.stream(
                "Plan a one-day trip",
                user_id=None,
                session_id="fixed-session",
            ):
                outputs.append(token)
            return outputs

        assert asyncio.run(run_stream()) == ["OK"]
    finally:
        planner_module.is_langsmith_available = original_available
        planner_module.get_run_tree = original_get_run_tree

    assert lifecycle_calls[-2][0] == "end"
    assert lifecycle_calls[-1] == ("post",)


def test_maybe_summarize_runs_inside_active_event_loop():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)
    saved_summaries = []
    summarized_message_ids = []
    planner._store = SimpleNamespace(
        get_messages=lambda user_id, session_id, limit: [
            {"id": 1, "role": "user", "content": "user prefers food" * 100},
            {"id": 2, "role": "assistant", "content": "assistant response" * 100},
        ],
        save_summary=lambda *args, **kwargs: saved_summaries.append(args),
        update_preferences=lambda *args, **kwargs: None,
        mark_summarized=lambda message_ids: summarized_message_ids.append(message_ids),
    )

    async def summarize(messages):
        return SimpleNamespace(
            summary_text="user prefers food",
            preferences=SimpleNamespace(to_dict=lambda: {"interests": ["food"]}),
        )

    planner._summarizer = SimpleNamespace(summarize=summarize)

    original_threshold = CONFIG.memory_summarize_threshold
    CONFIG.memory_summarize_threshold = 1
    try:
        asyncio.run(planner._maybe_summarize("user-test", "session-test"))
    finally:
        CONFIG.memory_summarize_threshold = original_threshold

    assert saved_summaries
    assert summarized_message_ids == [[1, 2]]


def test_maybe_summarize_failure_does_not_mark_messages():
    planner = object.__new__(TripPlanner)
    planner._context_mgr = ContextManager(max_tokens=12000)
    saved_summaries = []
    summarized_message_ids = []
    planner._store = SimpleNamespace(
        get_messages=lambda user_id, session_id, limit: [
            {"id": 1, "role": "user", "content": "user prefers food" * 100},
            {"id": 2, "role": "assistant", "content": "assistant response" * 100},
        ],
        save_summary=lambda *args, **kwargs: saved_summaries.append(args),
        update_preferences=lambda *args, **kwargs: None,
        mark_summarized=lambda message_ids: summarized_message_ids.append(message_ids),
    )

    async def summarize(messages):
        raise RuntimeError("LLM unavailable")

    planner._summarizer = SimpleNamespace(summarize=summarize)

    original_threshold = CONFIG.memory_summarize_threshold
    CONFIG.memory_summarize_threshold = 1
    try:
        asyncio.run(planner._maybe_summarize("user-test", "session-test"))
    finally:
        CONFIG.memory_summarize_threshold = original_threshold

    assert saved_summaries == []
    assert summarized_message_ids == []


if __name__ == '__main__':
    test_prepare_agent_messages_injects_history_summaries()
    test_stream_reuses_session_id_and_passes_user_message_only()
    test_invoke_without_memory_passes_thread_id()
    test_invoke_without_memory_generates_thread_id()
    test_stream_posts_langsmith_run_after_success()
    test_maybe_summarize_runs_inside_active_event_loop()
    test_maybe_summarize_failure_does_not_mark_messages()
    print('[PASS] planner memory regression tests')
