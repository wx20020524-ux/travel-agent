"""
测试 LangSmith 集成 —— 模块导入、初始化、RunTree 创建、优雅降级
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_langsmith_config():
    """测试 config.py 中的 LangSmith 配置项"""
    print("=" * 60)
    print("[TEST] LangSmith 配置项")

    from config import CONFIG

    assert hasattr(CONFIG, "langsmith_enabled")
    assert hasattr(CONFIG, "langsmith_api_key")
    assert hasattr(CONFIG, "langsmith_project")
    assert hasattr(CONFIG, "langsmith_endpoint")
    print(f"  [OK] langsmith_enabled = {CONFIG.langsmith_enabled}")
    print(f"  [OK] langsmith_project = {CONFIG.langsmith_project}")
    print(f"  [OK] langsmith_endpoint = {CONFIG.langsmith_endpoint}")

    print("[PASS] LangSmith 配置项测试通过\n")


def test_langsmith_unavailable():
    """测试未配置 API Key 时的优雅降级"""
    print("=" * 60)
    print("[TEST] LangSmith 优雅降级（无 API Key）")

    from monitor.langsmith_ import (
        is_langsmith_available,
        init_langsmith,
        get_run_tree,
        trace_metadata,
        traceable,
    )

    # 1. is_langsmith_available
    available = is_langsmith_available()
    print(f"  [OK] is_langsmith_available() = {available}")
    # 未配置 API Key 时应返回 False
    if not available:
        print("  (预期行为: 无 API Key, 不启用 LangSmith)")

    # 2. init_langsmith 不传 key
    result = init_langsmith()
    print(f"  [OK] init_langsmith() = {result}")
    # 应该跳过
    assert result is False, "未配置 key 时应跳过"

    # 3. get_run_tree 应返回 None
    run = get_run_tree("test.run")
    assert run is None, "未配置 key 时 get_run_tree 应返回 None"
    print("  [OK] get_run_tree() 返回 None（优雅降级）")

    # 4. trace_metadata 应能正常工作（不依赖 LangSmith）
    meta = trace_metadata(user_id="user_001", session_id="sess_001")
    assert meta["user_id"] == "user_001"
    assert meta["session_id"] == "sess_001"
    assert meta["framework"] == "LangGraph"
    assert "model" in meta
    assert "temperature" in meta
    print(f"  [OK] trace_metadata() = {list(meta.keys())}")

    # 5. @traceable 装饰器应返回原函数（no-op）
    @traceable(name="test.fn", run_type="chain")
    def dummy_func(x):
        return x * 2

    result = dummy_func(5)
    assert result == 10
    print("  [OK] @traceable 装饰器在无 LangSmith 时为 no-op")

    print("[PASS] 优雅降级测试通过\n")


def test_langsmith_init_mocked():
    """测试 init_langsmith 的环境变量设置逻辑"""
    print("=" * 60)
    print("[TEST] init_langsmith 环境变量设置")

    from monitor.langsmith_ import init_langsmith

    # 测试传入 API Key 时的行为
    # 注意：这需要 langsmith SDK 安装，如果没有安装会返回 False
    result = init_langsmith(api_key="lsv2_pt_fake_key_for_test")
    
    # 如果 SDK 没安装，会返回 False
    try:
        import langsmith
        has_sdk = True
    except ImportError:
        has_sdk = False

    if has_sdk:
        print(f"  [OK] init_langsmith(with_key) = {result}")
        if result:
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_pt_fake_key_for_test"
            print("  [OK] 环境变量已正确设置")
            # 清理
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
    else:
        print("  [SKIP] langsmith SDK 未安装 (不影响功能)")

    print("[PASS] init_langsmith 测试通过\n")


def test_monitor_init_exports():
    """测试 monitor/__init__.py 导出 LangSmith 模块"""
    print("=" * 60)
    print("[TEST] monitor 模块导出")

    from monitor import (
        init_langsmith,
        is_langsmith_available,
        get_run_tree,
        trace_metadata,
        traceable,
        new_trace,
        end_trace,
        Span,
    )

    # 验证 LangSmith 导出
    assert callable(init_langsmith)
    assert callable(is_langsmith_available)
    assert callable(get_run_tree)
    assert callable(trace_metadata)
    assert callable(traceable)
    print("  [OK] LangSmith 函数从 monitor 正确导出")

    # 验证原有 trace 导出未受影响
    assert callable(new_trace)
    assert callable(end_trace)
    print("  [OK] 原有 trace 模块导出未受影响")

    print("[PASS] monitor 模块导出测试通过\n")


def test_planner_import_with_langsmith():
    """测试 planner.py 能正确导入 LangSmith 依赖"""
    print("=" * 60)
    print("[TEST] Planner 导入 LangSmith 依赖")

    # 直接测试导入路径
    from monitor.langsmith_ import (
        is_langsmith_available,
        get_run_tree,
        trace_metadata,
    )
    print("  [OK] planner 所需的 LangSmith 函数可导入")

    # 测试 trace_metadata 输出格式
    meta = trace_metadata("u1", "s1", extra={"custom": "value"})
    assert "framework" in meta
    assert "model" in meta
    assert "temperature" in meta
    assert meta["user_id"] == "u1"
    assert meta["session_id"] == "s1"
    assert meta["custom"] == "value"
    print(f"  [OK] trace_metadata 包含所有预期字段: {sorted(meta.keys())}")

    print("[PASS] Planner 导入测试通过\n")


def test_get_run_tree_uses_compatible_runtree_class():
    """测试当前 langsmith SDK 下 RunTree 可以创建（不触发网络请求）"""
    from config import CONFIG
    from monitor import langsmith_

    original_available = langsmith_._langsmith_available
    langsmith_._langsmith_available = True
    try:
        run = langsmith_.get_run_tree(
            "Planner.trip_plan",
            metadata={"user_id": "user_001", "session_id": "session_001"},
        )
        assert run is not None
        assert run.name == "Planner.trip_plan"
        assert run.session_name == CONFIG.langsmith_project
        assert run.extra["metadata"]["user_id"] == "user_001"
    finally:
        langsmith_._langsmith_available = original_available


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LangSmith 集成 — 单元测试")
    print("=" * 60 + "\n")

    test_langsmith_config()
    test_langsmith_unavailable()
    test_langsmith_init_mocked()
    test_monitor_init_exports()
    test_planner_import_with_langsmith()
    test_get_run_tree_uses_compatible_runtree_class()

    print("=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
