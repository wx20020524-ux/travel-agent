"""
测试长期记忆管理模块 —— UserProfileStore + ConversationSummarizer + ContextManager + TripPlanner 集成
"""
import asyncio
import os
import sys
import tempfile

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_user_profile_store():
    """测试 UserProfileStore 的 CRUD 操作"""
    print("=" * 60)
    print("[TEST] UserProfileStore CRUD")
    
    from memory.store import UserProfileStore, UserPreferences
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        store = UserProfileStore(db_path)
        
        # 1. 默认偏好
        prefs = store.get_preferences("user_test")
        assert prefs.budget_level == "中等"
        assert prefs.transport == []
        print("  [OK] 默认偏好为空")
        
        # 2. 更新偏好
        new_prefs = UserPreferences(
            budget_level="豪华",
            transport=["自驾"],
            hotel_type="豪华型酒店",
            interests=["美食探店", "自然风光"],
            travel_style="悠闲",
            favorite_cities=["成都"],
        )
        store.update_preferences("user_test", new_prefs)
        loaded = store.get_preferences("user_test")
        assert loaded.budget_level == "豪华"
        assert "自驾" in loaded.transport
        assert "美食探店" in loaded.interests
        print("  [OK] 偏好保存并读取成功")
        
        # 3. 增量合并
        merge_prefs = UserPreferences(
            transport=["公共交通"],
            interests=["历史文化"],
        )
        store.update_preferences("user_test", merge_prefs)
        merged = store.get_preferences("user_test")
        assert "自驾" in merged.transport
        assert "公共交通" in merged.transport
        assert "美食探店" in merged.interests
        assert "历史文化" in merged.interests
        print("  [OK] 偏好增量合并成功")
        
        # 4. to_prompt_fragment
        fragment = merged.to_prompt_fragment()
        assert "预算偏好" in fragment
        assert "交通偏好" in fragment
        assert "住宿偏好" in fragment
        print("  [OK] to_prompt_fragment 生成成功")
        print(f"  Prompt fragment:\n{fragment}")
        
        # 5. 对话消息
        store.add_message("user_test", "sess_1", "user", "我想去成都玩3天")
        store.add_message("user_test", "sess_1", "assistant", "好的，为您规划成都3日游...")
        msgs = store.get_messages("user_test", "sess_1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        print(f"  [OK] 对话消息存储成功: {len(msgs)} 条")
        
        # 6. 批量添加
        store.add_messages_batch("user_test", "sess_2", [
            {"role": "user", "content": "换一个城市"},
            {"role": "assistant", "content": "好的，请问想去哪里？"},
            {"role": "user", "content": "杭州"},
        ])
        msgs2 = store.get_messages("user_test", "sess_2")
        assert len(msgs2) == 3
        print(f"  [OK] 批量消息存储成功: {len(msgs2)} 条")
        
        # 7. 会话总结
        store.save_summary("user_test", "sess_1", 
            "用户想去成都3日游，偏好美食和自然风光",
            {"budget_level": "中等", "interests": ["美食"]}, 2)
        summaries = store.get_summaries("user_test")
        assert len(summaries) == 1
        assert "成都" in summaries[0]["summary"]
        print(f"  [OK] 会话总结存储成功: {len(summaries)} 条")
        
        # 8. 标记消息已总结
        msg_ids = [m["id"] for m in msgs if "id" in m]
        if msg_ids:
            store.mark_summarized(msg_ids)
            remaining = store.get_messages("user_test", "sess_1")
            assert len(remaining) == 0  # 全部标记为已总结
            print("  [OK] 消息标记已总结成功")
        
        # 9. 最近会话
        sessions = store.get_recent_sessions("user_test")
        assert "sess_1" in sessions or "sess_2" in sessions
        print(f"  [OK] 最近会话: {sessions}")

    print("[PASS] UserProfileStore 全部测试通过\n")


def test_context_manager():
    """测试 ContextManager 的 Token 估算和消息裁剪"""
    print("=" * 60)
    print("[TEST] ContextManager")
    
    from memory.context import ContextManager
    
    ctx = ContextManager(max_tokens=2000)
    
    # 1. Token 估算
    assert ctx.estimate_tokens("") == 0
    assert ctx.estimate_tokens("hello") == 2  # 5 chars // 2
    assert ctx.estimate_tokens("你好世界") == 2  # 4 chars // 2
    print("  [OK] Token 估算正常")
    
    # 2. 消息裁剪 - 预算充足
    system = "你是一个助手" * 10  # ~100 chars → 50 tokens
    messages = [
        {"role": "user", "content": f"消息{i}" * 20} for i in range(20)
    ]
    result, total = ctx.prepare(system, messages)
    assert len(result) >= 2  # system + at least 1 message
    assert result[0]["role"] == "system"
    assert total <= ctx.max_tokens
    print(f"  [OK] 消息裁剪: 输入{len(messages)}条 → 输出{len(result)-1}条, tokens~{total}")
    
    # 3. should_summarize 判断
    short = [{"role": "user", "content": "hi"}] * 5
    assert not ctx.should_summarize(short, threshold_tokens=500)
    long = [{"role": "user", "content": "x" * 10000}] * 3
    assert ctx.should_summarize(long, threshold_tokens=500)
    print("  [OK] should_summarize 判断正常")
    
    # 4. 历史摘要注入
    result2, _ = ctx.prepare(
        system, messages[:5],
        history_summaries=["用户偏好豪华酒店", "用户喜欢美食"],
        user_preferences_text="预算: 豪华\n交通: 自驾",
    )
    system_content = result2[0]["content"]
    assert "用户长期偏好" in system_content
    assert "历史会话摘要" in system_content
    print("  [OK] 历史摘要 + 用户偏好注入到 system prompt")
    
    print("[PASS] ContextManager 全部测试通过\n")


def test_summarizer_structure():
    """测试 ConversationSummarizer 的 JSON 解析（不调 LLM）"""
    print("=" * 60)
    print("[TEST] ConversationSummarizer (JSON 解析)")
    
    from memory.summarizer import ConversationSummarizer
    
    # 创建一个 mock，只测试 _parse_json
    class MockLLM:
        pass
    
    summarizer = ConversationSummarizer(MockLLM())
    
    # 测试各种 JSON 格式解析
    test_cases = [
        ('{"summary": "test", "preferences": {"budget_level": "豪华"}}',
         {"summary": "test", "preferences": {"budget_level": "豪华"}}),
        ('```json\n{"summary": "test2", "preferences": {}}\n```',
         {"summary": "test2", "preferences": {}}),
        ('前面文字{"summary": "test3", "preferences": {"interests": ["美食"]}}后面文字',
         {"summary": "test3", "preferences": {"interests": ["美食"]}}),
    ]
    
    for i, (inp, expected) in enumerate(test_cases):
        result = summarizer._parse_json(inp)
        assert result.get("summary") == expected["summary"], f"case {i} failed"
        print(f"  [OK] 解析 case {i+1}: {result.get('summary')}")
    
    # 测试 Preferences.from_dict
    from memory.store import UserPreferences
    prefs = UserPreferences.from_dict({
        "budget_level": "经济",
        "transport": ["公共交通"],
        "interests": ["自然风光"],
    })
    assert prefs.budget_level == "经济"
    assert prefs.transport == ["公共交通"]
    print("  [OK] UserPreferences.from_dict")
    
    print("[PASS] ConversationSummarizer 全部测试通过\n")


def test_integration_imports():
    """测试所有 memory 模块能否正确导入"""
    print("=" * 60)
    print("[TEST] Memory 模块导入")
    
    from memory import UserProfileStore, ConversationSummarizer, ContextManager
    from memory.store import UserPreferences
    from memory.context import ContextManager as CM
    from memory.summarizer import ConversationSummarizer as CS, SummaryResult
    
    print("  [OK] memory 顶层导入")
    print("  [OK] UserProfileStore")
    print("  [OK] UserPreferences")
    print("  [OK] ConversationSummarizer")
    print("  [OK] ContextManager")
    print("  [OK] SummaryResult")
    
    # 测试 prompts 新函数（不需要导入 TripPlanner）
    from agents.prompts import build_planner_prompt, PLANNER_AGENT_PROMPT
    prompt1 = build_planner_prompt("")
    prompt2 = build_planner_prompt("预算: 豪华\n交通: 自驾")
    assert "用户长期偏好" not in prompt1
    assert "用户长期偏好" in prompt2
    print("  [OK] build_planner_prompt 动态生成")
    print("  [OK] PLANNER_AGENT_PROMPT 向后兼容（无偏好占位符: " +
          str("用户长期偏好" not in PLANNER_AGENT_PROMPT) + ")")
    
    print("[PASS] 模块导入全部通过\n")


def test_preferences_merge():
    """测试偏好合并逻辑"""
    print("=" * 60)
    print("[TEST] UserPreferences 合并逻辑")
    
    from memory.store import UserPreferences
    
    base = UserPreferences(
        budget_level="中等",
        transport=["自驾"],
        hotel_type="豪华型",
        interests=["美食探店"],
    )
    
    new = UserPreferences(
        budget_level="豪华",
        transport=["公共交通"],
        interests=["历史文化"],
        travel_style="悠闲",
    )
    
    base.merge(new)
    
    assert base.budget_level == "豪华"  # 被覆盖
    assert "自驾" in base.transport
    assert "公共交通" in base.transport  # 合并
    assert base.hotel_type == "豪华型"  # 保持
    assert "美食探店" in base.interests
    assert "历史文化" in base.interests  # 合并
    assert base.travel_style == "悠闲"
    print(f"  [OK] 合并后: budget={base.budget_level}, transport={base.transport}, "
          f"interests={base.interests}, style={base.travel_style}")
    
    # 空值不覆盖
    empty = UserPreferences()
    base2 = UserPreferences(budget_level="豪华", hotel_type="豪华型")
    base2.merge(empty)
    assert base2.budget_level == "豪华"
    assert base2.hotel_type == "豪华型"
    print("  [OK] 空偏好不覆盖已有值")
    
    print("[PASS] 偏好合并全部通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  长期记忆管理模块 — 单元测试")
    print("=" * 60 + "\n")
    
    test_user_profile_store()
    test_context_manager()
    test_summarizer_structure()
    test_integration_imports()
    test_preferences_merge()
    
    print("=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
