"""
智能旅行助手 —— 入口。

用法:
    python agents/agent.py                      # 流式输出（默认，无记忆）
    python agents/agent.py --no-stream          # 非流式输出
    python agents/agent.py --build-rag          # 仅重建 RAG 知识库向量库
    python agents/agent.py --user-id user_001   # 启用长期记忆
    python agents/agent.py --demo-memory        # 演示长期记忆功能
"""
import asyncio
import os
import sys
import uuid

# 确保项目根目录在 sys.path 中（支持 python agents/agent.py 方式运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import CONFIG
from rag.rag_engine import get_rag_engine
from agents.planner import TripPlanner
from render import format_plan_cli


# ==================== 演示 ====================

async def demo_stream(planner: TripPlanner, user_input: str,
                      user_id: str = None):
    """流式输出演示"""
    print("=" * 60)
    print(f"[Planner] 正在为您规划旅行...\n输入: {user_input}\n")
    if user_id:
        print(f"[Memory] user_id={user_id}")
    print("=" * 60)

    buffer = ""
    async for token in planner.stream(user_input, user_id=user_id):
        print(token, end="", flush=True)
        buffer += token

    formatted = format_plan_cli(buffer)
    if formatted:
        print(formatted)

    print("=" * 60)
    print("[Planner] 旅行计划生成完毕")


async def demo_invoke(planner: TripPlanner, user_input: str,
                      user_id: str = None):
    """非流式输出演示"""
    print("=" * 60)
    print(f"[Planner] 正在为您规划旅行...\n输入: {user_input}\n")
    print("=" * 60)

    result = await planner.invoke(user_input, user_id=user_id)
    formatted = format_plan_cli(result)
    if formatted:
        print(formatted)
    else:
        print(result)

    print("=" * 60)
    print("[Planner] 旅行计划生成完毕")


async def demo_memory(planner: TripPlanner):
    """演示长期记忆功能：连续两轮对话，第二轮自动注入偏好"""
    user_id = f"demo_user_{uuid.uuid4().hex[:8]}"
    print("=" * 60)
    print("[Memory Demo] 长期记忆功能演示")
    print(f"  user_id = {user_id}")
    print("=" * 60)

    # 第一轮：用户表达偏好
    round1 = ("成都3日游，2026年8月15日-2026年8月17日，"
              "喜欢美食探店和休闲度假，住豪华型酒店，预算充裕")
    print(f"\n[Round 1] 用户输入: {round1}\n")
    print("-" * 60)
    buffer1 = ""
    async for token in planner.stream(round1, user_id=user_id):
        print(token, end="", flush=True)
        buffer1 += token
    print("\n")

    # 等待总结完成
    await asyncio.sleep(2)

    # 打印提取的偏好
    from config import CONFIG as cfg
    from memory.store import UserProfileStore
    store = UserProfileStore(cfg.memory_db_path)
    prefs = store.get_preferences(user_id)
    print(f"\n[Memory] 提取的用户偏好:")
    print(f"  预算: {prefs.budget_level}")
    print(f"  兴趣: {prefs.interests}")
    print(f"  住宿: {prefs.hotel_type}")
    print(f"  风格: {prefs.travel_style}")

    # 第二轮：简短输入，依赖记忆
    round2 = "帮我也规划一个杭州2日游，其他要求跟上次一样"
    print(f"\n[Round 2] 用户输入: {round2}")
    print("(系统将从长期记忆中加载偏好，自动注入 system prompt)\n")
    print("-" * 60)
    async for token in planner.stream(round2, user_id=user_id):
        print(token, end="", flush=True)
    print("\n")

    print("=" * 60)
    print("[Memory Demo] 长期记忆演示完毕")
    print(f"  数据已持久化到: {cfg.memory_db_path}")
    print("=" * 60)


async def main():
    # ---- 初始化 LangSmith（如已配置） ----
    from monitor.langsmith_ import init_langsmith
    init_langsmith()

    args = sys.argv[1:]

    if "--build-rag" in args:
        print("[RAG] 强制重建 RAG 知识库向量库...")
        rag = get_rag_engine()
        ok = await rag.build(force_rebuild=True)
        if ok:
            stats = rag.get_stats()
            print(f"\n[RAG] 知识库构建成功！")
            print(f"   - 文档数: {stats['docs_count']}")
            print(f"   - 文本块数 (chunks): {stats['chunks_count']}")
            print(f"   - chunk_size: {stats['chunk_size']}")
            print(f"   - chunk_overlap: {stats['chunk_overlap']}")
            print(f"   - Embedding模型: {stats['embedding_model']}")
            print(f"   - 向量库目录: {stats['vectorstore_dir']}")
        else:
            print("[RAG] 构建失败")
        return

    llm = CONFIG.create_llm()
    planner = TripPlanner(llm)

    # --demo-memory: 演示长期记忆功能
    if "--demo-memory" in args:
        await demo_memory(planner)
        return

    # 解析 --user-id
    user_id = None
    for a in args:
        if a.startswith("--user-id="):
            user_id = a.split("=", 1)[1]
        elif a == "--user-id":
            idx = args.index("--user-id")
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                user_id = args[idx + 1]

    user_input = "长沙3日游，2026年5月21日-2026年5月23日，喜欢自然风光和历史文化，中等预算，住五一广场"

    if "--no-stream" in args:
        await demo_invoke(planner, user_input, user_id=user_id)
    else:
        await demo_stream(planner, user_input, user_id=user_id)


if __name__ == "__main__":
    asyncio.run(main())
