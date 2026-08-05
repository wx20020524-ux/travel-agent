"""
智能旅行助手 —— 入口。

用法:
    python Agent.py                # 流式输出（默认）
    python Agent.py --no-stream    # 非流式输出
    python Agent.py --build-rag    # 仅重建 RAG 知识库向量库（knowledge/文档更新后运行）
"""
import asyncio
import sys
from config import CONFIG
from rag_engine import get_rag_engine
from agents.planner import TripPlanner
from render import format_plan_cli


# ==================== 演示 ====================

async def demo_stream(planner: TripPlanner, user_input: str):
    """流式输出演示 —— 实时打印 token，结束后渲染格式化计划"""
    print("=" * 60)
    print(f"🚀 正在为您规划旅行...\n输入: {user_input}\n")
    print("=" * 60)

    buffer = ""
    async for token in planner.stream(user_input):
        print(token, end="", flush=True)
        buffer += token

    # 格式化渲染
    formatted = format_plan_cli(buffer)
    if formatted:
        print(formatted)

    print("=" * 60)
    print("✅ 旅行计划生成完毕")


async def demo_invoke(planner: TripPlanner, user_input: str):
    """非流式输出演示"""
    print("=" * 60)
    print(f"🚀 正在为您规划旅行...\n输入: {user_input}\n")
    print("=" * 60)

    result = await planner.invoke(user_input)
    formatted = format_plan_cli(result)
    if formatted:
        print(formatted)
    else:
        print(result)

    print("=" * 60)
    print("✅ 旅行计划生成完毕")


async def main():
    args = sys.argv[1:]

    if "--build-rag" in args:
        print("🔧 强制重建 RAG 知识库向量库...")
        rag = get_rag_engine()
        ok = await rag.build(force_rebuild=True)
        if ok:
            stats = rag.get_stats()
            print(f"\n✅ RAG 知识库构建成功！")
            print(f"   - 文档数: {stats['docs_count']}")
            print(f"   - 文本块数 (chunks): {stats['chunks_count']}")
            print(f"   - chunk_size: {stats['chunk_size']}")
            print(f"   - chunk_overlap: {stats['chunk_overlap']}")
            print(f"   - Embedding模型: {stats['embedding_model']}")
            print(f"   - 向量库目录: {stats['vectorstore_dir']}")
        else:
            print("❌ RAG 构建失败")
        return

    llm = CONFIG.create_llm()
    planner = TripPlanner(llm)

    user_input = "长沙3日游，2026年5月21日-2026年5月23日，喜欢自然风光和历史文化，中等预算，住五一广场"

    if "--no-stream" in args:
        await demo_invoke(planner, user_input)
    else:
        await demo_stream(planner, user_input)


if __name__ == "__main__":
    asyncio.run(main())
