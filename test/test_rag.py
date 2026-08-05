"""
RAG 知识库功能验证脚本。

用法:
    python test_rag.py          # 完整流程测试：构建向量库 + 3 个检索用例
    python test_rag.py build    # 仅构建/重建向量库
    python test_rag.py search   # 仅做检索测试（库已存在时）
    python test_rag.py stats    # 查看知识库状态
"""
import asyncio
import sys
from rag_engine import get_rag_engine


TEST_QUERIES = [
    ("杭州西湖最佳游览顺序", "杭州旅行攻略"),
    ("北京烤鸭推荐哪家店", "北京旅行攻略"),
    ("成都必吃美食排行", "成都旅行攻略"),
    ("亲子游出行准备清单", "旅行通用贴士"),
    ("上海外滩夜景几点看最好", "上海旅行攻略"),
]


async def build_rag():
    print("=" * 60)
    print("🧱 步骤 1: 构建/重建 RAG 知识库向量库")
    print("=" * 60)
    rag = get_rag_engine()
    ok = await rag.build(force_rebuild=True)
    if not ok:
        print("❌ 构建失败，检查 Embedding 和 FAISS 依赖")
        return False

    stats = rag.get_stats()
    print("\n✅ 构建成功！知识库状态:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    return True


async def search_rag():
    print("\n" + "=" * 60)
    print("🔍 步骤 2: 检索质量验证")
    print("=" * 60)
    rag = get_rag_engine()
    if not rag.is_ready():
        ok = await rag.build(force_rebuild=False)
        if not ok:
            print("❌ RAG 未就绪，无法检索")
            return False

    all_pass = True
    for query, expected_source in TEST_QUERIES:
        print(f"\n--- 查询: {query} (预期来源: {expected_source}) ---")
        results = rag.search_knowledge(query, top_k=3)
        if not results:
            print(f"   ❌ 未检索到任何结果")
            all_pass = False
            continue
        for i, r in enumerate(results, 1):
            print(f"   [{i}] score={r.score:.3f} | source={r.source}")
            snippet = r.content[:80].replace("\n", " ")
            print(f"        {snippet}...")
        top_hit = results[0]
        hit_ok = expected_source in top_hit.source
        mark = "✅" if hit_ok else "⚠️"
        print(f"   {mark} Top1 来源匹配: {expected_source} → {top_hit.source}")

    return all_pass


async def show_stats():
    rag = get_rag_engine()
    print("=" * 60)
    print("📊 RAG 知识库状态")
    print("=" * 60)
    if rag.is_ready():
        for k, v in rag.get_stats().items():
            print(f"   {k}: {v}")
    else:
        print("   知识库尚未构建，运行 python test_rag.py build")


async def main():
    args = sys.argv[1:]
    if "stats" in args:
        await show_stats()
    elif "build" in args:
        await build_rag()
    elif "search" in args:
        await search_rag()
    else:
        ok1 = await build_rag()
        ok2 = await search_rag() if ok1 else False
        print("\n" + "=" * 60)
        if ok1 and ok2:
            print("🎉 全部通过！RAG 知识库功能正常工作")
        else:
            print("⚠️ 存在异常，请检查上方日志")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
