"""
验证链路追踪 Trace 功能。

用法:
    python test/test_trace.py              # 基础测试
    python test/test_trace.py --integration  # 完整集成测试
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.trace import new_trace, current_trace, end_trace, LOG_FILE
from config import CONFIG


async def test_trace_module():
    """测试 Trace 模块基础功能：span 创建、结束、JSONL 写入"""
    print("=" * 60)
    print("[Test 1] Trace module basics")
    print("=" * 60)

    trace = new_trace("Hangzhou 3-day trip test")

    # simulate agent chain
    span1 = trace.start_span("query_knowledge")
    await asyncio.sleep(0.1)
    span1.end(success=True, metadata={"hit_count": 5})

    span2 = trace.start_span("search_hotel")
    await asyncio.sleep(0.2)
    span2.end(success=True, metadata={"city": "Hangzhou"})

    span3 = trace.start_span("Specialist.WeatherAgent")
    await asyncio.sleep(0.15)
    span3.end(success=True)

    summary = end_trace(trace, success=True)

    print(f"\nTrace summary:")
    print(f"  trace_id:    {summary['trace_id']}")
    print(f"  total_ms:    {summary['total_ms']}")
    print(f"  span_count:  {summary['span_count']}")
    print(f"  span_stats:  {summary['span_stats']}")

    # verify log file
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        span_lines = sum(1 for l in lines if '"type":"span"' in l)
        summary_lines = sum(1 for l in lines if '"type":"trace_summary"' in l)
        print(f"\nPASS - JSONL log written: {LOG_FILE}")
        print(f"  {len(lines)} lines ({span_lines} spans + {summary_lines} summary)")
    else:
        print(f"\nFAIL - JSONL log not found: {LOG_FILE}")


async def test_integration():
    """测试完整集成：Planner.stream() 自动产生 trace"""
    print("\n" + "=" * 60)
    print("[Test 2] Planner.stream() integration")
    print("=" * 60)

    llm = CONFIG.create_llm()
    from agents.planner import TripPlanner
    planner = TripPlanner(llm)

    try:
        buffer = ""
        async for token in planner.stream("A quick one-day trip in Hangzhou"):
            buffer += token
        print(f"\nPASS - stream ok, output {len(buffer)} chars")
    except Exception as e:
        print(f"\nWARN - call failed: {e}")
        print("  (expected if MCP/API unreachable, trace still written)")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        summaries = [l for l in lines if '"type":"trace_summary"' in l]
        print(f"\nLog file: {len(lines)} lines, {len(summaries)} summaries")


if __name__ == "__main__":
    asyncio.run(test_trace_module())
    if "--integration" in sys.argv:
        print("\nHint: integration test requires API + MCP access")
        asyncio.run(test_integration())
