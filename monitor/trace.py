"""
Agent 链路追踪 + JSONL 日志。

核心能力：
  - new_trace()         : 生成 Trace ID，创建根 Span
  - TraceContext.start_span() : 记录子任务起止 & 耗时
  - Span.end()          : 结束 Span 并写入 `logs/trace.jsonl`
  - end_trace()         : 结束整条 Trace，写入摘要行 + 控制台报告

输出格式（每条一行 JSON）：
  1. Span 日志：{"type":"span", "trace_id":"...", "span_id":"...", "name":"search_hotel", "duration_ms":1234, ...}
  2. 摘要日志：{"type":"trace_summary", "trace_id":"...", "total_ms":5678, "span_count":7, ...}
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---- logs 目录 ----
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "trace.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# Span
# ============================================================

@dataclass
class Span:
    """单次操作（工具调用 / Agent调用 / RAG检索）的追踪记录"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"          # running / ok / error
    error_msg: Optional[str] = None
    _ended: bool = False

    def end(self, success: bool = True, metadata: Optional[Dict] = None,
            error: Optional[str] = None) -> float:
        """结束 Span，写入 JSONL，返回 duration_ms"""
        if self._ended:
            return self.duration_ms or 0.0
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = "ok" if success else "error"
        if error:
            self.error_msg = str(error)[:500]
        if metadata:
            self.metadata.update(metadata)
        self._ended = True
        _append_line(self._to_line())
        return self.duration_ms

    def _to_line(self) -> dict:
        return {
            "type": "span",
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_ts": datetime.fromtimestamp(self.start_time).isoformat(timespec="milliseconds"),
            "end_ts": datetime.fromtimestamp(self.end_time).isoformat(timespec="milliseconds") if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_msg": self.error_msg,
            "metadata": self.metadata,
        }


# ============================================================
# TraceContext
# ============================================================

@dataclass
class TraceContext:
    """一次用户请求的完整追踪上下文"""
    trace_id: str
    user_input: str
    start_time: float
    spans: List[Span] = field(default_factory=list)
    success: Optional[bool] = None
    total_ms: Optional[float] = None
    _root_span: Optional[Span] = None

    def start_span(self, name: str, parent: Optional[Span] = None,
                   metadata: Optional[Dict] = None) -> Span:
        """创建一个子 Span 并注册到 trace 中"""
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=(parent.span_id if parent
                            else (self._root_span.span_id if self._root_span else None)),
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        short_id = self.trace_id[-6:]
        print(f"  [TRACE {short_id}] START {name}")
        return span

    def _close(self, success: bool, error: Optional[str] = None):
        """结束根 Span + 写摘要"""
        if self._root_span and not self._root_span._ended:
            self._root_span.end(success=success, error=error)
        self.total_ms = round((time.time() - self.start_time) * 1000, 2)
        self.success = success
        # 写摘要 JSONL
        _append_line(_summary_line(self))
        # 打印控制台报告
        _print_console(self)

    def summary(self) -> dict:
        stats = _span_stats(self.spans)
        return {
            "trace_id": self.trace_id,
            "user_input": (self.user_input or "")[:200],
            "start_ts": datetime.fromtimestamp(self.start_time).isoformat(timespec="milliseconds"),
            "total_ms": self.total_ms,
            "success": self.success,
            "span_count": len(self.spans),
            "span_stats": stats,
        }


# ============================================================
# 公开 API
# ============================================================

_CURRENT_TRACE: Optional[TraceContext] = None


def new_trace(user_input: str) -> TraceContext:
    """开启一次用户请求的追踪。在整个请求生命周期内复用同一个 TraceContext"""
    global _CURRENT_TRACE
    tid = uuid.uuid4().hex[:16]
    trace = TraceContext(trace_id=tid, user_input=user_input, start_time=time.time())

    # 创建根 Span（Planner 总控）
    root = Span(
        trace_id=tid, span_id=uuid.uuid4().hex[:12],
        parent_span_id=None, name="Planner.root",
        start_time=trace.start_time,
        metadata={"user_input": (user_input or "")[:200]},
    )
    trace._root_span = root
    trace.spans.append(root)

    _CURRENT_TRACE = trace
    short_id = tid[-6:]
    user_snippet = (user_input or '')[:60]
    print(f"\n[TRACE {short_id}] NEW | {user_snippet}...")
    return trace


def current_trace() -> Optional[TraceContext]:
    """获取当前活跃的 TraceContext（供跨模块使用）"""
    return _CURRENT_TRACE


def end_trace(trace: TraceContext, success: bool = True,
              error: Optional[str] = None) -> dict:
    """结束并写入追踪。返回 summary dict"""
    global _CURRENT_TRACE
    trace._close(success=success, error=error)
    _CURRENT_TRACE = None
    return trace.summary()


# ============================================================
# 内部：JSONL 写入 & 控制台可视化
# ============================================================

def _append_line(data: dict):
    try:
        line = json.dumps(data, ensure_ascii=False)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
            print(f"  [WARN] trace write failed: {e}")


def _summary_line(trace: TraceContext) -> dict:
    return {"type": "trace_summary", "dt": datetime.now().strftime("%Y-%m-%d"),
            **trace.summary()}


def _span_stats(spans: List[Span]) -> dict:
    """按 span name 聚合统计：count / total / avg / max"""
    by_name: Dict[str, List[float]] = {}
    for s in spans:
        if s.duration_ms is not None:
            by_name.setdefault(s.name, []).append(s.duration_ms)
    stats = {}
    for name, durs in by_name.items():
        stats[name] = {
            "count": len(durs),
            "total_ms": round(sum(durs), 1),
            "avg_ms":   round(sum(durs) / len(durs), 1),
            "max_ms":   round(max(durs), 1),
        }
    return stats


def _print_console(trace: TraceContext):
    icon = "OK" if trace.success else "FAIL"
    print(f"\n{icon} [TRACE {trace.trace_id[-6:]}] END  "
          f"total={trace.total_ms}ms  |  spans={len(trace.spans)}")
    # 按总耗时降序打印各 span 的统计
    stats = _span_stats([s for s in trace.spans if s.duration_ms is not None])
    if stats:
        print("   -- Latency Breakdown --")
        for name, st in sorted(stats.items(), key=lambda x: -x[1]["total_ms"]):
            print(f"   * {name:<32} calls={st['count']:<3}"
                  f" total={st['total_ms']:>7.0f}ms"
                  f" avg={st['avg_ms']:>7.0f}ms"
                  f" max={st['max_ms']:>7.0f}ms")
    print(f"   log: {LOG_FILE}")
