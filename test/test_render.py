import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import TOOL_LABELS
from render import STATUS_PREFIXES, is_status_update, parse_plan


def test_parse_plan_extracts_balanced_json_from_mixed_text():
    text = """模型说明 { 这里不是 JSON }
最终结果:
```json
{"city": "北京", "note": "包含 } 和 {", "days": []}
```
"""

    assert parse_plan(text) == {
        "city": "北京",
        "note": "包含 } 和 {",
        "days": [],
    }


def test_parse_plan_returns_none_for_invalid_output():
    assert parse_plan("模型没有返回 JSON") is None


def test_status_prefixes_cover_planner_tool_labels():
    label_prefixes = {prefix for prefix, _ in TOOL_LABELS.values()}
    assert label_prefixes.issubset(STATUS_PREFIXES)


def test_status_update_matches_planner_labels():
    assert is_status_update("[Weather] 查询天气...\n")
    assert is_status_update("  [Hotel] 搜索酒店...\n")
    assert not is_status_update("普通模型输出")
    assert not is_status_update("[Unknown] 未知状态")


if __name__ == "__main__":
    test_parse_plan_extracts_balanced_json_from_mixed_text()
    test_parse_plan_returns_none_for_invalid_output()
    test_status_update_matches_planner_labels()
    test_status_prefixes_cover_planner_tool_labels()
    print("[PASS] render regression tests")
