import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.prompts import (
    ATTRACTION_AGENT_PROMPT,
    HOTEL_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
)


def test_specialist_prompts_use_native_tool_calling():
    prompts = {
        "weather": WEATHER_AGENT_PROMPT,
        "attraction": ATTRACTION_AGENT_PROMPT,
        "hotel": HOTEL_AGENT_PROMPT,
    }

    for name, prompt in prompts.items():
        assert "TOOL_CALL" not in prompt, name
        assert "方括号" not in prompt, name
        assert "工具 schema" in prompt, name

    assert "maps_weather" in WEATHER_AGENT_PROMPT
    assert "maps_text_search" in ATTRACTION_AGENT_PROMPT
    assert "maps_text_search" in HOTEL_AGENT_PROMPT


if __name__ == "__main__":
    test_specialist_prompts_use_native_tool_calling()
    print("[PASS] agent prompt regression tests")
