"""
历史对话总结器 —— 用 LLM 对对话历史做结构化总结，提取用户旅行偏好。

核心功能:
  1. 将长对话压缩为结构化摘要
  2. 从对话中提取用户偏好（预算、交通、住宿、兴趣等）
  3. 支持增量更新：新对话追加到已有总结

用法:
    summarizer = ConversationSummarizer(llm)
    result = await summarizer.summarize(messages, existing_summary="")
    # result.summary_text:  自然语言摘要
    # result.preferences:   UserPreferences 对象
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from memory.store import UserPreferences


_SUMMARIZE_SYSTEM_PROMPT = """你是一个对话总结专家。分析以下旅行规划对话，输出结构化 JSON。

## 输出格式（严格 JSON）:
```json
{
  "summary": "1-3句话概括这次对话的主要内容（用户需求、生成的行程特点等）",
  "preferences": {
    "budget_level": "经济/中等/豪华 之一，无法判断则为空字符串",
    "transport": ["偏好的交通方式列表，如 自驾、公共交通、打车"],
    "hotel_type": "偏好的住宿类型，如 经济型酒店、豪华型酒店、民宿/客栈",
    "interests": ["偏好的旅行类型，如 自然风光、历史文化、美食探店、休闲度假"],
    "travel_style": "悠闲/紧凑/深度游 之一",
    "favorite_cities": ["提到的目的地城市"],
    "special_requirements": "特殊要求，如带老人、带小孩、无障碍等"
  }
}
```

## 提取规则:
1. 只提取用户**明确表达**的偏好，不要凭空推断
2. 无法判断的字段用空字符串或空数组
3. budget_level: 用户说"穷游"/"省钱"→经济，"豪华"/"高端"→豪华，其余→中等
4. summary 要简洁，突出关键信息
5. 如果对话内容完全无关旅行规划，preferences 全部留空，summary 如实描述

## 示例:
用户说: "想去成都玩3天，喜欢美食和历史文化，住好一点的酒店"
输出: {"summary": "用户计划成都3日游，偏好美食探店和历史文化景点，倾向高端住宿", "preferences": {"budget_level": "", "transport": [], "hotel_type": "豪华型酒店", "interests": ["美食探店", "历史文化"], "travel_style": "", "favorite_cities": ["成都"], "special_requirements": ""}}
"""


@dataclass
class SummaryResult:
    """总结结果"""
    summary_text: str
    preferences: UserPreferences
    raw_json: Dict[str, Any]


class ConversationSummarizer:
    """用 LLM 对对话历史做结构化总结与偏好提取"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _build_messages_text(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表转为可读文本"""
        lines = []
        for m in messages:
            role_label = "用户" if m["role"] == "user" else "AI助手"
            content = (m.get("content") or "")[:500]  # 每条消息截断 500 字
            lines.append(f"[{role_label}]: {content}")
        return "\n".join(lines)

    async def summarize(self, messages: List[Dict[str, str]],
                        existing_summary: str = "") -> SummaryResult:
        """
        对对话历史做结构化总结。

        Args:
            messages: [{"role": "user"/"assistant", "content": "..."}, ...]
            existing_summary: 已有的历史总结（用于增量更新）

        Returns:
            SummaryResult 包含摘要文本和提取的偏好
        """
        if not messages:
            return SummaryResult(
                summary_text="",
                preferences=UserPreferences(),
                raw_json={}
            )

        conversation_text = self._build_messages_text(messages)

        user_content = f"""## 对话历史
{conversation_text}
"""
        if existing_summary:
            user_content += f"""
## 已有历史总结
{existing_summary}

请基于以上已有总结和新对话，更新总结内容和偏好提取。
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            result_text = (response.content or "").strip()
        except Exception as e:
            print(f"[Summarizer] LLM 调用失败: {e}")
            return SummaryResult(
                summary_text=f"总结失败: {e}",
                preferences=UserPreferences(),
                raw_json={}
            )

        # 解析 JSON
        parsed = self._parse_json(result_text)

        summary = parsed.get("summary", "")
        prefs_dict = parsed.get("preferences", {})
        preferences = UserPreferences.from_dict(prefs_dict)

        return SummaryResult(
            summary_text=summary,
            preferences=preferences,
            raw_json=parsed,
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 最外层
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"[Summarizer] 无法解析 LLM 输出为 JSON: {text[:200]}")
        return {}
