"""
上下文窗口管理器 —— Token 估算 + 消息筛选裁剪 + 历史摘要注入。

策略:
  1. 保留 System Prompt（不裁剪）
  2. 保留最近 N 轮完整对话（默认 6 轮 = 12 条消息）
  3. 更早的消息用「历史摘要」替代
  4. 如果总 token 仍超限，缩小最近轮数
  5. 确保 user 输入作为最后一条消息

用法:
    ctx = ContextManager(max_tokens=8000)
    prepared = ctx.prepare(system_prompt, messages, summaries)
"""

from typing import Any, Dict, List, Optional, Tuple


class ContextManager:
    """
    上下文窗口管理器。

    负责:
      1. Token 数量估算
      2. 消息裁剪（保留 system + 最近 N 轮 + 摘要）
      3. 确保不超出模型上下文窗口
    """

    # 模型上下文窗口（qwen3-max = 128k tokens）
    MODEL_MAX_TOKENS = 128_000

    # 预留输出 buffer
    OUTPUT_BUFFER = 4000

    def __init__(self, max_tokens: int = 12000):
        """
        Args:
            max_tokens: 输入消息允许的最大 token 数（不含输出 buffer）
        """
        self.max_tokens = max_tokens

    # ==================== Token 估算 ====================

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        粗略估算 token 数量。
        中文:   约 1.5 字符/token
        英文:   约 4 字符/token
        混合:   平均约 2.5 字符/token，保守用 2 字符/token
        """
        if not text:
            return 0
        # 保守估算：中英文混合约 2 字符 = 1 token
        return max(1, len(text) // 2)

    @classmethod
    def estimate_message_tokens(cls, message: Dict[str, str]) -> int:
        """估算单条消息的 token 数"""
        content = message.get("content", "")
        tokens = cls.estimate_tokens(content)
        # role 占少量 token
        return tokens + 4

    @classmethod
    def estimate_messages_tokens(cls, messages: List[Dict[str, str]]) -> int:
        """估算消息列表的总 token 数"""
        return sum(cls.estimate_message_tokens(m) for m in messages)

    # ==================== 消息裁剪 ====================

    def prepare(
        self,
        system_content: str,
        messages: List[Dict[str, str]],
        history_summaries: Optional[List[str]] = None,
        user_preferences_text: str = "",
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        准备裁剪后的消息列表。

        Args:
            system_content: System Prompt 文本
            messages: 完整对话历史 [{"role": "user"/"assistant", "content": "..."}]
            history_summaries: 历史会话摘要列表（最新的在前）
            user_preferences_text: 用户偏好文本（注入 system prompt）

        Returns:
            (裁剪后的消息列表, 估算总 token 数)
        """
        history_summaries = history_summaries or []

        # 1. 构建 system message + 偏好 + 摘要
        system_full = system_content
        if user_preferences_text:
            system_full += (
                f"\n\n## 用户长期偏好（来自历史对话）\n{user_preferences_text}\n"
                "请优先参考以上偏好进行规划和推荐。"
            )
        if history_summaries:
            summaries_text = "\n".join(
                f"- {s}" for s in history_summaries[:3]  # 最多 3 条历史摘要
            )
            system_full += f"\n\n## 历史会话摘要\n{summaries_text}"

        system_msg = {"role": "system", "content": system_full}
        system_tokens = self.estimate_tokens(system_full)

        # 2. 从后往前保留最近几轮
        # 一轮 = user + assistant 或 user 单条
        budget = self.max_tokens - system_tokens - 200  # 200 token 安全余量

        # 确保最后一条是 user 消息
        if messages and messages[-1]["role"] != "user":
            # 找到最后一个 user 消息的位置
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break
            if last_user_idx >= 0:
                messages = messages[:last_user_idx + 1]

        # 从后往前选消息，直到 token 预算用完
        selected = []
        total = 0
        for m in reversed(messages):
            t = self.estimate_message_tokens(m)
            if total + t > budget:
                break
            selected.insert(0, m)
            total += t

        # 3. 如果选中的消息太少（< 2 条），至少保留最后一条 user
        if len(selected) < 2 and messages:
            selected = [messages[-1]]
            total = self.estimate_message_tokens(messages[-1])

        result = [system_msg] + selected
        return result, system_tokens + total

    # ==================== 辅助 ====================

    def should_summarize(self, messages: List[Dict[str, str]],
                         threshold_tokens: int = 6000) -> bool:
        """判断是否需要触发对话总结"""
        return self.estimate_messages_tokens(messages) > threshold_tokens

    def get_keep_recent_count(self, available_budget: int) -> int:
        """
        根据可用 token 预算，计算应保留最近几轮完整对话。
        粗略估计每轮对话约 500 tokens。
        """
        rounds = max(1, available_budget // 500)
        return min(rounds, 10)  # 最多保留 10 轮
