import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.summarizer import ConversationSummarizer


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("LLM unavailable")


def test_summarizer_propagates_llm_failure():
    summarizer = ConversationSummarizer(FailingLLM())
    messages = [{"role": "user", "content": "我想去成都玩"}]

    try:
        asyncio.run(summarizer.summarize(messages))
    except RuntimeError as error:
        assert str(error) == "LLM unavailable"
    else:
        raise AssertionError("summarizer swallowed the LLM failure")


if __name__ == "__main__":
    test_summarizer_propagates_llm_failure()
    print("[PASS] summarizer regression tests")
