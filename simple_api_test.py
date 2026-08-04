#!/usr/bin/env python3
"""简单的 API 测试"""
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

print("API Key 信息:")
print(f"  已加载: 是")
print(f"  长度: {len(api_key)}")
print(f"  前缀: {api_key[:15]}...")

print("\n正在测试通义千问 API...")
from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(
    model="qwen3-max",
    api_key=api_key,
    temperature=0.7
)

response = llm.invoke("hi")
result_text = response.content.replace('\n', ' ')[:100]
print(f"测试成功!")
print(f"响应: {result_text}")

print("\nAPI Key 验证完成 - 你的 API Key 是有效的！")
