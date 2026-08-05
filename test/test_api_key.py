#!/usr/bin/env python3
"""测试 API Key 是否有效"""
from dotenv import load_dotenv
import os
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"API Key 长度: {len(api_key) if api_key else 0}")
print(f"API Key 前缀: {api_key[:20] if api_key else 'None'}...")

try:
    print("\n正在测试通义千问 API...")
    llm = ChatTongyi(
        model="qwen3-max",
        api_key=api_key,
        temperature=0.7
    )
    
    response = llm.invoke("你好，请用一句话介绍自己")
    print(f"\n✓ API 测试成功!")
    print(f"响应: {response.content}")
    
except Exception as e:
    print(f"\n✗ API 测试失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
