#!/usr/bin/env python3
"""简单测试"""
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
print("API Key loaded:")
print("-" * 50)
print(repr(api_key))
print("-" * 50)
print(f"Length: {len(api_key)}")

# 测试是否能读取
if api_key:
    print("\nAPI Key seems to be loaded correctly")
else:
    print("\nERROR: API Key not found!")
