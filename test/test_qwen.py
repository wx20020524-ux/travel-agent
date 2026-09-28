from dotenv import load_dotenv
import os
from langchain_community.chat_models.tongyi import ChatTongyi

# 读取 .env
load_dotenv()

llm = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

response = llm.invoke("你好，请简单介绍一下自己")

print(response)
