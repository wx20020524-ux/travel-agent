from dotenv import load_dotenv
import os
from langchain_community.chat_models.tongyi import ChatTongyi

# 读取 .env
load_dotenv()

llm = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("sk-ws-H.RPXMLXR.3QPK.MEQCIBRgVtyxbygov4Z3IBW0VArFyU--ChCLtm5IdfumXKvuAiAr7U6NkVwmrVHoJxzA94kJucOWFDXwA8e-h_4DRpSMhA"),
)

response = llm.invoke("你好，请简单介绍一下自己")

print(response)