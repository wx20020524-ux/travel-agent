"""
配置中心 —— 统一管理环境变量、LLM 实例、MCP 连接参数。
"""
import os
import ssl
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv()

# ========== 修复 langchain_community ChatTongyi 流式 tool_calls 的 KeyError ==========
# 上游 bug: subtract_client_response 访问 prev_function["name"] / ["arguments"]
# 前没有检查 key 是否存在。流式首个 tool_call chunk 可能不含这些 key。


def _patched_subtract(self, resp, prev_resp):
    import json

    resp_copy = json.loads(json.dumps(resp))
    message = resp_copy["output"]["choices"][0]["message"]
    prev_message = json.loads(json.dumps(prev_resp))["output"]["choices"][0]["message"]

    message["content"] = message["content"].replace(
        prev_message.get("content", "") or "", ""
    )

    if message.get("tool_calls") and prev_message.get("tool_calls"):
        for index, tool_call in enumerate(message["tool_calls"]):
            function = tool_call["function"]
            prev_function = prev_message["tool_calls"][index]["function"]

            if "name" in function and "name" in prev_function:
                function["name"] = function["name"].replace(prev_function["name"], "")
            if "arguments" in function and "arguments" in prev_function:
                function["arguments"] = function["arguments"].replace(
                    prev_function["arguments"], ""
                )

    return resp_copy


ChatTongyi.subtract_client_response = _patched_subtract
# ========== 修复结束 ==========


def _create_robust_http_client():
    """创建具有 SSL 容错能力的 HTTP 客户端"""
    try:
        import httpx
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')
        
        transport = httpx.HTTPTransport(
            retries=3,
            uds=None
        )
        
        client = httpx.Client(
            timeout=60.0,
            verify=ssl_context,
            http2=False,
            transport=transport,
            follow_redirects=True,
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=60, max=1000"
            }
        )
        return client
    except Exception as e:
        print(f"创建自定义 HTTP 客户端失败，将使用默认: {e}")
        return None


@dataclass
class Config:
    """全局配置，单例语义 —— 模块级 CONFIG 实例"""

    # API 密钥
    api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )

    # LLM
    model_name: str = "qwen3-max"
    temperature: float = 0.7
    
    # 请求超时（秒）
    request_timeout: int = 120
    # 最大重试次数
    max_retries: int = 5

    # MCP 连接（阿里百炼高德地图）
    mcp_transport: str = "http"
    mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"

    # 工具领域映射
    tool_domains: dict = field(default_factory=lambda: {
        "poi":     ["maps_text_search", "maps_search_detail"],
        "weather": ["maps_weather"],
        "route":   [
            "maps_direction_walking_by_address",
            "maps_direction_driving_by_address",
            "maps_direction_transit_integrated_by_address",
        ],
    })

    # 自动检查初始化
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("请配置 DASHSCOPE_API_KEY")

    # 创建模型实例对象
    def create_llm(self) -> ChatTongyi:
        http_client = _create_robust_http_client()
        
        kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "streaming": True,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        
        if http_client is not None:
            kwargs["http_client"] = http_client
        
        return ChatTongyi(**kwargs)


CONFIG = Config()
