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
    model_name: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_MODEL_NAME", "qwen3-max")
    )
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

    # ========== RAG 知识库配置 ==========
    # 文本切分参数（用户常问的 chunk_size 在这里）
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 80
    rag_separators: list = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", ";", ".", " "
    ])
    # Embedding 模型（通义千问官方文本向量模型）
    rag_embedding_model: str = "text-embedding-v3"
    # 向量库本地存储路径
    rag_vectorstore_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vectorstore"
    ))
    # 知识库文档目录
    rag_knowledge_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "knowledge"
    ))
    # 检索时返回的 Top-K 文档块数
    rag_top_k: int = 5
    # 检索相似度阈值（越小越严格，0.0~1.0）
    rag_score_threshold: float = 0.3

    # ========== 长期记忆配置 ==========
    # SQLite 数据库路径
    memory_db_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "memory.db"
    ))
    # 上下文窗口 Token 上限（输入部分）
    memory_max_input_tokens: int = 12000
    # 触发总结的 token 阈值（超过该值后自动总结旧消息）
    memory_summarize_threshold: int = 6000
    # 保留最近对话轮数（裁剪后）
    memory_keep_recent_rounds: int = 6

    # ========== LangSmith 追踪配置 ==========
    # 是否启用 LangSmith（需配置 LANGCHAIN_API_KEY 环境变量）
    langsmith_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    # LangSmith API Key（也可通过 LANGCHAIN_API_KEY 环境变量设置）
    langsmith_api_key: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", "")
    )
    # LangSmith 项目名（用于在 Dashboard 中分组）
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "travel-agent")
    )
    # LangSmith 端点（SaaS 用户无需修改，私有化部署时设置）
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    )

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

    # 创建 Embedding 实例对象（用于 RAG）
    def create_embeddings(self):
        """创建通义千问文本向量 Embedding 模型"""
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            return DashScopeEmbeddings(
                model=self.rag_embedding_model,
                dashscope_api_key=self.api_key,
            )
        except Exception as e:
            print(f"[RAG] 创建通义 Embedding 失败，将退回 HuggingFace: {e}")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name="shibing624/text2vec-base-chinese",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as e2:
                print(f"[RAG] 创建 HuggingFace Embedding 也失败: {e2}")
                return None


CONFIG = Config()
