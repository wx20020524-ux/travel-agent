#!/usr/bin/env python3
"""检查 API Key 是否有效"""
import sys
from dotenv import load_dotenv
import os

def check_api_key_setup():
    """检查 API Key 配置"""
    print("=" * 60)
    print("API Key 检查工具")
    print("=" * 60)
    
    # 1. 检查 .env 文件
    print("\n[1] 检查 .env 文件...")
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"✓ .env 文件存在: {env_file}")
    else:
        print("✗ .env 文件不存在！")
        return False
    
    # 2. 加载并检查 API Key
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    print(f"\n[2] 检查 API Key...")
    if not api_key:
        print("✗ 未找到 DASHSCOPE_API_KEY 环境变量！")
        return False
    
    print(f"✓ API Key 已加载")
    print(f"  长度: {len(api_key)} 字符")
    print(f"  前缀: {api_key[:15]}...")
    
    # 3. 验证 API Key 格式
    print(f"\n[3] 验证 API Key 格式...")
    if api_key.startswith("sk-"):
        print("✓ API Key 格式正确 (以 sk- 开头)")
    else:
        print("✗ API Key 格式可能有问题，应该以 sk- 开头")
    
    # 4. 测试 API Key
    print(f"\n[4] 测试通义千问 API 连接...")
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
        
        llm = ChatTongyi(
            model="qwen3-max",
            api_key=api_key,
            temperature=0.7
        )
        
        response = llm.invoke("你好")
        print(f"✓ API 连接成功！")
        print(f"  模型响应: {response.content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ API 连接失败: {type(e).__name__}")
        print(f"  错误信息: {e}")
        return False

if __name__ == "__main__":
    success = check_api_key_setup()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ API Key 检查完成 - 配置正确！")
    else:
        print("✗ API Key 检查失败，请检查配置")
        print("\n💡 获取正确的 API Key:")
        print("  1. 访问 https://dashscope.console.aliyun.com/")
        print("  2. 登录你的阿里云账号")
        print("  3. 在控制台中创建或获取 API Key")
        print("  4. 将 API Key 更新到 .env 文件中")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
