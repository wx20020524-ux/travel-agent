#!/usr/bin/env python3
"""网络和 SSL 连接诊断工具"""
import socket
import ssl
import sys
from urllib.request import urlopen
from urllib.error import URLError

def check_dns(host):
    """检查 DNS 解析"""
    print(f"[1] DNS 解析: {host}")
    try:
        ips = socket.getaddrinfo(host, 443)
        print(f"    成功: 解析到 {len(ips)} 个地址")
        for i, ip in enumerate(ips[:3], 1):
            print(f"      {i}. {ip[4][0]}")
        return True
    except Exception as e:
        print(f"    失败: {e}")
        return False

def check_tcp_connect(host, port=443, timeout=10):
    """检查 TCP 连接"""
    print(f"[2] TCP 连接: {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f"    成功: 端口开放")
            return True
        else:
            print(f"    失败: 错误码 {result}")
            return False
    except Exception as e:
        print(f"    失败: {e}")
        return False

def check_ssl_handshake(host, port=443, timeout=10):
    """检查 SSL 握手"""
    print(f"[3] SSL 握手: {host}:{port}")
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                print(f"    成功: TLS {ssock.version()}")
                print(f"    加密: {ssock.cipher()[0]}")
                cert = ssock.getpeercert()
                if cert:
                    subject = dict(x[0] for x in cert['subject'])
                    print(f"    证书: {subject.get('commonName', 'N/A')}")
        return True
    except ssl.SSLError as e:
        print(f"    SSL 错误: {e}")
        return False
    except Exception as e:
        print(f"    失败: {e}")
        return False

def check_https_request(url):
    """检查完整的 HTTPS 请求"""
    print(f"[4] HTTPS 请求: {url}")
    try:
        response = urlopen(url, timeout=15)
        status = response.status
        print(f"    成功: HTTP {status}")
        print(f"    内容长度: {len(response.read(100))} bytes (已截断)")
        return True
    except URLError as e:
        print(f"    失败: {e.reason}")
        return False
    except Exception as e:
        print(f"    失败: {type(e).__name__}: {e}")
        return False

def check_dashscope_api_key():
    """检查 API Key 并尝试简单 API 调用"""
    print("\n[5] 通义千问 API 测试 (带重试)")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("    跳过: 未配置 API Key")
        return False
    
    print(f"    API Key: {api_key[:15]}...")
    
    import time
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
            import httpx
            
            client = httpx.Client(
                timeout=30.0,
                verify=True,
                http2=False
            )
            
            llm = ChatTongyi(
                model="qwen3-max",
                api_key=api_key,
                temperature=0.7,
                http_client=client
            )
            
            response = llm.invoke("hi")
            text = response.content.replace('\n', ' ')[:50]
            print(f"    尝试 {attempt}/{max_retries}: 成功")
            print(f"    响应: {text}...")
            return True
            
        except Exception as e:
            print(f"    尝试 {attempt}/{max_retries}: 失败 - {type(e).__name__}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"      等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                print(f"    最终错误: {e}")
    
    return False

def main():
    print("=" * 60)
    print("网络和 SSL 连接诊断")
    print("=" * 60)
    
    host = "dashscope.aliyuncs.com"
    results = []
    
    print()
    results.append(("DNS 解析", check_dns(host)))
    print()
    results.append(("TCP 连接", check_tcp_connect(host)))
    print()
    results.append(("SSL 握手", check_ssl_handshake(host)))
    print()
    results.append(("HTTPS 请求", check_https_request(f"https://{host}/")))
    print()
    results.append(("API 调用", check_dashscope_api_key()))
    
    print("\n" + "=" * 60)
    print("诊断结果汇总")
    print("=" * 60)
    
    for name, ok in results:
        status = "通过" if ok else "失败"
        print(f"  {name}: {status}")
    
    all_ok = all(ok for _, ok in results)
    
    print()
    if all_ok:
        print("所有检查通过！网络连接正常。")
        print("如果仍有 SSL 错误，可能是偶发网络波动，建议重试。")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"以下检查失败: {', '.join(failed)}")
        print()
        print("建议:")
        print("  1. 检查网络连接是否稳定")
        print("  2. 检查是否有代理或防火墙干扰")
        print("  3. 尝试切换网络 (如手机热点)")
        print("  4. 稍后重试，可能是服务端临时问题")
    
    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\n\n诊断已取消")
        sys.exit(1)
