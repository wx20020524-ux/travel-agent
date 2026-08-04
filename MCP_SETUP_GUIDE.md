# 高德地图 MCP 服务连接指南

## 📋 概述

本指南将帮助你完成阿里百炼高德地图 MCP 服务的配置和连接。

---

## 🚀 完整配置步骤

### 第一步：登录阿里百炼控制台

1. 访问 **https://dashscope.console.aliyun.com/**
2. 使用你的阿里云账号登录
3. 如果没有账号，需要先注册阿里云账号

### 第二步：获取 API Key

1. 在控制台左侧菜单中找到 **"API-KEY 管理"**
2. 点击 **"创建新的 API-KEY"** 按钮
3. 填写相关信息并创建
4. 复制生成的 API Key（格式：`sk-xxxxxxxxxx`）

### 第三步：开通高德地图 MCP 服务 ⭐（关键步骤）

1. 在阿里百炼控制台中，找到 **"模型服务"** 或 **"广场"** 或 **"服务市场"**
2. 搜索 **"高德地图"** 或 **"amap-maps"**
3. 找到 **"高德地图 MCP 服务"** 或类似名称的服务
4. 点击 **"开通"** 或 **"立即使用"** 按钮
5. 按照提示完成服务开通流程

**注意**：这是最关键的一步！你的 API Key 可能已经可以调用通义千问，但需要单独开通高德地图 MCP 服务权限。

### 第四步：配置项目

1. 在项目根目录找到 `.env` 文件
2. 打开文件，确保内容如下：
```env
DASHSCOPE_API_KEY=你的API_Key_在这里
```
3. 保存文件

**注意**：不要在 API Key 前后加引号或空格！

### 第五步：验证连接

#### 方法一：使用项目提供的测试脚本

在项目根目录运行：
```bash
python verify_api.py
```

#### 方法二：直接测试 MCP 连接

创建一个简单的测试脚本 `test_mcp_connection.py`：
```python
import asyncio
from config import CONFIG
from mcp_client import McpClientManager

async def test():
    manager = McpClientManager()
    try:
        tools = await manager.get_all_tools()
        print(f"成功连接！获取到 {len(tools)} 个工具")
        for tool in tools:
            print(f"  - {tool.name}")
    except Exception as e:
        print(f"连接失败: {e}")

asyncio.run(test())
```

然后运行：
```bash
python test_mcp_connection.py
```

### 第六步：启动应用

在项目根目录运行：
```bash
streamlit run app.py
```

然后在浏览器中打开 http://localhost:8501

---

## 🔍 常见问题排查

### 问题 1：500 Internal Server Error

**症状**：连接 MCP 时返回 500 错误

**解决方案**：
- 确认已在阿里百炼控制台开通高德地图 MCP 服务
- 检查 API Key 是否有足够的权限
- 尝试重新生成 API Key

### 问题 2：401 Unauthorized

**症状**：认证失败

**解决方案**：
- 检查 API Key 是否正确
- 确认 API Key 没有过期
- 检查 .env 文件格式是否正确

### 问题 3：403 Forbidden

**症状**：权限不足

**解决方案**：
- 在阿里百炼控制台确认已开通高德地图服务
- 检查 API Key 的权限范围设置

### 问题 4：找不到 MCP 服务入口

**症状**：在控制台找不到高德地图 MCP 服务

**解决方案**：
- 尝试搜索 "amap" 或 "高德"
- 查看阿里百炼官方文档
- 联系阿里云客服确认服务是否可用

---

## 📚 参考资料

- 阿里百炼官方文档：https://help.aliyun.com/zh/dashscope/
- 高德地图开放平台：https://lbs.amap.com/
- MCP 协议文档：https://modelcontextprotocol.io/

---

## 🆘 获取帮助

如果按照以上步骤仍然无法解决问题：

1. 查看阿里百炼控制台的错误日志
2. 检查阿里云账号是否有欠费
3. 联系阿里云技术支持
4. 在项目 Issues 中提问（如果有）

---

## ✅ 配置检查清单

- [ ] 已注册阿里云账号
- [ ] 已登录阿里百炼控制台
- [ ] 已创建 API Key
- [ ] 已开通高德地图 MCP 服务 ⭐
- [ ] 已将 API Key 配置到 .env 文件
- [ ] API Key 格式正确（以 sk- 开头）
- [ ] 已运行测试脚本验证连接
- [ ] Streamlit 应用可以正常启动

按照以上步骤完成后，你的智能旅行助手应该就可以正常使用高德地图 MCP 服务了！
