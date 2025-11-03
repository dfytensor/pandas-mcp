#!/usr/bin/env python3
"""
简单的测试脚本，用于验证服务器是否能正常工作
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_server():
    """测试服务器连接"""
    # 配置服务器参数
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__), 'server.py')],
        cwd=os.path.dirname(__file__)
    )

    try:
        # 创建客户端连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化会话
                await session.initialize()
                print("✓ 服务器连接成功")

                # 列出可用工具
                tools_response = await session.list_tools()
                print(f"✓ 可用工具: {[tool.name for tool in tools_response.tools]}")

                # 测试加载数据
                sample_data = """name,age,salary,department
Alice,30,50000,Engineering
Bob,25,45000,Marketing"""

                load_result = await session.call_tool(
                    "load_dataset",
                    {"data_content": sample_data, "data_type": "csv"}
                )
                print(f"✓ 数据加载结果: {load_result}")

                print("✓ 所有测试通过!")
                return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("测试 Pandas 积木 MCP 服务器...")
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)