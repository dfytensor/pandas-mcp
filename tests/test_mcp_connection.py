#!/usr/bin/env python3
"""
测试MCP服务连接
"""

import asyncio
import httpx


async def test_mcp_server():
    """测试MCP服务是否正常运行"""
    url = "http://127.0.0.1:8000/sse"
    
    try:
        async with httpx.AsyncClient() as client:
            # 测试服务器是否响应
            response = await client.get(f"{url}/")
            print(f"服务器响应状态: {response.status_code}")
            print(f"服务器响应内容: {response.text[:200]}...")
            
    except Exception as e:
        print(f"连接服务器时出错: {e}")
        print("请确保已运行: python server_mcp.py")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())