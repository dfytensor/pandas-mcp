"""
Pandas积木MCP使用示例
演示如何使用Pandas积木进行数据分析
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    """主示例函数"""
    # 配置服务器参数
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'server.py')],
        cwd=os.path.dirname(os.path.dirname(__file__))
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
Bob,25,45000,Marketing
Charlie,35,60000,Engineering
David,28,48000,Marketing
Eve,32,55000,Sales"""

                print("\n1. 加载数据...")
                load_result = await session.call_tool(
                    "load_dataset",
                    {"data_content": sample_data, "data_type": "csv"}
                )
                print(f"数据加载结果: {load_result}")
                
                dataset_id = load_result.result[0]['dataset_id']
                print(f"数据集ID: {dataset_id}")

                # 执行数据分析流水线
                print("\n2. 执行数据分析流水线...")
                pipeline = [
                    {
                        "type": "analyze",
                        "name": "基础统计分析",
                        "params": {
                            "analysis_type": "basic"
                        }
                    },
                    {
                        "type": "analyze",
                        "name": "相关性分析",
                        "params": {
                            "analysis_type": "correlation"
                        }
                    }
                ]

                analysis_result = await session.call_tool(
                    "execute_analysis_pipeline",
                    {"dataset_id": dataset_id, "blocks_config": pipeline}
                )
                print(f"分析结果: {analysis_result}")

                print("\n✓ 示例执行完成!")

    except Exception as e:
        print(f"✗ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Pandas积木MCP使用示例")
    print("=" * 30)
    asyncio.run(main())