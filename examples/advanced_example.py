"""
高级使用示例
演示更复杂的数据分析流程，包括数据清洗、转换和可视化
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    """高级示例函数"""
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

                # 加载带有缺失值的数据
                sample_data = """name,age,salary,department,join_date
Alice,30,50000,Engineering,2020-01-15
Bob,25,,Marketing,2021-03-22
Charlie,35,60000,Engineering,2019-07-10
David,28,48000,,2022-05-30
Eve,32,55000,Sales,2020-11-05
Frank,29,52000,Engineering,
Grace,,47000,Marketing,2021-09-18"""

                print("\n1. 加载原始数据...")
                load_result = await session.call_tool(
                    "load_dataset",
                    {"data_content": sample_data, "data_type": "csv"}
                )
                print(f"数据加载结果: {load_result}")
                
                dataset_id = load_result.result[0]['dataset_id']
                print(f"数据集ID: {dataset_id}")

                # 执行完整的数据处理流水线
                print("\n2. 执行数据处理流水线...")
                pipeline = [
                    {
                        "type": "clean",
                        "name": "数据清洗",
                        "params": {
                            "operations": [
                                {"method": "fillna", "value": 0, "columns": ["salary"]},
                                {"method": "fillna", "value": "Unknown", "columns": ["department"]},
                                {"method": "drop_duplicates"}
                            ]
                        }
                    },
                    {
                        "type": "transform",
                        "name": "数据转换",
                        "params": {
                            "operations": [
                                {"method": "create_column", "name": "salary_level", 
                                 "formula": "df['salary'].apply(lambda x: 'High' if x > 50000 else 'Low')"},
                                {"method": "select", "columns": ["name", "age", "salary", "department", "salary_level"]}
                            ]
                        }
                    },
                    {
                        "type": "analyze",
                        "name": "分组分析",
                        "params": {
                            "analysis_type": "groupby",
                            "groupby_columns": ["department"],
                            "agg_functions": {"salary": ["mean", "count"]}
                        }
                    }
                ]

                analysis_result = await session.call_tool(
                    "execute_analysis_pipeline",
                    {"dataset_id": dataset_id, "blocks_config": pipeline}
                )
                print(f"分析结果: {analysis_result}")

                print("\n✓ 高级示例执行完成!")

    except Exception as e:
        print(f"✗ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Pandas积木MCP高级使用示例")
    print("=" * 30)
    asyncio.run(main())