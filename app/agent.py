#!/usr/bin/env python3
"""
基于Agno的MCP客户端
连接到数据分析服务端并实现数据分析功能
"""

import asyncio
from config import system_prompt, mcp_url, model_name
from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.tools.mcp import MCPTools


async def analysis_agent(message: str) -> None:
    """
    运行数据分析智能体
    Args:
        message: 用户的查询消息
    """
    async with MCPTools(
            transport="sse",
            url=mcp_url
    ) as data_analysis_server:
        # 创建智能体，使用LMStudio模型（参考data_analyzer_agent.py）并连接到MCP服务端
        agent = Agent(
            model=LMStudio(id=model_name),
            tools=[data_analysis_server],
            markdown=True,
            # 系统提示词：指导智能体如何使用工具进行数据分析
            instructions=system_prompt)

        # 执行用户请求
        await agent.aprint_response(input=message, stream=True)


def main():
    """主函数，处理事件循环问题"""
    # 测试查询
    test_queries = [
        "请加载iris数据集并告诉我它包含哪些列",
        "分析iris数据集中各特征之间的相关性",
        "告诉我iris数据集的基础统计信息",
        "筛选出sepal length大于5.0的数据并显示前5行",
        "按target分组，计算sepal length的平均值"
    ]

    print("数据分析智能体MCP客户端")
    print("=" * 40)
    print("支持的查询示例:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    print("=" * 40)

    # 用户输入查询
    user_input = input("\n请输入您的数据分析请求 (或输入数字选择示例): ")

    # 处理用户输入
    if user_input.isdigit() and 1 <= int(user_input) <= len(test_queries):
        query = test_queries[int(user_input) - 1]
    else:
        query = user_input

    print(f"\n正在处理请求: {query}")
    print("-" * 40)

    # 运行智能体（正确处理事件循环）
    try:
        # 检查是否已存在事件循环
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有正在运行的事件循环，创建一个新的
            asyncio.run(analysis_agent(query))
        else:
            # 已有事件循环在运行，使用它来执行协程
            loop.run_until_complete(analysis_agent(query))
    except KeyboardInterrupt:
        print("\n\n已取消操作")
    except Exception as e:
        print(f"\n执行出错: {e}")
        # 打印详细的错误信息
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
