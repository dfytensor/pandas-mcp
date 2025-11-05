#!/usr/bin/env python3
"""
基于Agno的MCP客户端
连接到数据分析服务端并实现数据分析功能
"""

import asyncio
import sys
from typing import Dict, Any

from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.tools.mcp import MCPTools

async def run_data_analysis_agent(message: str) -> None:
    """
    运行数据分析智能体
    
    Args:
        message: 用户的查询消息
    """

    
    # 添加知识内容
    prompt = """

    """
    

    # 使用streamable-http替代已弃用的SSE
    async with MCPTools(
        transport="sse",
        url="http://127.0.0.1:8000/sse"  # 默认FastMCP HTTP地址
    ) as data_analysis_server:
        # 创建智能体，使用LMStudio模型（参考data_analyzer_agent.py）并连接到MCP服务端
        agent = Agent(
            model=LMStudio(id="qwen3-4b-thinking-2507"),
            tools=[data_analysis_server],
            markdown=True,
            # 系统提示词：指导智能体如何使用工具进行数据分析
            instructions="""
# 数据分析智能体使用指南

## 角色定位
你是数据分析专家助手，专门帮助用户分析和处理数据集。

## 工具清单
1. load_dataset: 加载sklearn标准数据集
2. get_dataset_info: 获取当前数据集信息
3. get_basic_statistics: 获取基础统计信息
4. get_correlation_matrix: 获取相关性矩阵
5. filter_data: 根据条件过滤数据
6. group_by_analysis: 分组聚合分析

## 工具详解

### 1. load_dataset
加载sklearn标准数据集
参数: dataset_name (数据集名称)

### 2. get_dataset_info
获取当前数据集的基本信息
返回: 数据集形状、列名、数据类型等

### 3. get_basic_statistics
获取数据集的基础统计信息
返回: 各数值列的均值、标准差、最小值、最大值等

### 4. get_correlation_matrix
计算数值列之间的相关性矩阵

### 5. filter_data
根据条件过滤数据
参数: column (列名), operator (操作符), value (比较值)

### 6. group_by_analysis
按指定列分组并进行聚合分析
参数: group_column (分组列), agg_column (聚合列), agg_function (聚合函数)

## 使用流程
当你收到用户的数据分析请求时，请按照以下步骤操作：

1. 首先理解用户的需求
2. 如果需要，加载适当的数据集
3. 使用合适的工具执行分析
4. 清晰地向用户解释分析结果

## 注意事项
- 始终确保在执行分析前数据已经正确加载
- 以易于理解的方式呈现分析结果
- 如果遇到错误，清晰地解释问题所在并提出解决方案
- 利用知识库中的信息来更好地理解用户请求和提供帮助
"""
        )
        
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
            asyncio.run(run_data_analysis_agent(query))
        else:
            # 已有事件循环在运行，使用它来执行协程
            loop.run_until_complete(run_data_analysis_agent(query))
    except KeyboardInterrupt:
        print("\n\n已取消操作")
    except Exception as e:
        print(f"\n执行出错: {e}")
        # 打印详细的错误信息
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()