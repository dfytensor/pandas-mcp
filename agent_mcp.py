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
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder


async def run_data_analysis_agent(message: str) -> None:
    """
    运行数据分析智能体
    
    Args:
        message: 用户的查询消息
    """
    # 初始化知识库，参考data_analyzer_agent.py的配置
    knowledge = Knowledge(
        vector_db=LanceDb(
            table_name="data_analysis_knowledge",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=OpenAIEmbedder(
                id="text-embedding-nomic-embed-text-v1.5",
                base_url="http://127.0.0.1:1234/v1",
                api_key="sk-nomic-api-key",
                dimensions=768
            ),
        ),
    )
    
    # 添加知识内容
    knowledge_content = """
# 数据分析智能体使用指南

## 基础概念

数据分析智能体是一个能够处理和分析数据集的强大工具。它基于MCP协议，可以通过自然语言指令执行各种数据分析任务。

## 核心功能

### 1. 数据加载
- 支持加载sklearn标准数据集 (iris, wine, breast_cancer, diabetes, california_housing)

### 2. 数据探索
- 描述性统计分析
- 相关性分析
- 数据分布查看
- 唯一值计数

### 3. 数据过滤
- 使用条件表达式筛选数据
- 支持多种比较操作符 (==, !=, >, <, >=, <=)

### 4. 分组聚合分析
- 按指定列分组
- 应用聚合函数 (mean, sum, count, min, max)

## 使用方法

### 加载数据
用户可以通过以下方式加载数据：
- "加载iris数据集"
- "请加载wine数据集进行分析"

### 发起分析请求
用户可以直接使用自然语言描述想要进行的分析，例如：
- "帮我看看这个数据的基本情况"
- "分析各特征之间的相关性"
- "筛选出满足条件的数据"
- "按类别分组计算平均值"

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
    """
    
    # 将知识添加到知识库（使用异步方式）
    await knowledge.add_content_async(
        name="Data Analysis Guide",
        content=knowledge_content,
    )
    
    # 使用streamable-http替代已弃用的SSE
    async with MCPTools(
        transport="streamable-http", 
        url="http://127.0.0.1:8000"  # 默认FastMCP HTTP地址
    ) as data_analysis_server:
        # 创建智能体，使用LMStudio模型（参考data_analyzer_agent.py）并连接到MCP服务端
        agent = Agent(
            model=LMStudio(id="qwen3-4b-thinking-2507"),
            tools=[data_analysis_server],
            knowledge=knowledge,
            # 启用RAG，在用户提示中添加来自知识库的参考信息
            add_knowledge_to_context=True,
            markdown=True,
            # 启用工具调用的系统提示
            instructions="""
你是数据分析专家助手，专门帮助用户分析和处理数据集。
你可以使用以下工具执行数据分析任务：

1. load_dataset: 加载sklearn标准数据集
2. get_dataset_info: 获取当前数据集信息
3. get_basic_statistics: 获取基础统计信息
4. get_correlation_matrix: 获取相关性矩阵
5. filter_data: 根据条件过滤数据
6. group_by_analysis: 分组聚合分析

当你收到用户的数据分析请求时，请按照以下步骤操作：

1. 首先理解用户的需求
2. 如果需要，加载适当的数据集
3. 使用合适的工具执行分析
4. 清晰地向用户解释分析结果

注意事项：
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