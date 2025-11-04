# 数据分析MCP服务

本项目提供基于MCP协议的数据分析服务，包括服务端和客户端两部分。

## 文件说明

- [server_mcp.py](file:///D:/pandas-mcp/server_mcp.py) - 数据分析MCP服务端，使用HTTP协议
- [agent_mcp.py](file:///D:/pandas-mcp/agent_mcp.py) - 数据分析智能体客户端
- [data_analyzer_agent.py](file:///D:/pandas-mcp/data_analyzer_agent.py) - 原始数据分析智能体
- [test_sklearn_datasets.py](file:///D:/pandas-mcp/test_sklearn_datasets.py) - sklearn数据集测试

## 启动服务端

```bash
python server_mcp.py
```

服务端将启动在默认端口（通常是8000），提供以下功能：
- load_dataset: 加载sklearn标准数据集
- get_dataset_info: 获取数据集信息
- get_basic_statistics: 获取基础统计信息
- get_correlation_matrix: 获取相关性矩阵
- filter_data: 数据过滤
- group_by_analysis: 分组聚合分析
- get_column_unique_values: 获取列的唯一值
- get_data_summary: 获取数据综合摘要

## 运行客户端

```bash
python agent_mcp.py
```

客户端将连接到服务端，并允许用户通过自然语言进行数据分析。

## 支持的数据集

- iris (鸢尾花数据集)
- wine (葡萄酒数据集)
- breast_cancer (乳腺癌数据集)
- diabetes (糖尿病数据集)
- california_housing (加州房价数据集)

## 功能示例

客户端支持以下类型的查询：
1. "请加载iris数据集并告诉我它包含哪些列"
2. "分析iris数据集中各特征之间的相关性"
3. "告诉我iris数据集的基础统计信息"
4. "筛选出sepal length大于5.0的数据并显示前5行"
5. "按target分组，计算sepal length的平均值"
6. "查看petal width列的唯一值分布"
7. "给我一份数据的综合分析报告"

## 工作原理

1. 客户端使用Agno框架和LMStudio模型理解用户请求
2. 通过MCP协议调用服务端提供的工具
3. 服务端执行具体的数据分析操作
4. 结果通过HTTP协议返回给客户端
5. 客户端将结果以自然语言形式呈现给用户

## 特色功能

- **向量知识库**：客户端集成了向量知识库，能够参考数据分析指南来更好地理解用户请求
- **自然语言交互**：用户可以用自然语言描述分析需求，无需掌握特定语法
- **丰富的分析工具**：服务端提供多种数据分析工具，涵盖从基础统计到高级分组聚合的各类操作
- **错误处理机制**：完善的错误处理和提示机制，帮助用户更好地使用系统
- **事件循环兼容**：修复了事件循环相关的问题，确保在各种环境下都能正常运行
- **协议更新**：使用推荐的streamable-http协议替代已弃用的SSE协议