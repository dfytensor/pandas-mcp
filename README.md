# Pandas积木MCP工具

让LLM通过简单的积木配置实现专业级数据分析能力。

## 项目结构

```
pandas-mcp/
├── agents/                 # 智能体相关模块
│   ├── data_analyzer_agent.py  # 智能数据分析智能体
│   └── agent_mcp.py           # MCP智能体接口
├── core/                   # 核心功能模块
│   ├── server.py              # 主服务器实现
│   └── server_mcp.py          # MCP协议服务器
├── docs/                   # 文档资料
│   ├── README.md              # 主要说明文档
│   ├── JOIN_FUNCTIONALITY.md  # 表关联功能说明
│   ├── README_MCP.md          # MCP集成说明
│   └── README_TEST.md         # 测试说明
├── examples/               # 使用示例
│   ├── example_usage.py       # 基础使用示例
│   └── advanced_example.py    # 高级使用示例
├── tests/                  # 测试文件
│   ├── test_agent.py
│   ├── test_join_functionality.py
│   ├── test_mcp_connection.py
│   ├── test_server.py
│   ├── test_sklearn_datasets.py
│   └── test_tool_calling.py
├── requirements.txt        # 项目依赖
└── install_and_test.py     # 安装和测试脚本
```

## 功能特性

- 🧩 **积木化操作**: 将复杂的数据操作分解为简单的积木块
- 🔄 **链式执行**: 支持多个积木的流水线执行
- 📊 **丰富的数据操作**: 清洗、转换、分析、可视化
- 🔧 **简单接口**: LLM只需提供数据和参数配置
- 📈 **可视化支持**: 自动生成图表和分析报告

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行MCP服务器
```bash
python core/server.py
```

### 配置Claude Desktop
将配置文件添加到Claude Desktop的MCP服务器配置中。

## 使用示例

### 1. 直接运行测试
```bash
# 运行简单的测试脚本验证服务器
python tests/test_server.py
```

### 2. 运行完整示例
```bash
# 运行基础示例（需要先启动服务器）
python examples/example_usage.py

# 运行高级示例
python examples/advanced_example.py
```

### 3. 手动测试
你可以手动启动服务器，然后使用任何MCP兼容的客户端连接到它。

## 智能数据分析智能体

我们新增了一个基于 Agno 框架的智能数据分析智能体，具备以下特点：

### 主要功能
1. **自然语言交互**: 用户可以用自然语言描述数据分析需求
2. **上下文管理**: 智能体维护分析过程的上下文状态
3. **自动积木配置**: 根据用户需求自动生成合适的分析流程
4. **知识增强**: 利用内置知识库提升分析质量

### 核心组件
- `DataAnalyzerAgent`: 主智能体类
- 知识库系统：包含数据分析最佳实践和指南
- 上下文管理系统：跟踪数据集和分析历史
- 自然语言接口：支持流式响应

### 使用方法
```python
from agents.data_analyzer_agent import DataAnalyzerAgent

# 创建智能体实例
analyzer = DataAnalyzerAgent()

# 加载数据
analyzer.load_data("path/to/data.csv", "my_dataset")

# 执行分析
analyzer.execute_analysis("请分析年龄和收入的关系")

# 与智能体对话
analyzer.chat("数据中有哪些列？")
```

## 积木类型说明

### 1. 数据清洗积木 (clean)
- 处理缺失值
- 删除重复项
- 列重命名
- 类型转换

### 2. 数据过滤积木 (filter)
- 条件过滤
- 查询过滤
- 列选择

### 3. 数据转换积木 (transform)
- 列创建
- 数据透视
- 排序操作

### 4. 数据分析积木 (analyze)
- 描述性统计
- 相关性分析
- 频数分析

### 5. 分组聚合积木 (groupby)
- 多列分组
- 多种聚合函数
- 结果重置索引

## LLM调用示例

```python
# 简单的数据分析流程
pipeline = [
    {
        "type": "clean",
        "params": {"operations": [{"method": "fillna", "value": 0}]}
    },
    {
        "type": "analyze",
        "params": {"analysis_type": "comprehensive"}
    }
]

result = execute_analysis_pipeline(pipeline)
```

## 高级功能

- **历史记录**: 跟踪所有分析操作
- **多数据集管理**: 支持同时处理多个数据集
- **错误处理**: 详细的错误信息和调试支持
- **扩展性**: 易于添加新的积木类型

## 故障排除

如果遇到连接问题，请检查：

1. 确保所有依赖已正确安装：
   ```bash
   pip install -r requirements.txt
   ```

2. 确保服务器路径正确，可以在项目根目录下运行：
   ```bash
   python core/server.py
   ```

3. 确保示例脚本中的路径配置正确。

## 许可证

MIT License