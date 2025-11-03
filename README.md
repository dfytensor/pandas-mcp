# Pandas积木MCP工具

让LLM通过简单的积木配置实现专业级数据分析能力。

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
python server.py
```

### 配置Claude Desktop
将配置文件添加到Claude Desktop的MCP服务器配置中。

## 使用示例

### 1. 直接运行测试
```bash
# 运行简单的测试脚本验证服务器
python test_server.py
```

### 2. 运行完整示例
```bash
# 运行完整示例（需要先启动服务器）
python examples/example_usage.py
```

### 3. 手动测试
你可以手动启动服务器，然后使用任何MCP兼容的客户端连接到它。

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
   python server.py
   ```

3. 确保示例脚本中的路径配置正确。

## 许可证

MIT License