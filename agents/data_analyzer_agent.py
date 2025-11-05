#!/usr/bin/env python3
"""
基于Agno框架的智能数据分析智能体
与Pandas积木MCP集成，实现对CSV文件的智能分析和处理
"""

import os
import tempfile
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.openai import OpenAIEmbedder

# 导入pandas-mcp服务器中的引擎类
from server import PandasBlocksEngine

# 尝试导入sklearn数据集
try:
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.datasets import fetch_california_housing
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataAnalyzerAgent:
    """智能数据分析智能体"""
    
    def __init__(self):
        """初始化智能体"""
        # 初始化知识库
        self.knowledge = Knowledge(
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
        self._setup_knowledge()
        
        # 初始化模型
        self.model = LMStudio(id="qwen3-4b-thinking-2507")
        
        # 创建智能体，启用工具调用功能
        self.agent = Agent(
            model=self.model,
            knowledge=self.knowledge,
            # 启用RAG，在用户提示中添加来自知识库的参考信息
            add_knowledge_to_context=True,
            # 设置为False，因为智能体默认为search_knowledge=True
            search_knowledge=False,
            markdown=True,
            # 启用工具调用
            tools=[self.get_dataset_columns, self.get_dataset_shape, self.get_current_dataset_info],
        )
        
        # 初始化数据分析引擎
        self.engine = PandasBlocksEngine()
        
        # 上下文管理
        self.context = {
            "current_dataset": None,
            "datasets": {},
            "history": []
        }

    def _setup_knowledge(self):
        """设置知识库内容"""
        # 创建知识文本文件
        knowledge_content = """
# 数据分析智能体使用指南

## 基础概念

数据分析智能体是一个能够处理和分析CSV文件的强大工具。它基于Pandas积木系统，可以通过自然语言指令执行各种数据分析任务。

## 核心功能

### 1. 数据加载
- 支持从本地文件路径加载CSV数据
- 支持从URL加载数据
- 支持直接粘贴CSV格式文本
- 支持从sklearn加载标准数据集

### 2. 数据清洗
- 处理缺失值
- 删除重复项
- 列重命名
- 数据类型转换

### 3. 数据探索
- 描述性统计分析
- 相关性分析
- 数据分布查看
- 唯一值计数

### 4. 数据转换
- 数据筛选
- 列选择和排序
- 创建新列
- 数据透视表

### 5. 数据可视化
- 条形图
- 折线图
- 直方图
- 散点图

## 使用方法

### 加载数据
用户可以通过以下方式提供数据：
1. 提供本地CSV文件路径
2. 提供在线CSV文件URL
3. 直接粘贴CSV格式的文本数据
4. 从sklearn加载标准机器学习数据集（如boston、iris、wine等）

### 发起分析请求
用户可以直接使用自然语言描述想要进行的分析，例如：
- "帮我看看这个数据的基本情况"
- "分析年龄和收入之间的关系"
- "找出销售额最高的前10个产品"

## 积木类型详解

### 1. clean (数据清洗)
操作选项：
- fillna: 填充空值
- drop_duplicates: 删除重复行
- drop_columns: 删除指定列
- rename_columns: 重命名列
- correct_types: 更正数据类型

### 2. filter (数据过滤)
- 使用条件表达式筛选数据
- 支持复杂的查询语句

### 3. transform (数据转换)
- select: 选择特定列
- sort: 数据排序
- create_column: 创建新列
- pivot: 数据透视

### 4. analyze (数据分析)
- basic: 基础统计信息
- correlation: 相关性分析
- value_counts: 值计数

### 5. groupby (分组聚合)
- 按指定列分组
- 应用聚合函数

### 6. visualize (可视化)
- bar: 条形图
- line: 折线图
- hist: 直方图
- scatter: 散点图

## 最佳实践

1. 在进行复杂分析之前，先查看数据基本信息
2. 注意处理缺失值和异常值
3. 对分类变量和数值变量采用不同的分析方法
4. 使用适当的可视化方式展示分析结果
5. 保留分析历史以便后续参考

## 工具使用说明

智能体可以使用以下工具来获取数据信息：

1. get_current_dataset_info(): 获取当前数据集的完整信息
2. get_dataset_columns(): 获取当前数据集的列信息
3. get_dataset_shape(): 获取当前数据集的形状信息

当用户询问关于数据集结构的问题时，智能体会自动调用相应工具获取真实数据信息。
        """
        
        # 写入知识文件
        with open("data_analysis_knowledge.txt", "w", encoding="utf-8") as f:
            f.write(knowledge_content)

        # 将知识添加到知识库
        self.knowledge.add_content(
            name="Data Analysis Guide",
            path="data_analysis_knowledge.txt",
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """
你是数据分析专家助手，专门帮助用户分析和处理CSV数据文件。
你的能力包括：

1. 理解用户的自然语言数据分析需求
2. 将需求转换为具体的数据处理步骤
3. 执行数据分析操作并解释结果
4. 生成可视化图表来展示数据洞察

当你收到用户的数据分析请求时，请按照以下步骤操作：

1. 首先理解用户的需求和提供的数据
2. 根据需求设计合适的分析流程
3. 使用Pandas积木引擎执行分析
4. 清晰地向用户解释分析结果

注意事项：
- 始终确保在执行操作前数据已经正确加载
- 如果用户没有明确指定要分析哪个数据集，则使用最近一次加载的数据
- 在执行每一步操作前，简要说明你将要做什么
- 以易于理解的方式呈现分析结果，避免过多的专业术语
- 如果遇到错误，清晰地解释问题所在并提出解决方案
- 当用户询问数据集结构信息时，使用工具自动获取真实数据

请记住，你的目标是帮助用户从数据中获得有价值的洞察，而不仅仅是执行技术操作。
        """

    def load_data(self, data_source: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        加载数据
        
        Args:
            data_source: 数据源，可以是文件路径、URL、CSV文本或sklearn数据集名
            dataset_name: 可选的数据集名称
            
        Returns:
            加载结果字典
        """
        # 更新上下文
        self.context["history"].append({
            "action": "load_data",
            "data_source": data_source,
            "dataset_name": dataset_name
        })
        
        # 检查是否是从sklearn加载数据集
        if SKLEARN_AVAILABLE and data_source in ['iris', 'wine', 'breast_cancer', 'diabetes', 'california_housing']:
            result = self._load_sklearn_dataset(data_source, dataset_name)
        else:
            # 调用引擎加载数据
            result = self.engine.load_data(data_source, "csv", dataset_name)
        
        # 如果加载成功，更新当前数据集
        if result.get("success"):
            self.context["current_dataset"] = result["dataset_id"]
            self.context["datasets"][result["dataset_id"]] = {
                "name": result["dataset_id"],
                "shape": result["data_shape"],
                "columns": result["columns"]
            }
        
        return result

    def _load_sklearn_dataset(self, dataset_name: str, custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        从sklearn加载标准数据集
        
        Args:
            dataset_name: 数据集名称
            custom_name: 自定义数据集名称
            
        Returns:
            加载结果字典
        """
        try:
            # 根据数据集名称加载对应的数据
            if dataset_name == 'iris':
                data = load_iris()
            elif dataset_name == 'wine':
                data = load_wine()
            elif dataset_name == 'breast_cancer':
                data = load_breast_cancer()
            elif dataset_name == 'diabetes':
                data = load_diabetes()
            elif dataset_name == 'california_housing':
                data = fetch_california_housing()
            
            # 构建DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            if hasattr(data, 'target'):
                # 添加目标变量列
                target_name = 'target'
                if hasattr(data, 'target_names') and dataset_name != 'california_housing':
                    # 对于分类数据集，添加目标名称映射
                    target_name = 'target_name'
                    df[target_name] = [data.target_names[i] for i in data.target]
                else:
                    # 对于回归数据集或其他情况，直接使用target值
                    df['target'] = data.target
            
            # 生成数据集ID
            dataset_id = custom_name if custom_name else dataset_name
            
            # 存储到引擎中
            self.engine.datasets[dataset_id] = df
            self.engine.current_dataset_id = dataset_id
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "data_shape": df.shape,
                "columns": list(df.columns),
                "message": f"成功加载sklearn数据集: {dataset_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"加载sklearn数据集失败: {str(e)}"
            }

    def execute_analysis(self, user_request: str) -> Dict[str, Any]:
        """
        执行数据分析
        
        Args:
            user_request: 用户的分析请求
            
        Returns:
            分析结果字典
        """
        # 更新上下文
        self.context["history"].append({
            "action": "execute_analysis",
            "request": user_request
        })
        
        # 构建分析管道配置
        blocks_config = self._generate_blocks_config(user_request)
        
        # 执行分析
        if self.context["current_dataset"]:
            result = self.engine.execute_pipeline(blocks_config, self.context["current_dataset"])
            return result
        else:
            return {"error": "请先加载数据"}

    def _generate_blocks_config(self, user_request: str) -> List[Dict]:
        """
        根据用户请求生成积木配置
        这里使用一个简化的方法，实际应用中可能需要更复杂的逻辑或LLM辅助
        
        Args:
            user_request: 用户请求
            
        Returns:
            积木配置列表
        """
        # 这里应该使用LLM来理解用户请求并生成相应的积木配置
        # 为了演示目的，我们根据关键词简单判断
        
        user_request_lower = user_request.lower()
        
        if "基本" in user_request_lower or "基础" in user_request_lower:
            # 基础分析
            return [
                {
                    "type": "clean",
                    "name": "基础清洗",
                    "params": {
                        "operations": [
                            {"method": "fillna", "value": 0}
                        ]
                    }
                },
                {
                    "type": "analyze",
                    "name": "基础分析",
                    "params": {
                        "analysis_type": "basic"
                    }
                }
            ]
        elif "相关性" in user_request_lower:
            # 相关性分析
            return [
                {
                    "type": "clean",
                    "name": "数据清洗",
                    "params": {
                        "operations": [
                            {"method": "fillna", "value": 0},
                            {"method": "drop_duplicates"}
                        ]
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
        elif "分组" in user_request_lower:
            # 分组聚合分析
            return [
                {
                    "type": "clean",
                    "name": "数据清洗",
                    "params": {
                        "operations": [
                            {"method": "fillna", "value": 0}
                        ]
                    }
                },
                {
                    "type": "groupby",
                    "name": "分组聚合",
                    "params": {
                        "group_columns": [],  # 实际使用时需要指定分组列
                        "agg_functions": {}   # 实际使用时需要指定聚合函数
                    }
                }
            ]
        elif "可视化" in user_request_lower or "图表" in user_request_lower:
            # 可视化分析
            return [
                {
                    "type": "clean",
                    "name": "数据清洗",
                    "params": {
                        "operations": [
                            {"method": "fillna", "value": 0}
                        ]
                    }
                },
                {
                    "type": "visualize",
                    "name": "数据可视化",
                    "params": {
                        "chart_type": "bar"
                    }
                }
            ]
        else:
            # 综合分析
            return [
                {
                    "type": "clean",
                    "name": "数据清洗",
                    "params": {
                        "operations": [
                            {"method": "fillna", "value": 0},
                            {"method": "drop_duplicates"}
                        ]
                    }
                },
                {
                    "type": "analyze",
                    "name": "综合分析",
                    "params": {
                        "analysis_type": "basic"
                    }
                }
            ]

    def get_context(self) -> Dict[str, Any]:
        """
        获取当前上下文信息
        
        Returns:
            当前上下文字典
        """
        return self.context

    def get_current_dataset_info(self) -> Dict[str, Any]:
        """
        获取当前数据集信息
        
        Returns:
            数据集信息字典
        """
        if not self.context["current_dataset"]:
            return {"error": "当前没有加载任何数据集"}
        
        dataset_id = self.context["current_dataset"]
        if dataset_id not in self.engine.datasets:
            return {"error": f"数据集 {dataset_id} 不存在"}
        
        df = self.engine.datasets[dataset_id]
        return {
            "dataset_id": dataset_id,
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head().to_dict('records') if not df.empty else {}
        }

    def chat(self, message: str) -> Any:
        """
        与智能体对话
        
        Args:
            message: 用户消息
            
        Returns:
            智能体响应
        """
        # 将系统提示与用户消息结合
        full_message = f"{self._get_system_prompt()}\n\n用户请求: {message}"
        return self.agent.print_response(full_message, stream=True)

    # 工具函数，供智能体调用
    def get_dataset_columns(self) -> Dict[str, Any]:
        """
        获取当前数据集的列信息
        
        Returns:
            列信息字典
        """
        info = self.get_current_dataset_info()
        if "error" in info:
            return info
        
        return {
            "dataset_id": info["dataset_id"],
            "columns": info["columns"],
            "data_types": info["data_types"]
        }

    def get_dataset_shape(self) -> Dict[str, Any]:
        """
        获取当前数据集的形状信息
        
        Returns:
            形状信息字典
        """
        info = self.get_current_dataset_info()
        if "error" in info:
            return info
        
        return {
            "dataset_id": info["dataset_id"],
            "shape": info["shape"]
        }